"""Training loop for the chess style model.

PK sampler: each batch is N_PLAYERS_PER_BATCH players × GAMES_PER_PLAYER games.
This yields plenty of within-class positives and between-class negatives for
supervised contrastive loss.

Eval: k-shot identification on held-out games (build a per-player prototype
from K reference games, classify Q query games to the nearest prototype).
Plus a separation metric: mean within-player cosine sim − mean between-player.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset import GameRecord, read_games
from .encode import N_INPUT_CHANNELS, moves_to_channels
from .loss import supcon_loss, variance_regularization
from .model import StyleModel

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    processed_dir: Path = Path("data/processed")
    out_dir: Path = Path("checkpoints")

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip: float = 1.0
    steps: int = 1000
    eval_every: int = 100
    log_every: int = 25

    # Batch shape
    n_players_per_batch: int = 8
    games_per_player: int = 4
    max_len: int = 128

    # Loss
    loss_mode: str = "ce"            # "ce", "supcon", or "ce+supcon"
    temperature: float = 0.1
    supcon_weight: float = 1.0
    variance_weight: float = 0.0     # VICReg-style anti-collapse (off when CE is on)
    target_std: float = 1.0

    # Model
    d_model: int = 256
    cnn_ch: int = 128
    cnn_blocks: int = 4
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # Data
    val_frac: float = 0.2
    min_games_per_player: int = 8

    # System
    device: str = "auto"
    seed: int = 0


# ---------------------------------------------------------------- helpers


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _game_to_input(g: GameRecord, max_len: int) -> np.ndarray:
    T = min(g.boards.shape[0], max_len)
    board = g.boards[:T].astype(np.float32)
    moves = moves_to_channels(g.moves_uci[:T]).astype(np.float32)
    return np.concatenate([board, moves], axis=1)  # (T, 24, 8, 8)


def games_to_padded_batch(
    games: list[GameRecord], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a list of games into padded (B, T, 24, 8, 8) + mask (B, T)."""
    xs = [_game_to_input(g, max_len) for g in games]
    lens = [x.shape[0] for x in xs]
    T_max = max(lens)
    B = len(xs)
    padded = np.zeros((B, T_max, N_INPUT_CHANNELS, 8, 8), dtype=np.float32)
    mask = np.ones((B, T_max), dtype=bool)
    for i, x in enumerate(xs):
        padded[i, : lens[i]] = x
        mask[i, : lens[i]] = False
    return torch.from_numpy(padded), torch.from_numpy(mask)


def split_games_by_player(
    processed_dir: Path,
    val_frac: float,
    min_games: int,
    seed: int = 0,
) -> tuple[dict[str, list[GameRecord]], dict[str, list[GameRecord]]]:
    rng = random.Random(seed)
    train: dict[str, list[GameRecord]] = {}
    val: dict[str, list[GameRecord]] = {}
    for path in sorted(processed_dir.glob("*.parquet")):
        if path.name.startswith("_"):
            continue
        pid = path.stem
        games = list(read_games(path))
        if len(games) < min_games:
            log.warning("skipping %s: only %d games (need >= %d)", pid, len(games), min_games)
            continue
        idxs = list(range(len(games)))
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(games) * val_frac)))
        val_set = set(idxs[:n_val])
        train[pid] = [g for i, g in enumerate(games) if i not in val_set]
        val[pid] = [g for i, g in enumerate(games) if i in val_set]
    return train, val


# ---------------------------------------------------------------- sampler


class PKSampler:
    """Sample N players × K games per batch.

    Returns labels in TWO spaces:
      - batch_labels: 0..N-1 (slot in this batch)  — fine for supcon
      - global_labels: 0..n_classes-1 (player id)  — required for CE
    """

    def __init__(
        self,
        games_by_player: dict[str, list[GameRecord]],
        n_players: int,
        k_per_player: int,
        max_len: int,
        seed: int = 0,
        player_to_label: dict[str, int] | None = None,
    ):
        self.games = games_by_player
        self.n = n_players
        self.k = k_per_player
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.player_ids = sorted(games_by_player.keys())
        self.player_to_label = player_to_label or {p: i for i, p in enumerate(self.player_ids)}

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        eligible = [p for p in self.player_ids if len(self.games[p]) >= self.k]
        chosen = self.rng.sample(eligible, min(self.n, len(eligible)))
        batch_games: list[GameRecord] = []
        batch_labels: list[int] = []
        global_labels: list[int] = []
        for slot, pid in enumerate(chosen):
            gid = self.player_to_label[pid]
            for g in self.rng.sample(self.games[pid], self.k):
                batch_games.append(g)
                batch_labels.append(slot)
                global_labels.append(gid)
        x, mask = games_to_padded_batch(batch_games, self.max_len)
        return (
            x,
            mask,
            torch.tensor(batch_labels, dtype=torch.long),
            torch.tensor(global_labels, dtype=torch.long),
            chosen,
        )


# ---------------------------------------------------------------- eval


@torch.no_grad()
def embed_games(
    model: StyleModel,
    games: list[GameRecord],
    device: torch.device,
    max_len: int,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    embs = []
    for start in range(0, len(games), batch_size):
        batch = games[start : start + batch_size]
        x, mask = games_to_padded_batch(batch, max_len)
        emb = model.encode(x.to(device), mask.to(device)).cpu().numpy()
        embs.append(emb)
    return np.concatenate(embs, axis=0) if embs else np.empty((0,))


def k_shot_eval(
    model: StyleModel,
    val_games: dict[str, list[GameRecord]],
    device: torch.device,
    cfg: TrainConfig,
    k_shot: int = 5,
    n_query: int = 10,
    n_trials: int = 30,
    max_eval_games_per_player: int = 60,
) -> dict:
    rng = random.Random(cfg.seed + 1)
    eligible = [p for p in val_games if len(val_games[p]) >= k_shot + 1]
    if len(eligible) < 2:
        return {"n_eligible": len(eligible), "k5_top1": 0.0, "k5_top5": 0.0,
                "within_cos": 0.0, "between_cos": 0.0, "separation": 0.0}

    # Cap embeddings per player so eval doesn't dominate runtime.
    embeddings = {}
    for p in eligible:
        gs = val_games[p]
        if len(gs) > max_eval_games_per_player:
            idx = rng.sample(range(len(gs)), max_eval_games_per_player)
            gs = [gs[i] for i in idx]
        embeddings[p] = embed_games(model, gs, device, cfg.max_len)

    # K-shot id: K reference games -> prototype, classify Q query games per player.
    correct1 = correct5 = total = 0
    for _ in range(n_trials):
        protos: dict[str, np.ndarray] = {}
        queries: dict[str, np.ndarray] = {}
        for pid in eligible:
            embs = embeddings[pid]
            n_take = k_shot + min(n_query, len(embs) - k_shot)
            idx = rng.sample(range(len(embs)), n_take)
            ref = embs[idx[:k_shot]]
            qry = embs[idx[k_shot:]]
            proto = ref.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-9)
            protos[pid] = proto
            queries[pid] = qry

        pids = list(protos.keys())
        P = np.stack([protos[p] for p in pids], axis=0)  # (P, D)
        for pid, qs in queries.items():
            sims = qs @ P.T  # (n_q, P)
            ranks = np.argsort(-sims, axis=1)
            true_idx = pids.index(pid)
            for r in ranks:
                total += 1
                if r[0] == true_idx:
                    correct1 += 1
                if true_idx in r[:5]:
                    correct5 += 1

    # Within / between cosine separation across the entire val embedding set.
    all_e = np.concatenate([embeddings[p] for p in eligible], axis=0)
    all_l = np.concatenate([np.full(len(embeddings[p]), i) for i, p in enumerate(eligible)])
    sim = all_e @ all_e.T
    same = all_l[:, None] == all_l[None, :]
    eye = np.eye(len(all_l), dtype=bool)
    within = float(sim[same & ~eye].mean()) if (same & ~eye).any() else 0.0
    between = float(sim[~same].mean()) if (~same).any() else 0.0

    return {
        "n_eligible": len(eligible),
        "k5_top1": correct1 / max(total, 1),
        "k5_top5": correct5 / max(total, 1),
        "within_cos": within,
        "between_cos": between,
        "separation": within - between,
    }


# ---------------------------------------------------------------- training


def train(cfg: TrainConfig) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    log.info("device: %s", device)

    train_games, val_games = split_games_by_player(
        cfg.processed_dir, cfg.val_frac, cfg.min_games_per_player, cfg.seed,
    )
    log.info("loaded %d players for training, %d for val", len(train_games), len(val_games))
    for pid in sorted(train_games):
        log.info("  %-22s train=%d  val=%d", pid, len(train_games[pid]), len(val_games[pid]))

    player_to_label = {p: i for i, p in enumerate(sorted(train_games.keys()))}
    n_classes = len(player_to_label)

    sampler = PKSampler(
        train_games,
        n_players=cfg.n_players_per_batch,
        k_per_player=cfg.games_per_player,
        max_len=cfg.max_len,
        seed=cfg.seed,
        player_to_label=player_to_label,
    )

    needs_classifier = "ce" in cfg.loss_mode
    model = StyleModel(
        d_model=cfg.d_model,
        cnn_ch=cfg.cnn_ch,
        cnn_blocks=cfg.cnn_blocks,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        n_classes=n_classes if needs_classifier else None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model: %d params", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step: int) -> float:
        if cfg.warmup_steps > 0 and step <= cfg.warmup_steps:
            return cfg.lr * step / max(1, cfg.warmup_steps)
        return cfg.lr

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_metric = -float("inf")
    best_path = cfg.out_dir / "best.pt"
    last_path = cfg.out_dir / "last.pt"

    t0 = time.time()
    running_loss = running_sup = running_var = 0.0
    running_n = 0
    running_ce = running_acc = 0.0
    for step in range(1, cfg.steps + 1):
        model.train()
        x, mask, y_batch, y_global, _ = sampler.sample()
        x = x.to(device)
        mask = mask.to(device)
        y_batch = y_batch.to(device)
        y_global = y_global.to(device)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if needs_classifier:
            raw, logits = model.classify(x, mask)
        else:
            raw = model.forward(x, mask)
            logits = None

        ce_loss = torch.tensor(0.0, device=device)
        acc = 0.0
        if "ce" in cfg.loss_mode and logits is not None:
            ce_loss = torch.nn.functional.cross_entropy(logits, y_global)
            acc = (logits.argmax(dim=-1) == y_global).float().mean().item()

        sup_loss = torch.tensor(0.0, device=device)
        if "supcon" in cfg.loss_mode:
            emb = torch.nn.functional.normalize(raw, dim=-1)
            sup_loss = supcon_loss(emb, y_batch, temperature=cfg.temperature, already_normalized=True)

        var_loss = torch.tensor(0.0, device=device)
        if cfg.variance_weight > 0:
            var_loss = variance_regularization(raw, target_std=cfg.target_std)

        loss = ce_loss + cfg.supcon_weight * sup_loss + cfg.variance_weight * var_loss
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        running_loss += loss.item()
        running_ce += ce_loss.item()
        running_sup += sup_loss.item()
        running_var += var_loss.item()
        running_acc += acc
        running_n += 1

        if step % cfg.log_every == 0:
            n = running_n
            log.info(
                "step %4d/%-4d  loss=%.4f  ce=%.4f acc=%.3f  sup=%.4f var=%.4f  (%.1fs)",
                step, cfg.steps,
                running_loss / n, running_ce / n, running_acc / n,
                running_sup / n, running_var / n, time.time() - t0,
            )
            history.append({
                "step": step, "loss": running_loss / n,
                "ce": running_ce / n, "acc": running_acc / n,
                "sup": running_sup / n, "var": running_var / n,
            })
            running_loss = running_ce = running_sup = running_var = running_acc = 0.0
            running_n = 0

        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = k_shot_eval(model, val_games, device, cfg)
            log.info(
                "  eval @ %d:  top1=%.3f  top5=%.3f  within=%.3f  between=%.3f  sep=%+.3f  (n=%d)",
                step, metrics["k5_top1"], metrics["k5_top5"],
                metrics["within_cos"], metrics["between_cos"],
                metrics["separation"], metrics["n_eligible"],
            )
            history.append({"step": step, **metrics})
            if metrics["k5_top1"] > best_metric:
                best_metric = metrics["k5_top1"]
                torch.save(
                    {"state_dict": model.state_dict(), "cfg": asdict(cfg),
                     "player_to_label": player_to_label, "n_classes": n_classes,
                     "step": step, "metric": best_metric},
                    best_path,
                )
                log.info("    new best top1=%.3f -> %s", best_metric, best_path.name)

    torch.save(
        {"state_dict": model.state_dict(), "cfg": asdict(cfg),
         "player_to_label": player_to_label, "n_classes": n_classes,
         "step": cfg.steps},
        last_path,
    )
    (cfg.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("done in %.1fs. best top1=%.3f", time.time() - t0, best_metric)
    return {"best_metric": best_metric, "history": history, "n_train_players": len(train_games)}
