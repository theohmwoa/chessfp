# chessfp — Chess Fingerprint

Identify a chess.com player from their moves alone, and tell anyone *which pro they play like*.

A behavioral-stylometry model trained on a curated set of chess.com pros and streamers (Magnus, Hikaru, Naroditsky, Levy/GothamChess, Firouzja, Anna Cramling, Ben Finegold, etc.). End-to-end pipeline: scrape chess.com → encode boards + moves into tensors → train a CNN-over-board → transformer-over-game model → embedding-based inference.

> **Status: in-progress / educational.** The data pipeline and architecture are working end-to-end. Training accuracy is currently ~2.5× random baseline on 14 players (tracked best at ~21% top-1 train accuracy / ~13% val top-1 vs. 7% random) and climbing. This is a portfolio/learning project — see [`LEARNINGS.md`](LEARNINGS.md) for the engineering journey including the failed attempts.

## Why this exists

Behavioral stylometry in chess has been shown to work very well — McIlroy-Young et al. (NeurIPS 2021) hit 98% accuracy across 2,500 Lichess players using transformer architectures. What hasn't been built is a polished, public-facing tool focused on the **chess.com pro/streamer ecosystem** — that's the gap this project aims at.

## What works

- **Chess.com fetcher** — rate-limited, resumable, alias-aware (handles Magnus's known smurf accounts). Pulled 805 MB / 124k games / 5.7M decisions across 16 active handles in ~17 minutes.
- **PGN parser + filters** — keeps rated rapid+blitz only, standard-start games only, ≥20 plies. Skip reasons tracked.
- **Board+move encoder** — 18-channel board (piece planes + side-to-move + castling + en passant) + 6-channel move (from-sq, to-sq, promotion flags) = 24 channels × 8 × 8 uint8 per decision.
- **Dataset format** — one parquet per player, zstd-compressed, ~70× smaller on disk than raw tensors. Round-trip verified.
- **Architecture** — ResNet CNN (with `flatten + linear`, NOT global average pool — see LEARNINGS.md for why) → transformer over per-game decisions → masked mean-pool → optional classifier head. ~3.4M params.
- **Training loop** — supports `ce`, `supcon`, or `ce+supcon` loss modes; warmup, gradient clipping, checkpointing.
- **Inference CLI** (`scripts/playlike.py`) — fetches a chess.com user's games, embeds them, cosine-ranks against per-pro centroids.

## What didn't work (yet)

- **Pure supervised contrastive (SupCon)** trained on its own — embeddings collapse to a single point on the unit sphere even with grad clipping, GroupNorm, and variance regularization. Need cross-entropy supervision to break collapse.
- **Global average pool over the 8×8 CNN feature map** — destroys exactly the spatial structure that distinguishes player styles. Switching to flatten+linear was the breakthrough.
- **Several handle typos in `players.json`** — Eric Rosen, Alexandra Botez, Sam Shankland, Anna Rudolf returned no archives because the chess.com handle field in the curated list was wrong. The pipeline correctly logs and skips these.

See [`LEARNINGS.md`](LEARNINGS.md) for the full debugging journey.

## Layout

```
chessengine/
├── src/chessfp/
│   ├── fetch.py       # chess.com Published-Data API client
│   ├── parse.py       # JSON archive → ParsedGame, with filters
│   ├── encode.py      # board / move → uint8 tensor channels
│   ├── dataset.py     # parquet reader
│   ├── model.py       # CNN + transformer + classifier head
│   ├── loss.py        # SupCon, variance-regularization
│   └── train.py       # PK-sampler + training loop + k-shot eval
├── scripts/
│   ├── fetch_games.py # CLI: pull archives
│   ├── build_dataset.py # CLI: parse → parquet
│   ├── train.py       # CLI: training entry point
│   └── playlike.py    # CLI: "who does this user play like?"
├── data/              # gitignored
│   ├── raw/           # chess.com JSON archives
│   └── processed/     # parquet datasets
├── checkpoints/       # gitignored
├── players.json       # curated chess.com handles
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: set the contact email chess.com sees in your User-Agent
export CHESSFP_CONTACT="you@example.com"
```

## Pull data

```bash
# Everyone in players.json since 2021-01:
python scripts/fetch_games.py --since 2021-01

# Or just one player:
python scripts/fetch_games.py --only hikaru_nakamura --since 2024-01
```

The fetcher is rate-limited (1 req/sec by default), resumable (skips already-downloaded months), and respects chess.com's published API guidelines.

## Build the training set

```bash
python scripts/build_dataset.py
# writes data/processed/{player_id}.parquet
```

## Train

```bash
# CE-only, 1500 steps on the full 14-player corpus
python scripts/train.py \
  --steps 1500 --eval-every 250 --log-every 50 \
  --warmup-steps 200 --lr 3e-4 \
  --loss-mode ce \
  --n-players-per-batch 12 --games-per-player 4 \
  --min-games-per-player 50 \
  --out-dir checkpoints/full
```

Use `--loss-mode ce+supcon` to also shape the embedding geometry for cosine similarity (needed for the `playlike` demo to work well).

## Demo: who does this user play like?

```bash
python scripts/playlike.py MagnusCarlsen --since 2025-01 --top-k 10
```

(Quality of the answer depends on how trained the checkpoint is.)

## Data source

[chess.com Published-Data API](https://www.chess.com/news/view/published-data-api) — public, no auth required, free. Be polite: include a contact email in the User-Agent (`CHESSFP_CONTACT` env var) and keep request rate ≲ 1/s.

## Prior art

- McIlroy-Young, Wang, Sen, Kleinberg, Anderson — *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess* (NeurIPS 2021). [paper](https://arxiv.org/abs/2208.01366) · [code](https://github.com/CSSLab/maia-individual)
- [Maia Chess](https://www.maiachess.com/) — same group, personalized human-move prediction.

## License

MIT. See `LICENSE`.
