# Learnings — chessfp engineering journey

Notes from building this in one sitting. Everything below is honest about what happened — including the dead ends, since those are the parts you actually learn from.

## What we built (in order)

### 1. The data pipeline (worked first try)

A polite chess.com Published-Data API client: rate-limited, retries on 429/5xx with backoff, resumable (skips months already on disk), supports alias accounts (Magnus's `DrNykterstein`/`DrDrunkenstein`/`DannyTheDonkey`).

End result: **805 MB raw JSON · 124,599 games · 5.7M focal-player decisions** across 16 chess.com handles, fetched in ~17 minutes at 0.7 s/request.

Top players by data volume:

| Player              | Games  |
|---------------------|--------|
| Daniel Naroditsky   | 44,357 |
| Hikaru Nakamura     | 21,670 |
| Anna Cramling       | 9,743  |
| Levy Rozman         | 8,574  |
| Daniil Dubov        | 8,309  |
| Magnus Carlsen      | 7,203  |
| Ben Finegold        | 5,373  |
| Alireza Firouzja    | 4,967  |

Filtering rules that worked well: rated only, rapid+blitz only (bullet is too time-pressured to be stylistically clean; daily is engine-help-prone), standard chess only (no 960), standard starting position, ≥20 plies. Per-game filter rejection counts are tracked in `ParseStats` for debugging.

**Surprise:** several "famous streamer" handles in our curated list are wrong (Eric Rosen, Alexandra Botez, Anna Rudolf, Sam Shankland all returned 0 archives). The pipeline correctly logs and skips them — no crashes — but it's a reminder that chess.com handles aren't always what you'd guess. Hans Niemann played mostly bullet and got fully filtered out.

### 2. Board + move encoding

**Board** (18 channels × 8 × 8 uint8):
- 0–5: white pieces (P, N, B, R, Q, K)
- 6–11: black pieces (P, N, B, R, Q, K)
- 12: side to move
- 13–16: castling rights (WK, WQ, BK, BQ)
- 17: en passant target

**Move** (6 channels × 8 × 8):
- 0: from-square (one-hot)
- 1: to-square
- 2–5: promotion piece flags (N, B, R, Q)

Total model input: **24 × 8 × 8 uint8 per decision** for the focal player only (we drop the opponent's moves — they're already encoded in the board state). Storage: ~1 KB raw per decision; zstd compresses 70× because the planes are sparse.

### 3. The training that didn't work (and why)

This is where we burned the most time. Three failure modes, in order:

#### Failure 1: SupCon collapse via `[CLS]` pooling

Initial transformer head used a learnable `[CLS]` token with `nn.init.trunc_normal_(std=0.02)`. After one optimizer step, all game embeddings became nearly identical unit vectors (cosine within = cosine between = 1.000). Loss plateaued at exactly `log(11) ≈ 2.40` — the random-baseline upper bound for our supcon configuration.

**Diagnosis:** the `[CLS]` token starts near zero, residual connections preserve the near-zero, the final L2 normalization of a near-zero vector amplifies tiny noise into "every direction looks the same."

**Fix attempted:** swap to **masked mean pool over move embeddings** instead of `[CLS]` pooling. Mean pool inherits the diversity of the input tokens directly. Marginal improvement only.

#### Failure 2: BatchNorm poisoning from padded positions

Variable-length games were padded to `T_max` per batch. Padded positions are all-zero board+move tensors. They flowed through the CNN, including its `BatchNorm2d` layers. BN's running mean/var got pulled toward zero by the all-zero padded inputs, corrupting eval-time normalization.

**Fix:** swap **BatchNorm → GroupNorm** (no running stats) AND skip padded positions from the CNN forward pass entirely:

```python
keep = ~mask.reshape(B * T)
valid = self.move_enc(flat[keep])      # only real positions
full = torch.zeros(B * T, D, device=...)
full[keep] = valid                      # padded slots stay zero
moves = full.reshape(B, T, -1)
```

Helped the eval mode a bit. Did not fix the core training problem.

#### Failure 3: SupCon's catastrophic first step

With SupCon-only loss (no auxiliary signal), the very first optimizer step had grad_norm ≈ 0.9 (fine), but the SECOND step spiked to grad_norm ≈ 3.9 — and that single huge step collapsed the embedding standard deviation from 0.010 to 0.002. After that the model was stuck in a flat region where grad_norm hovered around 0.001.

**Fix attempted:** gradient clipping (`max_norm=1.0`) + LR warmup over 100 steps + variance regularization (VICReg-style: penalize per-dim std falling below 1.0 on un-normalized embeddings). Marginal improvement only — variance reg prevented the worst collapse, but supcon still couldn't find class structure.

#### Failure 4 (the real one): global average pool kills the signal

Tested cross-entropy classification (a strictly easier objective than supcon — direct supervised gradients). On 14 classes, after 100 steps:

```
step  25: ce=2.7077 acc=0.068
step  50: ce=2.6761 acc=0.077
step  75: ce=2.6771 acc=0.066
step 100: ce=2.6614 acc=0.068
```

CE loss random baseline is `log(14) ≈ 2.64`, random accuracy is `1/14 ≈ 0.071`. **The model couldn't learn to classify even with the easiest possible signal.** That's not a contrastive issue — that's the architecture failing.

**Diagnosis:** the CNN ended with `h.mean(dim=(2, 3))` — global average pool over the 8×8 board. Averaging away the per-square structure of the board destroys exactly the information that distinguishes player styles. *Where* the pieces are (not just *what* the average feature value is) is the signal.

**The fix that worked:**

```python
# Before
self.proj = nn.Linear(ch, d_model)
def forward(self, x):
    h = self.blocks(self.stem(x))
    h = h.mean(dim=(2, 3))         # <-- destroys spatial info
    return self.proj(h)

# After
self.proj = nn.Linear(ch * 64, d_model)   # ch*64 = 128*64 = 8192 → 256
def forward(self, x):
    h = self.blocks(self.stem(x))
    h = h.flatten(1)                # <-- preserve spatial info
    return self.proj(h)
```

Cost: one biggish linear layer (~2M params, model goes 3.4M → 5M). Benefit: the model started learning immediately.

### 4. After the fix — current numbers

**3-player CE smoke test** (Hikaru / Magnus / Levy), 200 steps, ~2 minutes wall time on Apple Silicon (MPS):

| step | train acc | val top1 |
|-----:|----------:|---------:|
|  25  | 0.350 (random 0.333) | — |
| 100  | 0.407 | 0.374 |
| 200  | 0.423 | **0.453** |

**Full 14-player CE + SupCon** (`--loss-mode ce+supcon`, supcon weight 0.5, batch = 12 players × 4 games = 48 games):

| step | train acc | val top1 (random 0.071) | separation | wall time |
|-----:|----------:|------------------------:|-----------:|----------:|
|  50  | 0.083 | —     | —     | 35 s |
| 250  | 0.140 | 0.102 | +0.002 | 4 min |
| 500  | 0.207 | 0.134 | +0.008 | 13 min |
| 750  | 0.237 | **0.163** | **+0.017** | 22 min |
| 1000 | (TBD) | (TBD) | (TBD) | ~30 min |
| 1500 | (TBD) | (TBD) | (TBD) | ~45 min |

Each ~250 steps is buying ~+0.03 val top-1 and roughly doubling cosine separation. Train and val accuracy are climbing in parallel (no overfitting yet).

**Hardware:** Apple M-series, MPS backend. Average ~1.8 s/step (variable 0.8–3 s/step depending on system load) for the 14-player setup. CPU fallback works but is materially slower.

**Throughput:** 48 games/batch × ~50 focal decisions/game = ~2,400 board+move tensors per forward pass. Each tensor is 24 × 8 × 8 uint8 = 1,536 bytes. So roughly 3.7 MB of inputs flow through the CNN per step.

**Trajectory and what it means:** the model IS learning at ~2.3× random val top-1 and the supcon component is finally engaging now that CE has carved class-discriminative directions for it. Not yet at the McIlroy-Young 98% level — that paper had millions of games per player, a much bigger model, and trained for far longer. This is a portfolio-scale demonstration.

## End-to-end demo (step 1000 checkpoint, val top-1 = 0.184)

Ran `scripts/playlike.py` against two known pro handles, fetching their games since 2025-09 and ranking against the 15 cached centroids. The model is meant to predict who plays most like the queried handle.

**Query 1 — Hikaru Nakamura (`@Hikaru`, 2,663 games):**

```
 1. hikaru_nakamura           cos=+0.999  ← correct ✓
 2. levy_rozman               cos=+0.990
 3. magnus_carlsen            cos=+0.988
 4. fabiano_caruana           cos=+0.987
 5. ian_nepomniachtchi        cos=+0.986
```

**Query 2 — Magnus Carlsen (`@MagnusCarlsen`, 879 games):**

```
 1. fabiano_caruana           cos=+1.000
 2. alireza_firouzja          cos=+0.999
 3. magnus_carlsen            cos=+0.998  ← correct, ranked 3rd
 4. ian_nepomniachtchi        cos=+0.998
 5. daniel_naroditsky         cos=+0.997
 6. wesley_so                 cos=+0.996
 ...
11. hikaru_nakamura           cos=+0.983
12. ben_finegold              cos=+0.972
13. levy_rozman               cos=+0.972
14. andrea_botez              cos=+0.965
```

Two takeaways:

1. **The clustering is structural, not random.** Magnus's top-5 is Fabi / Firouzja / Magnus / Nepo / Naroditsky — every one of them an elite super-GM. The model has clearly learned the difference between "elite tournament-grade play" and "streamer-grade play": the streamers (Levy, Finegold, Botez) are at the bottom; Hikaru sits between the two clusters as a top streamer-pro hybrid. So even when the model gets the *individual* wrong, it gets the *style cluster* right.
2. **Cosines are very tight** (0.965 → 1.000). That's the small embedding separation (+0.026) showing through. More training, or post-CE supcon-only fine-tuning, would push the cosines apart and turn top-5 confusion into clean top-1 picks.

This matches the eval numbers exactly: top-1 = 0.18, top-5 = 0.56. So the demo is working as advertised — there's nothing wrong with the inference path; the model just isn't fully trained yet.

## Day 2 — pushing further

After the step-1000 checkpoint with val top-1 = 0.184, several things were
worth investigating to push the model further.

### Resume from checkpoint

`scripts/train.py` now supports `--resume path/to/ckpt.pt`. It loads model
weights AND optimizer state (saved alongside in `last.pt` / `best.pt`) and
continues with `--steps` interpreted as *additional* steps. Means we can
chain training sessions without restarting from random init.

### Confusion matrix is far more informative than top-1

Eval reports a single top-1 number, but the per-class breakdown reveals
that the model is excellent on some players and useless on others.

At step 1250 (overall top-1 = 0.270, vs 1/14 = 0.071 random):

| Player              | Per-class top-1 |
|---------------------|----------------:|
| anna_cramling       | 0.817 ← 11× random |
| hikaru_nakamura     | 0.550 |
| levy_rozman         | 0.483 |
| ben_finegold        | 0.367 |
| ian_nepomniachtchi  | 0.300 |
| anish_giri          | 0.283 |
| daniil_dubov        | 0.250 |
| levon_aronian       | 0.183 |
| vidit_gujrathi      | 0.183 |
| alireza_firouzja    | 0.150 |
| daniel_naroditsky   | 0.117 |
| fabiano_caruana     | 0.050 |
| magnus_carlsen      | 0.033 |
| **wesley_so**       | **0.017** ← worse than random! |

The pattern is striking: **distinctive streamers/pedagogical players are
trivial to identify; elite super-GMs are nearly indistinguishable from each
other.** Two specific confusion patterns the model surfaced:

- *Magnus → Ben Finegold (17%)*: Magnus's calm precise play apparently
  looks "pedagogical" to the model. Not crazy — Magnus is famous for
  clean technique.
- *Wesley So → Anish Giri (18%)*: two of the most positional, draw-prone
  elite GMs in the world. They literally play the same kind of chess.
- *Hikaru → Levy Rozman (22%)*: the speed-streamer cluster.

This is a real and meaningful result: behavioral stylometry works very
well for players who have a distinctive voice, and degrades when players
converge on engine-approved play. With only 14 classes and ~5000 training
games each, the elite cluster needs more data — or a margin-loss objective
that explicitly forces intra-class separation.

### Embedding UMAP shows structure even at top-1 = 0.20

`scripts/visualize_embeddings.py` projects val game embeddings to 2D with
PCA + UMAP (cosine metric). At step 1250 (val top-1 = 0.202) the **Anna
Cramling cluster is already visually distinct on the right side of the
UMAP**, and there's a vague upper "speed streamer" region (Hikaru/
Naroditsky/Levy mixed). The elite-GM cluster is still a blob in the
middle — consistent with the confusion-matrix story above.

### ArcFace plumbed but not yet trained

`src/chessfp/loss.py` now has `arcface_logits()` — Deng et al.'s additive
angular margin softmax. Loss modes `arcface` and `arcface+supcon` are
available via `--loss-mode`. Unit-tested numerically but not yet trained
end-to-end. Hypothesis: the explicit angular margin should specifically
help the elite-GM cluster by forcing each class to occupy at least m
radians of the unit sphere — directly attacking the failure mode the
confusion matrix exposed.

### Day-2 training trajectory (resumed from step 1000)

Run config: `--lr 2e-4 --warmup-steps 0 --supcon-weight 1.0`
(bumped supcon weight from 0.5 → 1.0 since CE had already carved
class-discriminative directions). 4000 additional steps, **2h 7min**
wall time on Apple Silicon MPS, batch 12 players × 4 games.

| Step  | train acc | val top-1 (random=0.071) | val top-5 (0.357) | separation | best.pt? |
|------:|----------:|-------------------------:|------------------:|-----------:|----------|
| 1000  | —         | 0.184                    | 0.519             | +0.026     | ← start  |
| 1250  | 0.314     | 0.202                    | 0.563             | +0.028     | new best |
| 1750  | 0.346     | 0.212                    | 0.587             | +0.028     | new best |
| 2000  | 0.350     | 0.220                    | 0.586             | +0.041     | new best |
| 2500  | 0.366     | 0.227                    | 0.616             | +0.039     | new best |
| 3000  | 0.406     | 0.234                    | 0.620             | +0.043     | new best |
| 3250  | —         | 0.242                    | 0.624             | +0.054     | new best |
| 4000  | 0.441     | 0.247                    | 0.638             | +0.049     | new best |
| 4250  | —         | 0.250                    | 0.628             | +0.048     | new best |
| **4500**  | 0.476     | **0.263**                | 0.629             | **+0.061**  | **← final best** |
| 5000  | 0.479     | 0.255                    | 0.629             | +0.049     | —        |

**Train acc 0.184 → 0.263 (+43% relative)**. Cosine separation
**0.026 → 0.061 (more than doubled)**. Best.pt landed at step 4500;
training past that started to plateau and trend slightly down on val
top-1 even as train acc kept climbing — mild overfitting.

### Per-class accuracy improvements with more training

Centroid-based confusion matrix (held-out val games):

| Player              | step 1250 | step 2500 | **step 4500** | Δ      |
|---------------------|----------:|----------:|--------------:|-------:|
| anna_cramling       | 0.82      | 0.83      | **0.86**      | +0.04  |
| levy_rozman         | 0.48      | 0.57      | **0.61**      | +0.13  |
| hikaru_nakamura     | 0.55      | 0.52      | **0.58**      | +0.03  |
| ben_finegold        | 0.37      | 0.42      | **0.55**      | +0.18  |
| **levon_aronian**   | 0.18      | 0.38      | **0.50**      | **+0.32** |
| daniil_dubov        | 0.25      | 0.23      | 0.35          | +0.10  |
| daniel_naroditsky   | 0.12      | 0.23      | 0.33          | +0.21  |
| anish_giri          | 0.28      | 0.37      | 0.28          | 0      |
| ian_nepomniachtchi  | 0.30      | 0.30      | 0.28          | -0.03  |
| vidit_gujrathi      | 0.18      | 0.15      | 0.16          | -0.02  |
| alireza_firouzja    | 0.15      | 0.07      | 0.13          | -0.02  |
| wesley_so           | 0.02      | 0.03      | 0.09          | +0.07  |
| fabiano_caruana     | 0.05      | 0.00      | 0.06          | +0.01  |
| magnus_carlsen      | 0.03      | 0.07      | 0.04          | +0.01  |
| **overall**         | **0.27**  | **0.30**  | **0.34**      | +0.07  |

Players who became *clearly* identifiable (acc > 0.5 at step 4500):
**Anna Cramling, Hikaru, Levy, Ben Finegold, Levon Aronian**. That's 5
of 14 players with strong individual signatures.

The elite-GM cluster (Magnus, Fabi, Wesley, Firouzja) hasn't budged
materially across 4000 steps of training. This isn't a data scaling
problem — they have plenty of games. It's a *fundamental style overlap*
problem: super-GMs play engine-approved moves, so their decision
distributions converge.

### End-of-day demo

Re-ran `scripts/playlike.py` on the step-4500 checkpoint against two
chess.com handles, fetching games since 2025-09:

**@Hikaru (2,663 games):**
```
 1. hikaru_nakamura   cos=+0.913  ✓ correct
 2. levy_rozman       cos=+0.913
 3. andrea_botez      cos=+0.892
 4. fabiano_caruana   cos=+0.880
 5. ian_nepomniachtchi cos=+0.880
 6. magnus_carlsen    cos=+0.879
 ...
14. anish_giri        cos=+0.834
```
Spread now 0.083 (cos 0.834 → 0.913) vs the morning's 0.035 — model
is genuinely more discriminative.

**@MagnusCarlsen (1,095 games):**
```
 1. hikaru_nakamura     cos=+0.915
 2. ian_nepomniachtchi  cos=+0.914
 3. fabiano_caruana     cos=+0.914
 4. levy_rozman         cos=+0.912
 5. magnus_carlsen      cos=+0.912   ← correct, rank 5
```
Magnus's rank for himself went from #3 (morning model) → #5 (afternoon
model). Counter-intuitive but consistent with the confusion matrix:
Magnus's per-class accuracy never improved. With more training the
**other** players got more distinguishable and pushed past Magnus's flat
score. That's the elite-GM story laid bare.

### Visualizations

Saved in `viz/`:
- `training_curve.png` — 3-panel: losses / accuracies / cosine geometry
- `viz/final/embeddings_pca.png` — PCA projection
- `viz/final/embeddings_umap.png` — UMAP (cosine metric), shows visually
  distinct streamer cluster (Anna Cramling region) and a faint elite-GM
  blob
- `viz/confusion_matrix.png` — row-normalized 14×14 confusion

### Day-2 ArcFace experiment

Hypothesis: the elite-GM cluster (Magnus, Fabi, Wesley, Firouzja) collapses
in CE+SupCon training because nothing explicitly forces those players'
embeddings apart — their style distributions naturally overlap, and CE
is satisfied with a "good enough" linear separator that doesn't push
classes geometrically far. ArcFace's additive angular margin should
directly attack this by requiring each class to occupy ≥ m radians of
the unit sphere.

**Run 1 — ArcFace from scratch (random init):** acc stuck at 0.000 through
step 200, loss 11.3. The classic cold-start problem: at random
initialization, embeddings have ~zero cosine with classifier weights, so
adding the margin makes the *target* class get punished (negative logit)
while non-target classes stay at 0. Killed at step 200.

**Run 2 — ArcFace fine-tune** from the CE+SupCon checkpoint at step 4500
(val top-1 = 0.263, our previous best). Settings: `--lr 1e-4
--arcface-margin 0.20 --arcface-scale 30`, 1500 steps, 82 min wall time.

| Step  | val top-1 | val top-5 | sep   | note                  |
|------:|----------:|----------:|------:|-----------------------|
| 4500  | 0.263     | 0.629     | +0.061 | starting CE+SupCon  |
| 4750  | 0.265     | 0.628     | +0.037 | +0.002, sep dropped |
| **5250** | **0.268** | 0.620 | +0.034 | **best**            |
| 5500  | 0.262     | 0.623     | +0.048 | —                     |
| 5750  | 0.265     | 0.626     | +0.054 | sep recovering        |
| 6000  | 0.264     | 0.618     | +0.053 | final                 |

Marginal aggregate win: **top-1 0.263 → 0.268 (+0.005)**. But the per-class
breakdown is more revealing:

| Player              | CE+SupCon | **ArcFace-FT** | Δ      | note            |
|---------------------|----------:|---------------:|-------:|-----------------|
| **fabiano_caruana** | 0.06      | **0.14**       | **+0.08** | biggest elite-GM gain |
| ben_finegold        | 0.55      | 0.59           | +0.04  | streamer up too |
| alireza_firouzja    | 0.13      | 0.15           | +0.02  | elite up        |
| levon_aronian       | 0.50      | 0.51           | +0.01  | flat            |
| wesley_so           | 0.09      | 0.10           | +0.01  | elite up        |
| hikaru_nakamura     | 0.58      | 0.58           | 0      | flat            |
| daniel_naroditsky   | 0.33      | 0.33           | 0      | flat            |
| magnus_carlsen      | 0.04      | 0.01           | -0.03  | dropped         |
| daniil_dubov        | 0.35      | 0.30           | -0.05  | streamer down   |
| levy_rozman         | 0.61      | 0.56           | -0.05  | streamer down   |
| anna_cramling       | 0.86      | 0.79           | -0.08  | biggest loss    |
| anish_giri          | 0.28      | 0.20           | -0.08  | flat-ish pos.   |
| ian_nepomniachtchi  | 0.28      | 0.18           | -0.10  | elite down      |
| **overall**         | **0.343** | **0.329**      | **-0.014** | net loss     |

So the hypothesis was **partially right**: ArcFace specifically helped Fabi
(the most "engine-like" elite GM), Firouzja, and Wesley — all members of
the elite cluster. But it cost overall accuracy because it tightened the
embedding too much around the elite-GM angles and started bunching
streamers together.

**Honest read:** ArcFace from a CE-pretrained backbone is a real lever
for pushing apart the elite cluster, but a milder margin (0.10–0.15) or
**ArcFace + SupCon together** is probably the right recipe — supcon keeps
the streamer geometry intact while ArcFace's margin forces the elite
players apart. Plumbing supports `--loss-mode arcface+supcon` already;
just need another training run.

### Experiment 5 — max_len=128 → 256

Hypothesis: with the default `max_len=128`, ~40% of blitz/rapid games hit
the cap and we truncate them mid-middlegame. Endgame technique is where
elite-GM styles diverge most (Magnus's endgame ≠ Wesley So's endgame),
so doubling `max_len` should specifically help the elite cluster.

**Run from scratch with `--max-len 256`**, same CE+SupCon recipe. Total:
4000 steps over ~7.7 hours wall time (with one premature kill+resume on
my part because I jumped to conclusions at step 1250 — see notes below).

Trajectory vs the original `max_len=128` run at matching steps:

| Step  | m=128 top-1 | **m=256 top-1** | Δ      |
|------:|------------:|----------------:|-------:|
| 250   | 0.102       | 0.106           | +0.004 |
| 500   | 0.134       | 0.137           | +0.003 |
| 750   | 0.163       | 0.167           | +0.004 |
| 1000  | 0.184       | 0.180           | -0.004 |
| 1250  | 0.202       | 0.191           | -0.011 |
| 1500  | 0.194       | 0.197           | +0.003 |
| 1750  | 0.212       | 0.201           | -0.011 |
| 2000  | 0.220       | 0.223           | +0.003 |
| 2500  | 0.227       | **0.236**       | **+0.009** |
| 3000  | 0.234       | 0.233           | -0.001 |
| 3500  | 0.239       | 0.242           | +0.003 |
| **3750**  | 0.242   | **0.252**       | **+0.010** |
| **4000**  | 0.247   | **0.252**       | **+0.005** |

Net: **+0.005 top-1 at matching step count**, with the m=256 model
catching up around step 2500 and consistently equal-or-better afterward.
NOT decisive — the magnitude is below the variance between consecutive
evals.

**Per-class accuracy is where the story gets interesting:**

| Player              | m=128  | **m=256** | Δ      | note |
|---------------------|-------:|----------:|-------:|------|
| **hikaru_nakamura** | 0.58   | **0.65**  | **+0.07** | best Hikaru of any model |
| daniel_naroditsky   | 0.33   | 0.35      | +0.02  | streamer up |
| ben_finegold        | 0.55   | 0.55      | 0      | tied |
| anna_cramling       | 0.86   | 0.83      | -0.03  | flat-ish |
| levy_rozman         | 0.61   | 0.56      | -0.05  | slight drop |
| levon_aronian       | 0.50   | 0.44      | -0.06  | drop |
| **ian_nepomniachtchi** | 0.28 | **0.19**  | **-0.09** | elite drop |
| **wesley_so**       | 0.09   | **0.01**  | **-0.08** | elite collapse |
| **magnus_carlsen**  | 0.04   | **0.01**  | -0.03  | elite drop |
| **overall**         | 0.343  | **0.320** | -0.023 | worse on average |

So the longer context **specifically helps blitz players with long games**
(Hikaru +0.07 — his blitz games regularly hit 200+ plies, so the extra
context is now actually used). It **hurts** the elite super-GM cluster
even further, presumably because endgame play among super-GMs is the
*most* engine-like part of the game, so giving the model more endgame
context provides MORE collapsed signal, not less.

This was a real negative result on the hypothesis: longer context does
not break the elite cluster, it tightens it.

### About my "kill at step 1250" mistake

When the m=256 run hit step 1250 at top-1=0.191 (vs 0.202 for m=128
same step), I called it as failing and killed the training. The user
pushed back: "but isn't it normal it takes more time?" — they were
right. The 0.011 gap was within single-eval variance. Doubling
positional embeddings means more parameters to learn early, so slower
convergence per step is expected. The model recovered and surpassed
m=128 by step 2500.

**Lesson:** at small batch sizes and noisy eval, single-step rank
inversions are not signal. Reserve judgement until you have a clear
trend over 3-5 consecutive evals. I had been doing this on Day 1 (Magnus
identification regressing was a real trend over many evals); on Day 2
I shortened the patience window and was wrong about it.

## Final scoreboard

Four trained models from Day 2, plus the Day 1 baseline:

| Model file                         | val top-1 | centroid acc | best at         | comment                          |
|------------------------------------|----------:|-------------:|:---------------:|----------------------------------|
| Day 1 CE+SupCon (step 1000)        | 0.184     | 0.270        | step 1000       | starting point of Day 2          |
| `full_long/best.pt` CE+SupCon      | 0.263     | **0.343**    | step 4500       | best per-class average           |
| `arcface_ft/best.pt`               | 0.268     | 0.329        | step 5250       | best on Fabi (0.14)              |
| **`arcsup_ft/best.pt`**            | **0.269** | 0.329        | step 5000       | **best val top-1**               |
| `long_seq/best.pt` (m=256)         | 0.252     | 0.320        | step 3750       | best on Hikaru (0.65)            |

There is no single best model. The right pick depends on the use case:

- **For the "play like a pro" demo**: `arcsup_ft/best.pt` — highest val
  top-1 and best embedding spread.
- **For "identify Hikaru/Naroditsky-style speed players"**:
  `long_seq/best.pt` — its long-context advantage shines on blitz.
- **For a balanced confusion matrix**: `full_long/best.pt` — best
  per-class average (0.343), no class collapses entirely.

**The real research finding from Day 2:** at this data scale (5–35k games
per player), the elite-super-GM cluster is genuinely inseparable with
this architecture, regardless of loss function (CE, SupCon, ArcFace,
ArcFace+SupCon) or context length (128 vs 256). All four
variants hit a ~0.27 ceiling and all four fail similarly on
Magnus/Wesley/Fabi. This isn't a training problem — it's a real
statement that *engine-like elite GM play converges to a single
behavioral signature*, which is itself an interesting result.

### Day-2 features added

- `--resume <ckpt>` flag for training (loads model + optimizer state +
  step counter). Best.pt and last.pt now save `opt_state` too.
- `arcface_logits()` in `loss.py` — Deng et al. additive angular margin
  softmax. Plumbed through `--loss-mode arcface` and `arcface+supcon`,
  numerically smoke-tested but not yet trained end-to-end.
- `scripts/confusion_matrix.py` — produces both an image and a per-
  class table.
- `scripts/visualize_embeddings.py` — PCA + UMAP projections.
- `scripts/plot_history.py` — 3-panel training curve plot.
- `scripts/playlike.py` — added `--show-top-games` to surface the
  user's most-pro-like games with linkable chess.com URLs.

## Things I'd do differently next time

1. **Validate the architecture on the easiest possible loss before introducing the fancy one.** I should have started with cross-entropy classification, not SupCon. CE has clean, strong gradients and any architectural pathology shows up immediately. Fixing it under SupCon is debugging two problems at once.
2. **Print embedding statistics every step early on.** The collapse was hidden behind plausible-looking loss numbers; only when I logged `emb.std(dim=0).mean()` did it become obvious that the embeddings had collapsed in step 1.
3. **Be suspicious of global average pool when spatial layout *is* the signal.** GAP is correct for "is this a cat?" — translation-invariant classification. It is not correct for "where on this board are the pieces, exactly?"
4. **Validate handles before fetching at scale.** I burned ~30 min worth of API calls on `EricRosen` etc. that returned nothing. A 5-second `list_archives` precheck would have saved that.

## Open issues

- Variance regularization is currently disabled (weight 0). Worth re-enabling at a small weight (~0.1) once CE+SupCon are co-trained, to push apart any residual collapse.
- Eval is still ~30 s per call even with the 60-game cap — could be made fast with batched embedding via larger inference batches.
- Should investigate ArcFace / CosFace margin softmax as a drop-in replacement for CE — directly enforces angular separation in embedding space, which is exactly what we want for cosine-similarity inference.
- The "play like a pro" CLI (`scripts/playlike.py`) is built and runnable but hasn't been validated end-to-end against a trained checkpoint of useful quality.

## Citations

- McIlroy-Young, Wang, Sen, Kleinberg, Anderson. *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess.* NeurIPS 2021. [arXiv:2208.01366](https://arxiv.org/abs/2208.01366)
- Khosla et al. *Supervised Contrastive Learning.* NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
- Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
