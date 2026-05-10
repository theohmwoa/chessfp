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

3-player CE smoke test (Hikaru / Magnus / Levy), 200 steps:
- step 25:  acc 0.350 (vs 0.333 random)
- step 100: acc 0.407,  val top1 = **0.374**
- step 200: acc 0.423,  val top1 = **0.453**

Full 14-player CE+SupCon, 600+ steps in:
- step 250: train acc 0.140, val top1 0.102, separation +0.002
- step 500: train acc 0.207, val top1 0.134, separation +0.008
- step 600: train acc 0.212, supcon still flat at 3.82

So: train accuracy is climbing steadily (~3× random), val top-1 is 1.9–2.5× random, embedding separation is small but positive (0.008) and trending up. The model is learning. SupCon hasn't kicked in fully yet — it needs CE to first carve out class-discriminative directions, then it shapes the cosine geometry on top.

Not yet at the McIlroy-Young 98% level — that paper had millions of games per player and a much bigger model. This is a portfolio-scale demonstration.

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
