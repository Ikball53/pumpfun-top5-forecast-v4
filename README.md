# Pumpfun Top-5 Forecast v4 — eightbal

Forecasts the 5 largest pump.fun coins by market cap at each answer read,
for competition `pumpfun-top5-forecast-01` ($200 escrow).

## What changed from v3

v3 scored **-36.3** on the dev (order -19.46, size -16.86) against the
leader's -31.7 (order -18.18, size -13.56). The remaining **size gap
(+3.30 nats)** is still larger than the **order gap (+1.28)**, so the
market-cap prediction is still the weakest part. v4 targets it:

* **Sizes are rebuilt as tight, directly calibrated per-read mixtures.**
  v3's size distribution came from the Monte Carlo histogram of winning
  incumbents' caps plus one pooled newborn normal. That histogram is far
  too wide (sd 0.56–0.69 dex per place, top bin only 0.04–0.09) because
  each coin's cap sd in the MC is ~0.60 dex at one hour. But the actual
  top-5 caps at a given horizon are tightly clustered (sd 0.03–0.13 dex
  across (cut, read) pairs): the top of the market is a crowd,
  persisting coins grow smoothly, and a crash just hands the place to
  the next coin at nearly the same cap. v4 builds each place's size
  distribution per read as a moment-matched normal mixture of two
  tight components — an incumbent centre (the k-th largest of the
  cut's top-20 caps, grown toward a ceiling set by the measured
  top-cap growth rate) and a newborn centre (this place's newborn
  mean, grown from its reference horizon) — mixed by the per-read
  unnamed target, averaged over the reads, and blended with a small
  broad prior (8%, sd 0.45) for robustness. The result is sd
  0.27–0.38 dex per place with a top bin of 0.05–0.08.

* **Newborn caps are per-place, and detrended by horizon.** Newborn
  winners' caps differ sharply by place (pooled means 3.45 / 3.20 /
  3.18 / 3.09 / 2.92, and they grow ~+0.07 dex per 10 min). A single
  pooled normal mis-centres every place. v4 measures each place's
  newborn mean and sd (recency-weighted, shrunk toward the pooled
  value). The sd is detrended by horizon — a least-squares line of
  cap vs horizon is fit per place and the sd is taken of the
  residuals — because the raw pooled sd is inflated by the growth
  across horizons (place 5's raw sd 0.38 is mostly the 2.63 → 3.11
  rise, not within-horizon spread).

* **The unnamed-mass target is raised to match the measured newborn
  share.** v3 capped it at 2× the per-step turnover (0.40) and U_MAX
  0.50, giving u_final ~0.37, but the history's (cut, read) pairs
  show the newborn share of top-5 places is ~0.50 at 10 min and
  0.67–0.80 at 20–40 min, and the practice's own last read had 3 of
  5 places won by post-cut coins (0.60). v4 saturates at 3× the
  turnover and U_MAX 0.65, giving u_final ~0.51, and uses the
  per-read target in the size mixture, so later reads — where
  newborns dominate — carry more newborn weight.

* **The order names more coins per place** (top 40 candidates by
  momentum, not only the ~14 that ever win a Monte Carlo scenario).
  Naming more coins shrinks the pool the unnamed mass is shared
  over, so a newborn that takes a place scores the unnamed share
  over fewer coins, and the extra coins carry real (if small)
  probability from the smoothing (raised to 0.05).

Everything else is v3's: runtime calibration of the newborn rate,
turnover, incumbent drift/volatility, and the Monte Carlo that sets
the order (probabilities). Only the size construction, the unnamed
target, and the named set change. The top-cap growth rate used for
the size centres is measured from the **last two** transitions only
(the late regime): the early marks were still filling in from
~2.0 dex (shifts of +0.5 to +0.9 per 10 min), which would inflate
the growth rate and push every centre far too high.

## Honest caveat

The local replica (kit-history splits) is **unreliable** as a
predictor of the dev score — it said v3 would be worse than v2 on
every split, yet v3 scored 3.4 nats better on the real dev (-36.3
vs -39.7). None of v4's numbers are tuned to the replica; every
change is a principled fix to a measured calibration error
(too-wide sizes, pooled newborn caps, too-low unnamed mass). The
dev score of v4 cannot be predicted locally. On the replica's three
short-horizon reads v4 scores slightly worse than v3 on sizes
(-13.21 vs -12.97), but those reads are 10–20 min out, where the
growth and newborn dominance that v4 targets are smallest; the real
dev reads are 10–70 min out. If v4 scores worse on the dev, the
entry can be reverted to v3 (the best so far, -36.3).

## How it works

1. Reads `/input/history.ndjson` (marks) and `/input/task.json`.
2. Calibrates at runtime from the history: newborn-winner caps
   (pooled and per-place, recency-weighted, horizon-detrended),
   strong-newborn rate, top-5 turnover, incumbent drift/volatility
   by venue, top-cap growth rate (last two transitions), growth
   ceiling (set so the saturating growth's initial slope equals the
   measured growth rate).
3. Runs an 8,000-scenario Monte Carlo over the answer reads for the
   order (probabilities), with per-coin hazard/drift/vol from the
   calibration, and rescales the named probabilities so the unnamed
   mass equals the calibrated target.
4. Builds sizes per place as the per-read moment-matched normal
   mixture described above, averaged over the reads, plus the prior.
5. Writes `/output/forecast.json` (probabilities, top5, sizes).

Deterministic: fixed seed 20261112, sorted iteration, no wall
clock, no network. Two runs on the same input are byte-identical.

## License

MIT. See LICENSE.
