# pumpfun-top5-forecast-01 entry v4: forecast the 5 largest pump.fun coins
# by market cap at each answer read.
#
# Why v4 is different from v3:
#   * v3 scored -36.3 on the dev (order -19.46, size -16.86) against the
#     leader's -31.7 (order -18.18, size -13.56).  The remaining SIZE gap
#     (+3.30 nats) is still larger than the ORDER gap (+1.28), so the cap
#     prediction is still the weakest part.  Diagnosing v3's size output
#     showed its size distribution is far too WIDE: sd 0.56-0.69 dex per
#     place, with the top bin holding only 0.04-0.09.  The width comes
#     mostly from the Monte Carlo histogram, whose per-coin cap sd is
#     ~0.60 dex at one hour (the pool volatility 0.34 dex per 10 min,
#     measured over ALL pool coins including ones that crash or jump,
#     scaled to an hour).  But the actual top-5 caps at a given horizon
#     are TIGHTLY clustered (sd 0.03-0.13 dex across (cut, read) pairs):
#     the top of the market is a crowd, persisting pool coins grow
#     smoothly (~+0.10 dex per 10 min, sd ~0.05), and when one coin
#     crashes another takes its place at nearly the same cap.  So v4
#     replaces the MC histogram for SIZES with a tight, directly
#     calibrated per-read mixture of two normals (incumbent centre +
#     newborn centre), moment-matched and averaged over the reads.
#   * The newborn component is now PER-PLACE.  Newborn winners' caps
#     differ sharply by place (pooled means 3.45 / 3.20 / 3.18 / 3.09 /
#     2.92, sds 0.12 / 0.13 / 0.27 / 0.30 / 0.38) and grow with horizon
#     (~+0.07 dex per 10 min), converging toward the incumbent level.
#     v3 used one pooled newborn normal for every place; v4 measures
#     each place's newborn mean and sd (recency-weighted) and grows it
#     with horizon, so each place's newborn component is centred and
#     scaled correctly.
#   * The unnamed-mass target is raised to match the measured newborn
#     share.  v3 capped it at 2x the per-step turnover (0.40) and U_MAX
#     0.50, giving u_final ~0.37, but the history's (cut, read) pairs
#     show the newborn share of top-5 places is ~0.50 at 10 min and
#     0.67-0.80 at 20-40 min, and the practice's own last read had 3 of
#     5 places won by post-cut coins (0.60).  v4 saturates at 3x the
#     turnover and U_MAX 0.65, giving u_final ~0.51, and uses the
#     per-read target (not just the mean) in the size mixture, so later
#     reads -- where newborns dominate -- carry more newborn weight.
#   * The ORDER names more coins per place (top 40 candidates, not
#     only the ~14 that ever win a Monte Carlo scenario).  Naming more
#     coins shrinks the pool the unnamed mass is shared over, so the
#     unnamed share per newborn is larger on the places newborns win,
#     and the smoothing is raised to 0.05 so the extra coins carry real
#     (if small) probability.
#
# Everything else is v3's: runtime calibration of the newborn rate,
# turnover, incumbent drift/volatility, and the Monte Carlo that sets
# the order (probabilities).  Only the size construction, the unnamed
# target, and the named set change.
#
# Honest caveat: the local replica (kit-history splits) is UNRELIABLE
# as a predictor of the dev score -- it said v3 would be worse than v2
# on every split, yet v3 scored 3.4 nats better on the real dev.  None
# of v4's numbers are tuned to the replica; every change is a
# principled fix to a measured calibration error (too-wide sizes,
# pooled newborn caps, too-low unnamed mass).  The dev score of v4
# cannot be predicted locally.
#
# Determinism: one fixed seed (20261112), candidates iterated in sorted
# mint order, no set iteration, no wall clock, no network, no randomness
# outside the seeded generator.  Two runs on the same history write
# byte-identical output.

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

SEED = 20261112
SCENARIOS = 8000
SMOOTH_PLACE = 0.05       # Laplace share spread over the named coins
MAX_CANDIDATES = 60
MAX_ROWS_PER_PLACE = 100
NAMED_EXTRA = 40          # candidates named per place beyond the MC winners
DEAD_CAP = 1.45           # log10 SOL cap of a dead curve coin
DEAD_SD = 0.08
SIGMA0 = 0.05             # base sd (dex) at the cut
SIGMA_CAP = 0.60          # cap on total sd (dex) at any horizon

# Priors (used only when the history is too thin to measure a quantity;
# every value is shrunk toward these by the measured sample size).
PRIOR_NB_MU = 3.25
PRIOR_NB_SD = 0.30
PRIOR_NB_RATE = 1.6       # strong newborns (>3.0 dex) per 10 min
PRIOR_TURNOVER = 0.20     # fraction of top-5 places changing per 10 min
PRIOR_VOL_POOL = 0.34     # pool sd (dex) per 10 min
PRIOR_VOL_CURVE = 0.094   # curve sd (dex) per 10 min
PRIOR_GROWTH = 0.08       # top-cap growth per 10 min (dex), late regime
SIZE_PRIOR_WEIGHT = 0.08  # weight of the broad normal mixed into sizes
SIZE_PRIOR_SD = 0.45      # sd (dex) of the broad normal mixed into sizes
U_MAX = 0.65              # cap on the unnamed-mass target
SIZE_SD0 = 0.10           # base sd (dex) of a place-cap at the cut
SIZE_SD_H = 0.05          # growth of place-cap sd per hour (dex)
SIZE_SD_CAP = 0.30        # cap on place-cap sd (dex)
GROWTH_TAU_H = 1.0        # saturating-growth time constant (hours)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def floor6(x):
    """Round down to 6 decimals (never -0), like the house starter."""
    r = math.floor(x * 1_000_000) / 1_000_000
    return 0.0 if r == 0 else r


def normal_cdf(x, mu, sd):
    if sd <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def bin_of(value, edges):
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if edges[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _shrink(measured, prior, n, strength=5.0):
    """Shrink a measured value toward a prior by sample size."""
    if n <= 0:
        return prior
    return (n * measured + strength * prior) / (n + strength)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")
    n_bins = len(edges) + 1

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")

    mark_times = [parse_time(m["at"]) for m in marks]
    cut_time = mark_times[-1]
    read_times = [parse_time(r) for r in task.get("reads", []) or []]
    if not read_times:
        read_times = [datetime.fromtimestamp(cut_time.timestamp() + 3600.0 * h, tz=timezone.utc)
                      for h in range(1, 8)]
    horizons = [(rt - cut_time).total_seconds() / 3600.0 for rt in read_times]
    horizons = [h if h > 0 else 0.0 for h in horizons]

    # ---- per-coin series from the marks -----------------------------------
    series = {}
    for mi, m in enumerate(marks):
        t_h = (mark_times[mi] - mark_times[0]).total_seconds() / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            cap_lamports = int(row["market_cap_lamports"])
            if cap_lamports <= 0:
                continue
            x = math.log10(cap_lamports) - 9.0  # log10 of the cap in SOL
            e = series.get(mint)
            if e is None:
                e = {"t": [], "x": [], "venue": row.get("venue", "curve"),
                     "complete": bool(row.get("complete", False)),
                     "created": row.get("created_at")}
                series[mint] = e
            e["t"].append(t_h)
            e["x"].append(x)
            e["venue"] = row.get("venue", e["venue"])
            e["complete"] = bool(row.get("complete", e["complete"]))

    last_rows = marks[-1]["rows"]
    last_by_mint = {}
    for row in last_rows:
        cap_lamports = int(row["market_cap_lamports"])
        if cap_lamports > 0:
            last_by_mint[row["mint"]] = math.log10(cap_lamports) - 9.0

    def top_of(m, n=5):
        return sorted(m["rows"], key=lambda r: -int(r["market_cap_lamports"]))[:n]

    # ---- RUNTIME CALIBRATION from the input history ------------------------
    # (a) newborn-winner caps: coins that won a top-5 place within 30 min
    # of being born.  Measured across all (cut, read) pairs in the history.
    # The caps rise over the history (the market is growing), so a
    # pooled mean is dragged down by the early, low-cap marks.  Weight
    # each (cut, read) pair by the recency of its cut: the late regime
    # (whose newborn winners sit at 3.31-3.56 dex) is the relevant one
    # for a forecast from the last mark.  The spread is pooled (it is
    # stable) with a floor so a tiny late sample cannot make it collapse.
    nb_caps = []  # (cap, weight)
    nb_by_place = {k: [] for k in range(places)}  # place (0-based) -> [(cap, weight, horizon_h)]
    nb_h_ref = []  # horizons (hours) of the newborn samples, for the ref horizon
    for i in range(len(marks)):
        hist_mints = set()
        for m in marks[:i + 1]:
            hist_mints |= {r["mint"] for r in m["rows"]}
        w = (i + 1.0) / float(len(marks))
        for j in range(i + 1, len(marks)):
            if (j - i) * 10 > 30:
                continue
            for k, r in enumerate(top_of(marks[j])):
                if r["mint"] not in hist_mints:
                    cap_lamports = int(r["market_cap_lamports"])
                    if cap_lamports > 0:
                        x = math.log10(cap_lamports) - 9.0
                        nb_caps.append((x, w))
                        nb_by_place[k].append((x, w, (j - i) * 10.0 / 60.0))
                        nb_h_ref.append((j - i) * 10.0 / 60.0)
    if nb_caps:
        wsum = sum(w for _, w in nb_caps)
        nb_mu_raw = sum(x * w for x, w in nb_caps) / wsum
        mu_pool = _mean([x for x, _ in nb_caps])
        nb_sd_raw = _sd([x for x, _ in nb_caps])
    else:
        nb_mu_raw = PRIOR_NB_MU
        mu_pool = PRIOR_NB_MU
        nb_sd_raw = PRIOR_NB_SD
    # shrink toward the prior; the unweighted pooled mean anchors the
    # shrinkage so a small late sample cannot swing the level far.
    n_nb = len(nb_caps)
    nb_mu = _shrink(nb_mu_raw, PRIOR_NB_MU, n_nb, strength=8.0)
    if n_nb >= 8:
        nb_mu = 0.6 * nb_mu + 0.4 * _shrink(mu_pool, PRIOR_NB_MU, n_nb, strength=8.0)
    nb_sd = _shrink(nb_sd_raw, PRIOR_NB_SD, n_nb, strength=8.0)
    nb_sd = max(0.20, min(0.60, nb_sd))
    nb_mu = max(3.05, min(3.60, nb_mu))
    # (a2) PER-PLACE newborn-winner caps.  Newborn winners' caps differ
    # sharply by place (place 1 newborns are the biggest, place 5 the
    # smallest), and a single pooled normal mis-centres every place.
    # Each place's mean/sd is recency-weighted like the pooled one and
    # shrunk toward the pooled (nb_mu, nb_sd) so a thin sample falls
    # back to the pooled value instead of a noisy per-place one.
    # The sd is DETRENDED by horizon: newborn caps grow with horizon
    # (~+0.07 dex per 10 min), so the raw pooled sd is inflated by the
    # growth across horizons (place 5's raw sd 0.38 is mostly the
    # 2.63 -> 3.11 rise, not within-horizon spread).  A least-squares
    # line of cap vs horizon is fit per place and the sd is taken of the
    # residuals, which is the true within-horizon spread.
    nb_h_ref_mean = _mean(nb_h_ref) if nb_h_ref else 0.42
    nb_mu_k = []
    nb_sd_k = []
    for k in range(places):
        xs = nb_by_place[k]
        if xs:
            wsum_k = sum(w for _, w, _ in xs)
            mu_k = sum(x * w for x, w, _ in xs) / wsum_k
            # detrended sd: residuals from a cap-vs-horizon line
            if len(xs) >= 3:
                mh = sum(h for _, _, h in xs) / len(xs)
                mx = sum(x for x, _, _ in xs) / len(xs)
                num = sum((h - mh) * (x - mx) for x, _, h in xs)
                den = sum((h - mh) ** 2 for _, _, h in xs)
                slope = num / den if den > 0 else 0.0
                slope = max(0.0, min(1.5, slope))
                inter = mx - slope * mh
                resid = [x - (inter + slope * h) for x, _, h in xs]
                sd_k = _sd(resid)
            else:
                sd_k = _sd([x for x, _, _ in xs])
        else:
            mu_k = nb_mu
            sd_k = nb_sd
        mu_k = _shrink(mu_k, nb_mu, len(xs), strength=4.0)
        sd_k = _shrink(sd_k, nb_sd, len(xs), strength=4.0)
        nb_mu_k.append(max(2.60, min(3.70, mu_k)))
        nb_sd_k.append(max(0.12, min(0.35, sd_k)))

    # (b) strong-newborn rate: DISTINCT coins reaching cap > 3.0 within
    # 30 min of birth, per 10-min interval, over the whole history.  A
    # coin can appear in several marks, so it is counted once (at the
    # first mark where it is both young and strong).
    strong_mints = set()
    for mi, m in enumerate(marks):
        at = mark_times[mi]
        for row in m["rows"]:
            created = row.get("created_at")
            if not created:
                continue
            age_min = (at - parse_time(created)).total_seconds() / 60.0
            if 0.0 <= age_min <= 30.0:
                cap_lamports = int(row["market_cap_lamports"])
                if cap_lamports > 0 and math.log10(cap_lamports) - 9.0 > 3.0:
                    strong_mints.add(row["mint"])
    span_min = max(10.0, (mark_times[-1] - mark_times[0]).total_seconds() / 60.0)
    n_intervals = max(1.0, span_min / 10.0)
    nb_rate = _shrink(len(strong_mints) / n_intervals, PRIOR_NB_RATE, n_intervals, strength=2.0)
    nb_rate = max(0.5, min(4.0, nb_rate))

    # (c) top-5 turnover per 10 min, from the last two transitions (the
    # recent regime is the relevant one; early marks were far more volatile).
    turn_rates = []
    for i in range(max(0, len(marks) - 3), len(marks) - 1):
        prev_top = {r["mint"] for r in top_of(marks[i])}
        cur_top = [r["mint"] for r in top_of(marks[i + 1])]
        changed = sum(1 for mint in cur_top if mint not in prev_top)
        turn_rates.append(changed / 5.0)
    turnover = _shrink(_mean(turn_rates), PRIOR_TURNOVER, len(turn_rates), strength=2.0)
    turnover = max(0.05, min(0.60, turnover))

    # (d) incumbent drift and volatility by venue (10-min log-cap changes).
    vol = {}
    drift = {}
    for venue in ("pool", "curve", "migrating"):
        ds = []
        for i in range(len(marks) - 1):
            a = {r["mint"]: math.log10(int(r["market_cap_lamports"])) - 9.0
                 for r in marks[i]["rows"] if int(r["market_cap_lamports"]) > 0}
            for row in marks[i + 1]["rows"]:
                if row.get("venue") != venue:
                    continue
                mint = row["mint"]
                cap_lamports = int(row["market_cap_lamports"])
                if mint in a and cap_lamports > 0:
                    ds.append(math.log10(cap_lamports) - 9.0 - a[mint])
        prior_vol = PRIOR_VOL_POOL if venue == "pool" else PRIOR_VOL_CURVE
        prior_drift = PRIOR_GROWTH if venue == "pool" else 0.0
        vol[venue] = max(0.05, min(0.60, _shrink(_sd(ds), prior_vol, len(ds), strength=3.0)))
        drift[venue] = max(-0.10, min(0.20, _shrink(_mean(ds), prior_drift, len(ds), strength=3.0)))

    # (e) top-cap growth rate: the mean 10-min change of the top-5 caps
    # over the LAST TWO transitions (the late regime).  This is the
    # growth the size centres use; it is measured, not assumed.  Pool
    # incumbents that persist grow smoothly (~+0.10 dex per 10 min in
    # the late practice).  Only the last two transitions are used
    # because the early marks were far more volatile -- the top-5 was
    # still filling in from ~2.0 dex (shifts of +0.5 to +0.9 per 10
    # min), which would inflate the growth rate and push every centre
    # far too high.
    max_cap = max(last_by_mint.values()) if last_by_mint else 3.0
    top_shifts = []
    for i in range(max(0, len(marks) - 3), len(marks) - 1):
        prev_top = [math.log10(int(r["market_cap_lamports"])) - 9.0
                    for r in top_of(marks[i]) if int(r["market_cap_lamports"]) > 0]
        cur_top = [math.log10(int(r["market_cap_lamports"])) - 9.0
                   for r in top_of(marks[i + 1]) if int(r["market_cap_lamports"]) > 0]
        for k in range(min(len(prev_top), len(cur_top))):
            top_shifts.append(cur_top[k] - prev_top[k])
    top_growth = _shrink(_mean(top_shifts), PRIOR_GROWTH, len(top_shifts), strength=3.0)
    top_growth = max(0.0, min(0.15, top_growth))
    # cut place caps (the centres the size model grows from)
    cut_top_caps = [math.log10(int(r["market_cap_lamports"])) - 9.0
                    for r in top_of(marks[-1]) if int(r["market_cap_lamports"]) > 0]
    while len(cut_top_caps) < places:
        cut_top_caps.append(max_cap if cut_top_caps else 3.0)

    # (f) growth ceiling: set so the saturating growth's INITIAL slope
    # equals the measured top-cap growth rate (top_growth per 10 min =
    # top_growth*6 per hour).  A saturating curve starting at that slope
    # and levelling at the ceiling tracks the market's fast early growth
    # and its later saturation, on both the dev (1 h) and the final
    # (7-13 h) horizons.
    growth_ceil = max_cap + max(0.30, top_growth * 6.0 * GROWTH_TAU_H)

    # ---- candidate selection -------------------------------------------------
    def momentum(e):
        if len(e["x"]) < 2:
            return 0.0
        return e["x"][-1] - e["x"][0]

    scored = []
    for mint, e in series.items():
        x = e["x"][-1]
        scored.append((x + 0.35 * momentum(e), mint))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    cand_mints = [mint for _, mint in scored[:MAX_CANDIDATES]]
    top20 = sorted(last_by_mint.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    for mint, _ in top20:
        if mint not in cand_mints:
            cand_mints.append(mint)
    cand_mints.sort()

    # ---- per-coin model parameters -------------------------------------------
    # Same three regimes as v2 (the data supports them: high-cap pool coins
    # persist and grow, young coins collapse, the rest decay slowly) but the
    # hazard and growth now come from the measured turnover/volatility
    # instead of fixed constants.  A winner's death hazard is the measured
    # turnover scaled down for a high, rising cap; a young coin's hazard is
    # the measured turnover scaled up (young coins are the ones turning
    # over); the rest sit between.
    coins = []
    for mint in cand_mints:
        e = series[mint]
        xs = e["x"]
        ts = e["t"]
        x = xs[-1]
        venue = e["venue"]
        if e["created"]:
            age_min = (cut_time - parse_time(e["created"])).total_seconds() / 60.0
        else:
            age_min = 999.0
        n = len(xs)
        if n >= 2:
            mt = sum(ts) / n
            mx = sum(xs) / n
            num = sum((t - mt) * (xv - mx) for t, xv in zip(ts, xs))
            den = sum((t - mt) ** 2 for t in ts)
            slope_h = num / den if den > 0 else 0.0
        else:
            slope_h = 0.0
        slope_10 = slope_h / 6.0
        # Calibrated volatility: the venue's measured 10-min sd, scaled to
        # an hour.  Pool coins are far more volatile than curve coins.
        vol_hourly = min(0.60, max(0.10, vol.get(venue, 0.20) * math.sqrt(6.0)))
        is_winner = (x >= 3.0 and slope_10 > -0.15) or (n >= 3 and slope_10 > 0.008)
        if is_winner:
            # winners turn over at roughly the measured turnover rate, but a
            # high rising cap protects them: hazard well below turnover.
            hazard = max(0.02, min(0.30, turnover * 0.35))
            regime = "winner"
        elif age_min < 15.0 or n <= 2:
            # young coins are the turnover: hazard near the measured rate.
            hazard = max(0.30, min(0.95, turnover * 3.0))
            if e["complete"]:
                hazard *= 0.8
            regime = "young"
        else:
            hazard = max(0.05, min(0.40, turnover * 0.8))
            regime = "other"
        coins.append({"mint": mint, "regime": regime, "x": x,
                      "hazard": hazard, "vol_hourly": vol_hourly,
                      "drift_h": drift.get(venue, 0.0) * 6.0,
                      "venue": venue})

    # ---- unnamed-mass target -------------------------------------------------
    # u(h): the probability a place is won by a coin born after the cut.
    # Calibrated from the measured turnover: each 10-min step turns over
    # `turnover` of the places, and the newborn share of that turnover is
    # what the history's (cut, read) pairs show; we target the cumulative
    # newborn share, growing with horizon and capped at U_MAX.  v3
    # saturated at 2x the per-step turnover (0.40), but the history's
    # (cut, read) pairs show the newborn share is ~0.50 at 10 min and
    # 0.67-0.80 at 20-40 min, and the practice's own last read (70 min)
    # had 3 of 5 places won by post-cut coins (0.60), so v4 saturates at
    # 3x the turnover and U_MAX 0.65.
    def u_target(h_min):
        if h_min <= 0:
            return 0.0
        # The share grows roughly linearly with the number of 10-min
        # steps (each step turns over `turnover` of the places, and in
        # the late regime that turnover is essentially all post-cut
        # newborns), but it SATURATES: once strong post-cut newborns hold
        # the top places they become stable incumbents and the effective
        # turnover falls.  A compounding 1-(1-turnover)^steps reaches
        # 0.79 by 70 min; the saturation level is ~3x the per-step
        # turnover, matching the ~0.60 the practice actually had.  Cap it.
        u = turnover * (h_min / 10.0)
        return min(U_MAX, min(3.0 * turnover, u))

    # ---- Monte Carlo over the reads ------------------------------------------
    rng = random.Random(SEED)
    n_cand = len(coins)
    n_reads = len(horizons)
    place_counts = []   # per read, per place: {mint: count} (named wins)
    size_named = []     # per read, per place: [bin counts] (named wins)
    for ri, h in enumerate(horizons):
        h_min = h * 60.0
        # Strong newborns: the calibrated rate per 10-min cohort since the
        # cut, capped so the field stays bounded.  A read at the cut has
        # none; the count grows with the number of cohorts.
        n_newborn = 0 if h_min < 5 else min(12, int(round(nb_rate * (h_min / 10.0))))
        pc = [dict() for _ in range(places)]
        sc = [[0] * n_bins for _ in range(places)]
        mus = []
        sds = []
        dead_p = []
        for c in coins:
            steps = h * 6.0  # 10-minute steps
            p_dead = 1.0 - (1.0 - c["hazard"]) ** steps if steps > 0 else 0.0
            if c["regime"] == "winner":
                # saturating growth toward the calibrated ceiling.
                mu = growth_ceil - (growth_ceil - c["x"]) * math.exp(-h / 1.0)
                # add the measured drift while far below the ceiling
                mu = min(mu, c["x"] + max(0.0, c["drift_h"]) * min(h, 2.0) + 0.10)
                sd = math.sqrt(SIGMA0 * SIGMA0 + (c["vol_hourly"] ** 2) * h)
            else:
                mu = c["x"] + c["drift_h"] * min(h, 2.0) - 0.08 * h
                sd = math.sqrt(SIGMA0 * SIGMA0 + (c["vol_hourly"] ** 2) * h)
            mus.append(mu)
            sds.append(min(SIGMA_CAP, sd))
            dead_p.append(p_dead)
        for _s in range(SCENARIOS):
            caps = []
            for i in range(n_cand):
                if dead_p[i] > 0.0 and rng.random() < dead_p[i]:
                    caps.append((rng.gauss(DEAD_CAP, DEAD_SD), i))
                else:
                    caps.append((rng.gauss(mus[i], sds[i]), i))
            for _n in range(n_newborn):
                caps.append((rng.gauss(nb_mu, nb_sd), -1))
            caps.sort(key=lambda kv: -kv[0])
            for placed in range(places):
                cap_val, i = caps[placed]
                if i >= 0:
                    pc[placed][coins[i]["mint"]] = pc[placed].get(coins[i]["mint"], 0) + 1
                    b = bin_of(cap_val, edges)
                    sc[placed][b] += 1
        place_counts.append(pc)
        size_named.append(sc)

    # ---- average over reads, build the forecast ------------------------------
    probabilities = []
    sizes = []
    top5 = []
    for k in range(places):
        # Mean MC frequency per named coin at this place, and the MC's own
        # unnamed mass per read (for reference; the target overrides it).
        freq = {}
        mc_unnamed = []
        for r in range(n_reads):
            named_total = sum(place_counts[r][k].values())
            mc_unnamed.append(1.0 - named_total / SCENARIOS)
            for mint, cnt in place_counts[r][k].items():
                freq[mint] = freq.get(mint, 0.0) + cnt / (SCENARIOS * n_reads)
        # Mean unnamed target over the reads (each read has its own
        # horizon, hence its own target).
        u_mean = _mean([u_target(h * 60.0) for h in horizons])
        # Rescale the named frequencies so the unnamed mass equals the
        # target.  The ranking among named coins is unchanged; only the
        # named-vs-newborn split is set by calibration.  If the MC already
        # produces *more* unnamed mass than the target (possible at very
        # long horizons), keep the MC's (do not shrink the unnamed mass).
        s_named = sum(freq.values())
        u_mc_mean = _mean(mc_unnamed)
        # The calibrated target is the reliable evidence at dev-length
        # horizons: it is measured from the turnover, and the Monte
        # Carlo's own unnamed mass overestimates there (at a 10-min read
        # the MC gave 0.35 where the measured target is 0.20, because its
        # newborn field is too strong against the incumbents that still
        # dominate).  So the target wins at short horizons.  Only at very
        # long horizons (mean read > 2 h, as in the final) does the MC's
        # value stand: by then the cut's incumbents are mostly dead, the
        # measured turnover's saturation level is too low, and the MC's
        # death dynamics are the evidence.  Cap the total either way.
        mean_h_min = _mean([h * 60.0 for h in horizons])
        if mean_h_min > 120.0:
            u_final = min(U_MAX, max(u_mean, u_mc_mean))
        else:
            u_final = min(U_MAX, u_mean)
        order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        named = [mint for mint, _ in order[:MAX_ROWS_PER_PLACE]]
        # v4: name the top extra candidates too (by momentum score).
        # Naming more coins shrinks the pool the unnamed mass is shared
        # over, so a newborn that takes a place scores the unnamed share
        # over fewer coins, and the extra coins carry real (if small)
        # probability from the smoothing instead of 0.
        seen_named = set(named)
        for _, mint in scored:
            if len(named) >= MAX_ROWS_PER_PLACE:
                break
            if len(named) >= len(order) + NAMED_EXTRA:
                break
            if mint not in seen_named:
                named.append(mint)
                seen_named.add(mint)
        n_named = len(named)
        for mint in named:
            # Named probabilities sum to (1 - u_final); the smoothing
            # spreads over that named share.  Relative order among named
            # coins is the MC's; only the named-vs-newborn split is set.
            if s_named > 0 and n_named:
                p = (1.0 - u_final) * ((1.0 - SMOOTH_PLACE) * freq.get(mint, 0.0) / s_named
                                       + SMOOTH_PLACE / n_named)
            else:
                p = 0.0
            p = floor6(p)
            if p > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": p})
        for mint, _ in order:
            if mint not in top5:
                top5.append(mint)
                break
        # ---- sizes: tight per-read mixture (v4) ---------------------------
        # v3 built sizes from the Monte Carlo histogram of winning
        # incumbents' caps plus one pooled newborn normal.  That histogram
        # is far too wide (sd 0.56-0.69 dex per place) because each coin's
        # cap sd in the MC is ~0.60 dex at one hour.  But the actual
        # top-5 caps at a given horizon are tightly clustered (sd
        # 0.03-0.13 dex across (cut, read) pairs): the top of the market
        # is a crowd, persisting coins grow smoothly, and a crash just
        # hands the place to the next coin at nearly the same cap.
        # So v4 builds each place's size distribution directly, per read,
        # as a moment-matched normal mixture of two tight components:
        #   incumbent: centre = the k-th largest of the cut's top-20 caps
        #     grown toward the ceiling (saturating growth); sd from the
        #     measured place-cap spread, growing slowly with horizon.
        #   newborn:   centre = this place's newborn mean (recency
        #     weighted), grown from its reference horizon toward the
        #     ceiling; sd = this place's measured newborn sd.
        # The two are mixed by the PER-READ unnamed target u(h) (not the
        # mean: later reads, where newborns dominate, carry more newborn
        # weight), the mixture's mean and variance are computed exactly,
        # and the per-read normals are averaged over the reads.  A small
        # broad prior is blended in for robustness, so a centre that is
        # off does not score the floor.
        tot = [0.0] * n_bins
        top20_caps = sorted(last_by_mint.values(), reverse=True)[:20]
        mu_acc = 0.0
        for r in range(n_reads):
            h = horizons[r]
            u_r = u_target(h * 60.0)
            # incumbent centre: k-th largest of the grown top-20 caps
            grown = []
            for x in top20_caps:
                g = growth_ceil - (growth_ceil - x) * math.exp(-h / GROWTH_TAU_H)
                grown.append(g)
            grown.sort(reverse=True)
            inc_c = grown[k] if k < len(grown) else (cut_top_caps[k] if k < len(cut_top_caps) else max_cap)
            inc_sd = min(SIZE_SD_CAP, math.sqrt(SIZE_SD0 * SIZE_SD0 + (SIZE_SD_H * h) ** 2))
            # newborn centre: this place's newborn mean, grown from its
            # reference horizon toward the ceiling
            dh = max(0.0, h - nb_h_ref_mean)
            nb_c = growth_ceil - (growth_ceil - nb_mu_k[k]) * math.exp(-dh / GROWTH_TAU_H)
            nb_sd_r = nb_sd_k[k]
            # moment-matched mixture of the two normals
            mu_r = (1.0 - u_r) * inc_c + u_r * nb_c
            var_r = ((1.0 - u_r) * inc_sd * inc_sd + u_r * nb_sd_r * nb_sd_r
                     + u_r * (1.0 - u_r) * (inc_c - nb_c) ** 2)
            sd_r = math.sqrt(max(0.01, var_r))
            mu_acc += mu_r
            for b in range(n_bins):
                lo = -float("inf") if b == 0 else edges[b - 1]
                hi = float("inf") if b == n_bins - 1 else edges[b]
                p_r = normal_cdf(hi, mu_r, sd_r) - normal_cdf(lo, mu_r, sd_r)
                tot[b] += p_r / n_reads
        # broad prior, centred on the mixture mean, for robustness
        mu_size = mu_acc / n_reads if n_reads > 0 else nb_mu
        for b in range(n_bins):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == n_bins - 1 else edges[b]
            prior = normal_cdf(hi, mu_size, SIZE_PRIOR_SD) - normal_cdf(lo, mu_size, SIZE_PRIOR_SD)
            p = (1.0 - SIZE_PRIOR_WEIGHT) * tot[b] + SIZE_PRIOR_WEIGHT * prior
            p = floor6(p)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    if len(top5) < places:
        for mint in cand_mints:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    doc = {"probabilities": probabilities, "top5": top5[:places], "sizes": sizes}
    text = json.dumps(doc, sort_keys=True) + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
