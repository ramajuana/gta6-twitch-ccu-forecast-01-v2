"""Forecast GTA VI launch-day Twitch viewers, hours 2..24, from hour 1's print.

Model (own design, validated by leave-one-out and chronological holdouts
on the comparables): an adaptive, bootstrap-averaged ensemble of three
per-hour forecasters that all predict log-ratios log(viewers_h / viewers_1):

  * A kernel forecaster: comparables are weighted by a Gaussian kernel
    (bandwidth 1.2) on the distance between their hour-1 print and this
    case's; comparables whose early-access status contradicts the case's
    are dropped. Each hour's centre is the weighted median log-ratio and
    its spread the weighted IQR / 1.349, floored at 0.25, both smoothed
    over a 3-hour window.
  * Two local-linear ridge forecasters (bandwidth 1.5 on log hour-1,
    ridge 0.3): per hour, a weighted ridge fit of the log-ratios on
    features centred at the case (log hour-1, three pre-release level
    features from the 24 hours before the release day, early-access
    flag, PC-platform flag). One takes its quantiles from the weighted
    residual quantiles, the other from the weighted residual spread
    under a normal shape.
  * The blend is adaptive: when the case's hour-1 print sits inside the
    comparables' interquartile range the kernel takes weight 0.65 and
    the regressions share 0.35; outside that range (where GTA VI itself
    sits, bigger than almost every comparable) the kernel drops to 0.35
    because the regressions extrapolate better at the edge.

  The whole ensemble is bootstrap-averaged: 25 resamples of the
  comparable pool for each of five fixed seeds, forecasts averaged in
  log space, so the output is deterministic and stable.

  Validation on the 37 comparables (mean log pinball, lower is better;
  the previous kernel-only entry in brackets): leave-one-out 0.2457 on
  the large launches and 0.3860 overall [0.2605 / 0.4055];
  chronological holdout 0.2821 / 0.4570 [0.3181 / 0.4685]; reversed
  chronological holdout 0.2162 / 0.2620 [0.2190 / 0.2796]; the six
  largest launches held out together 0.1471 [0.1975].

Fallbacks mirror the housekeeping any forecaster needs: too few usable
comparables, or any failure while forecasting, degrades to the flat
persistence shape (median = hour 1's print, spread 0.15 * sqrt(h - 1))
instead of crashing. Standard library only, no network.
"""

import bisect
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))
FEATURES = ["logp1", "pre1", "pre13", "prepeak", "ea", "has_pc"]

KERNEL_BW = 1.2        # kernel bandwidth on log hour-1 distance
REG_BW = 1.5           # local weight bandwidth for the ridge fits
REG_LAM = 0.3          # ridge penalty (never on the intercept)
SIGMA_FLOOR = 0.25     # floor on every log-spread
MEDIAN_FLOOR = 0.25    # median ratio floor, as a share of hour 1
KERNEL_W_IN = 0.65     # kernel blend weight inside the pool's IQR
KERNEL_W_OUT = 0.35    # kernel blend weight outside it
BAGGING_B = 25         # bootstrap resamples per seed
BAGGING_SEEDS = [20261117, 20261118, 20261119, 20261120, 20261121]
MIN_DAY_HOURS = 20     # a comparable's release day must be this complete
MIN_POOL = 5           # usable comparables needed before modelling
BASE_SIGMA = 0.15      # fallback spread per square-root hour


def _parse_time(text):
    """Parse an ISO-8601 timestamp (trailing Z allowed) to an aware datetime."""
    if not isinstance(text, str) or not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def _clamp_row(qs):
    """Force quantiles into (0, 1e9] and non-decreasing."""
    out = []
    prev = 0.0
    for q in qs:
        if not math.isfinite(q) or q <= 0:
            q = prev if prev > 0 else 1.0
        q = min(q, 1e9)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def _baseline_rows(p1, hours=None):
    """Flat persistence forecast: every median is hour 1's print."""
    hours = hours or list(range(2, 25))
    rows = []
    for h in hours:
        sigma = BASE_SIGMA * math.sqrt(h - 1)
        qs = [_round2(p1 * math.exp(z * sigma)) for z in Z]
        rows.append({"hour": h, "q": _clamp_row(qs)})
    return {"hours": rows}


def _forecast_hours(case):
    """The hours this case asks for (2..24 per the rules), in order."""
    hours = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if isinstance(h, int) and 2 <= h <= 24 and h not in hours:
            hours.append(h)
    return sorted(hours) if hours else list(range(2, 25))


def _hour1_print(case):
    """Hour-1 print, with fallbacks from the hourly history, else 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    history = case.get("history_hourly") or []
    if history:
        last = history[-1]
        viewers = last.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            return float(viewers)
    return 1.0


def _release_day(comp):
    """Map a comparable to {hour: viewers} for its 24 release-day hours.

    Returns None when the comparable has no usable hour-1 print or too
    few release-day hours (e.g. the GTA VI category's own pre-release
    history, which has no release day yet).
    """
    day_start = _parse_time(comp.get("day_start"))
    if day_start is None:
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        when = _parse_time(point.get("from"))
        viewers = point.get("viewers")
        if when is None or not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and hour not in by_hour:
            by_hour[hour] = float(viewers)
    if 1 not in by_hour or len(by_hour) < MIN_DAY_HOURS:
        return None
    return by_hour


def _text_ea(text):
    """Early-access flag from an unlock/notes text: 1 yes, 0 explicit no,
    None when the text does not say. A 'early-access-contaminated' tier
    label in the text forces 1."""
    raw = str(text or "")
    t = " " + raw.lower().replace("-", " ") + " "
    if "no early access" in t:
        ea = 0
    elif "early access" in t:
        ea = 1
    else:
        ea = None
    if "early access contaminated" in t:
        ea = 1
    return ea


def _pre_features(points, day_start, p1):
    """Pre-release level features from the 24 hours before day_start.

    points is an iterable of (datetime, viewers). Features are logs of
    ratios to p1: the last pre-hour (pre1), the mean of the last 13
    (pre13), the peak (prepeak), and the recent-vs-earlier slope
    (preslope). Returns None when fewer than 8 pre-hours are present.
    """
    if day_start is None:
        return None
    rel = {}
    for when, viewers in points:
        if when is None:
            continue
        h = int((when - day_start).total_seconds() // 3600)
        if -24 <= h <= -1 and h not in rel:
            rel[h] = float(viewers)
    if len(rel) < 8:
        return None

    def mean_of(lo, hi):
        vals = [rel[h] for h in range(lo, hi + 1) if h in rel]
        return sum(vals) / len(vals) if vals else None

    last = rel.get(-1)
    m13 = mean_of(-13, -1)
    m6 = mean_of(-6, -1)
    mrest = mean_of(-24, -13)
    peak = max(rel.values())
    if last is None or m13 is None or m6 is None or mrest is None:
        return None
    return {
        "pre1": math.log(last / p1),
        "pre13": math.log(m13 / p1),
        "prepeak": math.log(peak / p1),
        "preslope": math.log(m6 / mrest),
    }


def _hourly_points(rows):
    """(datetime, viewers) pairs from a case/comparable hourly list."""
    points = []
    for point in rows or []:
        viewers = point.get("viewers")
        if not isinstance(viewers, (int, float)) or viewers <= 0:
            continue
        points.append((_parse_time(point.get("from")), float(viewers)))
    return points


def _member_from_comp(comp):
    """A modelling member from one comparable, or None when unusable."""
    day = _release_day(comp)
    if day is None:
        return None
    pre = _pre_features(_hourly_points(comp.get("hourly")),
                        _parse_time(comp.get("day_start")), day[1])
    if pre is None:
        return None
    return {
        "day": day,
        "logp1": math.log(day[1]),
        "pre": pre,
        "ea": _text_ea(comp.get("unlock")),
        "has_pc": 1 if "PC" in str(comp.get("platforms") or "") else 0,
    }


def _has_pc_from_notes(notes):
    """PC-at-launch flag for the case's own game, from its notes: read the
    platform list right after 'released ... on' (stopping at a bracket,
    semicolon or full stop, so 'console only; PC 2019' stays a no)."""
    match = re.search(r"released[^.]*?\bon\b([^();.]*)", str(notes or ""),
                      re.IGNORECASE)
    if not match:
        return 0
    return 1 if re.search(r"\bPC\b", match.group(1)) else 0


def _target_from_case(case, p1, train):
    """The case itself as a modelling target. Pre-release features come
    from the case's own hourly history; when they cannot be formed they
    are imputed with the pool's medians so the model still runs."""
    pre = _pre_features(_hourly_points(case.get("history_hourly")),
                        _parse_time(case.get("day_start")), p1)
    if pre is None:
        pre = {}
        for feat in ("pre1", "pre13", "prepeak", "preslope"):
            vals = sorted(m["pre"][feat] for m in train)
            mid = len(vals) // 2
            pre[feat] = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    return {
        "logp1": math.log(p1),
        "pre": pre,
        "ea": _text_ea(case.get("notes")),
        "has_pc": _has_pc_from_notes(case.get("notes")),
    }


def _wq(sorted_vals, cumw, total, q):
    """Step weighted quantile: first value whose cumulative weight
    reaches q * total (the middle value when every weight is zero)."""
    if total <= 0:
        return sorted_vals[len(sorted_vals) // 2]
    idx = bisect.bisect_left(cumw, q * total)
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


def _sorted_wq_pairs(pairs):
    """Sort (value, weight) pairs by value; return values, cumulative
    weights and the total weight."""
    pairs = sorted(pairs, key=lambda p: p[0])
    vals = [p[0] for p in pairs]
    cumw = []
    acc = 0.0
    for _, w in pairs:
        acc += w
        cumw.append(acc)
    return vals, cumw, acc


def _smooth3(arr):
    """3-point moving average, edges using what is available."""
    n = len(arr)
    out = []
    for i in range(n):
        seg = arr[max(0, i - 1):min(n, i + 2)]
        out.append(sum(seg) / len(seg))
    return out


def _log_ratios(member):
    day = member["day"]
    p1 = day[1]
    return [math.log(day[h] / p1) if h in day else None for h in HOURS]


def _kernel_fc(target, train):
    """Kernel forecaster: weighted median / IQR spread of log-ratios."""
    t_ea = target["ea"]
    weights = []
    for t in train:
        if t_ea is not None and t["ea"] is not None and t["ea"] != t_ea:
            weights.append(0.0)
            continue
        d = (t["logp1"] - target["logp1"]) / KERNEL_BW
        weights.append(math.exp(-0.5 * d * d))
    lrs = [_log_ratios(t) for t in train]
    med = [0.0] * 23
    sig = [0.0] * 23
    for j in range(23):
        pairs = [(lrs[i][j], weights[i]) for i in range(len(train))
                 if lrs[i][j] is not None]
        vals, cumw, total = _sorted_wq_pairs(pairs)
        med[j] = _wq(vals, cumw, total, 0.5)
        q25 = _wq(vals, cumw, total, 0.25)
        q75 = _wq(vals, cumw, total, 0.75)
        sig[j] = max(SIGMA_FLOOR, (q75 - q25) / 1.349)
    med_s = _smooth3(med)
    sig_s = [max(s, SIGMA_FLOOR) for s in _smooth3(sig)]
    out = {}
    for j, h in enumerate(HOURS):
        med_adj = math.log(max(math.exp(med_s[j]), MEDIAN_FLOOR))
        out[h] = [med_adj + z * sig_s[j] for z in Z]
    return out


def _solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f != 0.0:
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][c] * x[c] for c in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def _featvec(t, target):
    """A member's features centred at the target (regression inputs)."""
    t_ea = target["ea"]
    if t_ea is not None:
        ea = (t["ea"] if t["ea"] is not None else t_ea) - t_ea
    else:
        ea = 0.0
    return {
        "logp1": t["logp1"] - target["logp1"],
        "pre1": t["pre"]["pre1"] - target["pre"]["pre1"],
        "pre13": t["pre"]["pre13"] - target["pre"]["pre13"],
        "prepeak": t["pre"]["prepeak"] - target["pre"]["prepeak"],
        "ea": ea,
        "has_pc": t["has_pc"] - target["has_pc"],
    }


def _reg_fc(target, train):
    """Local-linear ridge forecaster, in both quantile flavours at once:
    direct (weighted residual quantiles) and normal (residual spread)."""
    X = []
    for t in train:
        fv = _featvec(t, target)
        X.append([1.0] + [fv[f] for f in FEATURES])
    kw = []
    for t in train:
        d = (t["logp1"] - target["logp1"]) / REG_BW
        kw.append(math.exp(-0.5 * d * d))
    lrs = [_log_ratios(t) for t in train]
    k = len(FEATURES) + 1
    med = [0.0] * 23
    sig = [0.0] * 23
    direct = [[0.0] * 5 for _ in range(23)]
    for j in range(23):
        ok = [i for i in range(len(train)) if lrs[i][j] is not None]
        wsum = sum(kw[i] for i in ok)
        A = [[0.0] * k for _ in range(k)]
        b = [0.0] * k
        for i in ok:
            xi = X[i]
            wi = kw[i] / max(wsum, 1e-300)
            yi = lrs[i][j]
            for a in range(k):
                b[a] += wi * xi[a] * yi
                xia = xi[a] * wi
                for c in range(k):
                    A[a][c] += xia * xi[c]
        for d in range(1, k):
            A[d][d] += REG_LAM
        beta = _solve(A, b)
        mu = beta[0]
        pairs = []
        for i in ok:
            fit = sum(X[i][a] * beta[a] for a in range(k))
            pairs.append((lrs[i][j] - fit + mu, kw[i]))
        vals, cumw, total = _sorted_wq_pairs(pairs)
        med[j] = _wq(vals, cumw, total, 0.5)
        for m, tau in enumerate(LEVELS):
            direct[j][m] = _wq(vals, cumw, total, tau)
        q25 = _wq(vals, cumw, total, 0.25)
        q75 = _wq(vals, cumw, total, 0.75)
        sig[j] = max(SIGMA_FLOOR, (q75 - q25) / 1.349)
    med_s = _smooth3(med)
    sig_s = [max(s, SIGMA_FLOOR) for s in _smooth3(sig)]
    direct_s = [[0.0] * 5 for _ in range(23)]
    for m in range(5):
        col = _smooth3([direct[j][m] for j in range(23)])
        for j in range(23):
            direct_s[j][m] = col[j]
    out_d = {}
    out_n = {}
    for j, h in enumerate(HOURS):
        med_adj = math.log(max(math.exp(med_s[j]), MEDIAN_FLOOR))
        shift = med_adj - med_s[j]
        qs = []
        run = None
        for m in range(5):
            v = direct_s[j][m] + shift
            run = v if run is None or v > run else run
            qs.append(run)
        out_d[h] = qs
        out_n[h] = [med_adj + z * sig_s[j] for z in Z]
    return out_d, out_n


def _quantile(sorted_vals, q):
    """Plain quantile with linear interpolation (numpy's default)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])


def _adaptive_fc(target, train):
    """Blend the three forecasters; the kernel's share depends on whether
    the target's size sits inside the pool's interquartile range."""
    lp = sorted(t["logp1"] for t in train)
    q25 = _quantile(lp, 0.25)
    q75 = _quantile(lp, 0.75)
    edge = target["logp1"] > q75 or target["logp1"] < q25
    ak = KERNEL_W_OUT if edge else KERNEL_W_IN
    w_reg = (1.0 - ak) / 2.0
    fk = _kernel_fc(target, train)
    frd, frn = _reg_fc(target, train)
    out = {}
    for h in HOURS:
        out[h] = [ak * fk[h][m] + w_reg * frd[h][m] + w_reg * frn[h][m]
                  for m in range(5)]
    return out


def _bagged_fc(target, train):
    """Bootstrap-average the adaptive ensemble over the fixed seeds."""
    n = len(train)
    acc = {h: [0.0] * 5 for h in HOURS}
    reps = 0
    for seed in BAGGING_SEEDS:
        rng = random.Random(seed)
        for _ in range(BAGGING_B):
            idx = [rng.randrange(n) for _ in range(n)]
            sub = [train[i] for i in idx]
            fc = _adaptive_fc(target, sub)
            for h in HOURS:
                for m in range(5):
                    acc[h][m] += fc[h][m]
            reps += 1
    return {h: [v / reps for v in acc[h]] for h in HOURS}


def forecast_case(case):
    """Forecast hours 2..24 for one case dict; never raises."""
    try:
        return _forecast_case(case)
    except Exception:
        try:
            p1 = _hour1_print(case)
            hours = _forecast_hours(case)
        except Exception:
            p1 = 1.0
            hours = list(range(2, 25))
        return _baseline_rows(max(1.0, p1), hours)


def _forecast_case(case):
    p1 = max(1.0, _hour1_print(case))
    hours = _forecast_hours(case)
    train = []
    for comp in case.get("comparables") or []:
        member = _member_from_comp(comp)
        if member is not None:
            train.append(member)
    if len(train) < MIN_POOL:
        return _baseline_rows(p1, hours)
    target = _target_from_case(case, p1, train)
    fc = _bagged_fc(target, train)
    rows = []
    for h in hours:
        qs = [_round2(p1 * math.exp(v)) for v in fc[h]]
        rows.append({"hour": h, "q": _clamp_row(qs)})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = _baseline_rows(1.0)
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
