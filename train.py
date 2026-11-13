"""Real diamond data: baseline -> models -> monotonic constraint -> CONFORMAL prediction interval -> segment-wise coverage check"""
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import common as C

df, rep = C.load()
print(f"raw {rep['raw_rows']} | bad values dropped {rep['dropped_bad_values']} | duplicate rows removed {rep['duplicates_removed']} | clean {rep['clean_rows']}")
raw_twin = pd.read_csv("data/diamonds.csv")
tr0, te0 = train_test_split(raw_twin, test_size=0.2, random_state=1)
twin = te0.merge(tr0.drop_duplicates(), on=list(raw_twin.columns), how="inner").shape[0] / len(te0)
print(f"dedupe na korle test row er {twin:.2%} er hubohu twin train e thakto (leakage)\n")

# ---- Split: train 60% | calibration 20% (conformal er jonno) | test 20% (shesh e ekbar) ----
train, rest = train_test_split(df, test_size=0.4, random_state=42)
cal, test = train_test_split(rest, test_size=0.5, random_state=42)
Xtr, Xca, Xte = train[C.FEATURES], cal[C.FEATURES], test[C.FEATURES]
ytr, yca, yte = np.log(train.price.values), np.log(cal.price.values), np.log(test.price.values)
P_te = test.price.values


def hgb(**kw):
    return HGB(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, random_state=0, **kw)


def point_metrics(pred_log):
    p = np.exp(pred_log)
    ape = np.abs(p - P_te) / P_te
    return {"median_ape": float(np.median(ape)), "mean_ape": float(ape.mean()), "mae_usd": float(np.abs(p - P_te).mean()),
            "within_10pct": float((ape <= 0.10).mean()), "r2_log": float(1 - ((yte - pred_log) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())}


# ---- 1. Point models ----
ppc = (train.price / train.carat).groupby(train.clarity).median()            # baseline: clarity onujayi price/carat
base = np.log((test.carat * test.clarity.map(ppc)).values)
lin = LinearRegression().fit(np.column_stack([np.log(Xtr.carat), Xtr.drop(columns="carat")]), ytr)
lin_p = lin.predict(np.column_stack([np.log(Xte.carat), Xte.drop(columns="carat")]))
free = hgb().fit(Xtr, ytr)
mono = hgb(monotonic_cst=C.MONO).fit(Xtr, ytr)
models = {"baseline_price_per_carat": point_metrics(base), "linear_log": point_metrics(lin_p),
          "hgb_unconstrained": point_metrics(free.predict(Xte)), "hgb_monotonic": point_metrics(mono.predict(Xte))}
print(f"{'model':26s} {'median APE':>10s} {'MAE $':>8s} {'within10%':>10s}")
for k, v in models.items():
    print(f"{k:26s} {v['median_ape']:10.1%} {v['mae_usd']:8.0f} {v['within_10pct']:10.1%}")

# ---- 2. Monotonicity check: carat 0.1 barale dam komle 'shonkha' (sense hoy na) ----
sample = test.sample(3000, random_state=0)[C.FEATURES].copy()
bumped = sample.copy(); bumped["carat"] += 0.1
viol = {k: float((m.predict(bumped) < m.predict(sample) - 1e-9).mean()) for k, m in (("unconstrained", free), ("monotonic", mono))}
print(f"\ncarat barale dam kome jabe (violation rate): unconstrained {viol['unconstrained']:.1%} | monotonic {viol['monotonic']:.1%}")

# ---- 3. Prediction intervals: 3 ta upay compare ----
qlo = hgb(loss="quantile", quantile=C.ALPHA / 2, monotonic_cst=C.MONO).fit(Xtr, ytr)
qhi = hgb(loss="quantile", quantile=1 - C.ALPHA / 2, monotonic_cst=C.MONO).fit(Xtr, ytr)
# Conformalized Quantile Regression: calibration set e 'koto bhul hoy' mepe interval boro kori
lo_c, hi_c = qlo.predict(Xca), qhi.predict(Xca)
scores = np.maximum(lo_c - yca, yca - hi_c)
n = len(scores)
qhat = float(np.sort(scores)[min(math.ceil((n + 1) * (1 - C.ALPHA)), n) - 1])
mu_te = mono.predict(Xte)
lo_t, hi_t = qlo.predict(Xte), qhi.predict(Xte)
sd = float(np.std(yca - mono.predict(Xca)))
intervals = {"normal_residual (+-1.645 sd)": (mu_te - 1.645 * sd, mu_te + 1.645 * sd),
             "raw_quantile_regression": (lo_t, hi_t),
             "conformal_CQR": (np.minimum(lo_t - qhat, mu_te), np.maximum(hi_t + qhat, mu_te))}

segments = {"carat": pd.qcut(test.carat, 4, duplicates="drop").astype(str),
            "cut": test.cut, "predicted_price": pd.qcut(mu_te, 4, labels=["Q1 sosta", "Q2", "Q3", "Q4 dami"]).astype(str)}
cover = {}
print(f"\n{int((1 - C.ALPHA) * 100)}% interval er coverage (lokkho 90%) | shobcheye kharap segment e coverage:")
for name, (lo, hi) in intervals.items():
    hit = ((yte >= lo) & (yte <= hi))
    seg = {f"{s}={g}": float(hit[np.asarray(segments[s]) == g].mean()) for s in segments for g in sorted(set(np.asarray(segments[s])))}
    worst = min(seg, key=seg.get)
    cover[name] = {"overall": float(hit.mean()), "worst_segment": worst, "worst_segment_coverage": seg[worst],
                   "median_width_ratio": float(np.median(np.exp(hi - lo))), "segments": seg}
    print(f"  {name:30s} overall {hit.mean():.1%} | worst: {worst} {seg[worst]:.1%} | median high/low ratio {np.median(np.exp(hi - lo)):.2f}x")
print(f"conformal qhat (log) = {qhat:.4f}")

# ---- 4. Save ----
bundle = {"mean": mono, "lo": qlo, "hi": qhi, "qhat": qhat,
          "ranges": {"carat": [float(df.carat.min()), float(df.carat.max())], "carat_p99": float(df.carat.quantile(0.99))}}
joblib.dump(bundle, "model/appraiser.joblib")
json.dump({"data": rep, "twin_leak_without_dedupe": float(twin), "models": models, "monotonic_violation_rate": viol,
           "coverage": cover, "qhat_log": qhat, "n_train": len(train), "n_cal": len(cal), "n_test": len(test),
           "carat_p99": bundle["ranges"]["carat_p99"]}, open("model/metrics.json", "w"), indent=2)
print("\nsaved model/appraiser.joblib + metrics.json")
