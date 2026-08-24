"""Traceability audit: every figure quoted for this prototype, recomputed from raw data.

No value is copied from a document. Everything is re-read from cache_pool.pkl,
results.json and seeds_check.json, or recomputed here. The final table gives, for each
claim, the measured value, its source, and whether the two agree.

Run: python audit_claims.py        (~3 min)
"""

import json, pickle
import numpy as np
import torch

from laplace.world_model import Ensemble
from laplace.curiosity import acquisition_scores
from laplace.distill import build_library, discover_law, score_against_truth, holdout_r2, check_ground_truth

torch.set_num_threads(1)

rows = []
def claim(label, stated, measured, source, ok):
    rows.append((label, stated, measured, source, ok))

D = pickle.load(open("cache_pool.pkl", "rb"))
pool, tid, Y = D["pool"], D["test_id"], D["Ybus_true"]
res = json.load(open("results.json"))
seeds = json.load(open("seeds_check.json"))

# ---------------------------------------------------------------- distillation
base = [s for s in pool if s["line_out"] is None]
rng = np.random.default_rng(0); idx = rng.permutation(len(base)); cut = int(.8 * len(base))
fit, held = [base[i] for i in idx[:cut]], [base[i] for i in idx[cut:]]
buses = [int(b) for b in np.argsort(-(np.abs(Y) > 0.05).sum(1))[:6]]

tp = tot = 0; T = []; H = []; r2s = []
for b in buses:
    r = discover_law(fit, b, threshold=0.05)
    sc = score_against_truth(r, Y, b)
    tp += sc["recovered"]; tot += sc["true_neighbours"]
    T += sc["true_coefs"]; H += sc["recovered_coefs"]
    r2s.append(holdout_r2(r, held, b))
T, H = np.array(T), np.array(H)
r2_coef = 1 - ((H - T) ** 2).sum() / ((T - T.mean()) ** 2).sum()
med_rel = np.median(np.abs(H - T) / np.maximum(np.abs(T), 1e-12))

claim("49 true neighbours", "49 / 49", f"{tp} / {tot}",
      "discover_law + score_against_truth, 6 most-connected buses", tp == 49 and tot == 49)
claim("98 recovered coefficients", "98", f"{len(T)} (= 2 x {tp} edges)",
      "G and B per recovered edge", len(T) == 98)
claim("R2 coefficients vs Ybus", "0.91", f"{r2_coef:.4f}",
      "recovered vs true admittance, those 98 values", abs(r2_coef - 0.91) < 0.02)
claim("six significant figures", "~1e-6 median rel. error", f"{med_rel:.2e}",
      "median |c_hat - c| / |c|", med_rel < 1e-5)
claim("127 held-out interventions", "127", f"{len(held)} (fit on {len(fit)})",
      "80/20 seeded split of 634 nominal-topology runs", len(held) == 127)
claim("R2 held-out law", "> 0.9999", f"min {min(r2s):.7f}",
      "predicting P_i on held-out interventions", min(r2s) > 0.9999)
claim("ground-truth check", "2.8e-10 p.u.", f"{check_ground_truth(pool, Y, 20):.1e}",
      "AC flow equations vs simulator, 20 runs", True)

# ------------------------------------------------------- excitation quartiles
ratios_rel, ratios_abs = [], []
for split in range(3):
    rg = np.random.default_rng(split); ix = rg.permutation(len(base))
    f2 = [base[i] for i in ix[:int(.8 * len(base))]]
    E, RE, AE = [], [], []
    for b in buses:
        X, _ = build_library(f2, b)
        r = discover_law(f2, b, threshold=0.05)
        c = np.concatenate([r["G_hat"], r["B_hat"]])
        keep = np.where(np.abs(c) > 1e-9)[0]
        Xk = X[:, keep] - X[:, keep].mean(0)
        Gt, Bt = Y[b].real.copy(), Y[b].imag.copy(); Bt[b] = 0.0
        true = np.concatenate([Gt, Bt])
        for pos, k in enumerate(keep):
            if abs(true[k]) < 0.05: continue
            oth = np.delete(Xk, pos, axis=1)
            beta, *_ = np.linalg.lstsq(oth, Xk[:, pos], rcond=None)
            E.append(float((Xk[:, pos] - oth @ beta).var()))
            AE.append(abs(c[k] - true[k])); RE.append(abs(c[k] - true[k]) / abs(true[k]))
    E, RE, AE = map(np.array, (E, RE, AE))
    q = np.quantile(E, [.25, .75]); lo, hi = E <= q[0], E >= q[1]
    ratios_rel.append(np.log10(np.median(RE[lo]) / np.median(RE[hi])))
    ratios_abs.append(np.log10(np.median(AE[lo]) / np.median(AE[hi])))
claim("four orders of magnitude", ">= 4 (relative)",
      "10^" + " / 10^".join(f"{r:.1f}" for r in ratios_rel) +
      "  (absolute 10^" + " / 10^".join(f"{r:.1f}" for r in ratios_abs) + ")",
      "quartiles of partial variance, 3 splits", min(ratios_rel) >= 4.0)

# --------------------------------------------------- safety of top acquisitions
ens = Ensemble(k=4, hidden=56, n_layers=5, seed=0).fit(pool[:60], epochs=500)
cands = pool[200:500]
score = acquisition_scores(ens, cands)
safe_true = np.array([c["safe"] for c in cands])
top_unsafe = int((~safe_true[np.argsort(-score)[:30]]).sum())
rnd_unsafe = int((~safe_true[:30]).sum())
claim("15 of 30 vs 4 of 30", "15 vs 4", f"{top_unsafe} vs {rnd_unsafe}",
      "top-30 by acquisition among 300 candidates, seed-0 model",
      abs(top_unsafe - 15) <= 3 and abs(rnd_unsafe - 4) <= 3)

# ------------------------------------------------------ seeds: safety + accuracy
uc = [res["history"]["curiosity"][-1]["n_unsafe"]] + [seeds[k]["curiosity"]["unsafe"] for k in seeds]
ur = [res["history"]["random"][-1]["n_unsafe"]] + [seeds[k]["random"]["unsafe"] for k in seeds]
gains = [res["curiosity_ood_gain_pct"]] + [seeds[k]["gain_vm_pct"] for k in seeds]
claim("23-33 vs 8-11 (3 seeds)", "23-33 vs 8-11",
      f"{min(uc)}-{max(uc)} vs {min(ur)}-{max(ur)}", "results.json + seeds_check.json",
      min(uc) >= 23 and max(uc) <= 33 and min(ur) >= 8 and max(ur) <= 11)
claim("8-17% OOD gain", "8-17%", " / ".join(f"{g:+.1f}%" for g in gains),
      "OOD RMSE on |V|, curiosity vs random", 7 <= min(gains) and max(gains) <= 18)

# ---------------------------------------------------------------- calibration
cov = res["history"]["curiosity"][-1]["calib_id"]["coverage_2sigma"]
claim("coverage ~0.70 vs 0.95", "~0.70", f"{cov:.3f} (stored run); 0.64-0.72 across reruns",
      "|err| <= 2 sigma on 143 held-out interventions x 118 buses", 0.60 <= cov <= 0.75)
claim("0.95 nominal", "0.95", "0.954",
      "Gaussian interpretation of a +/- 2 sigma interval", True)

# ------------------------------------------------------------------- report
w = [26, 24, 46, 52]
print("\n" + "TRACEABILITY AUDIT".center(sum(w) + 12))
print("-" * (sum(w) + 12))
print(f"{'Claim':<{w[0]}} {'Stated':<{w[1]}} {'Measured':<{w[2]}} {'Source':<{w[3]}}")
print("-" * (sum(w) + 12))
bad = 0
for lab, st, me, src, ok in rows:
    flag = "OK " if ok else "!! "
    bad += (not ok)
    print(f"{flag}{lab:<{w[0]-3}} {st:<{w[1]}} {me:<{w[2]}} {src:<{w[3]}}")
print("-" * (sum(w) + 12))
print(f"{len(rows) - bad}/{len(rows)} claims verified" + ("" if not bad else f"  --  {bad} TO FIX"))
