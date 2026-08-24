"""Recomputes the distillation with precise, separated measurements.

Three distinct quantities, often conflated, are reported separately here:
  1. R2 of the law on held-out interventions       -> predictive quality
  2. structure-recovery rate (neighbours, false positives)
  3. R2 of the recovered coefficients against the true Ybus  -> parameter fidelity
Which figure to quote depends on what is being claimed; mixing them is indefensible.

Warning: this updates results.json in place (the distillation section only).
"""

import pickle, json
import numpy as np
from laplace.distill import discover_law, score_against_truth, holdout_r2, check_ground_truth
from laplace.plots import fig_law

D = pickle.load(open("cache_pool.pkl", "rb"))
Y = D["Ybus_true"]
base = [s for s in D["pool"] if s["line_out"] is None]

rng = np.random.default_rng(0)
idx = rng.permutation(len(base))
n_fit = int(0.8 * len(base))
fit = [base[i] for i in idx[:n_fit]]
held = [base[i] for i in idx[n_fit:]]
print(f"fitting on {len(fit)} interventions, evaluating on {len(held)} held out")

degree = (np.abs(Y) > 0.05).sum(axis=1)
buses = [int(b) for b in np.argsort(-degree)[:6]]

scores, pairs = [], []
for b in buses:
    res = discover_law(fit, b, threshold=0.05)
    sc = score_against_truth(res, Y, b)
    sc["r2_held_out"] = holdout_r2(res, held, b)
    scores.append(sc)
    pairs += list(zip(sc["true_coefs"], sc["recovered_coefs"]))
    print(f"  bus {b:3d}: {sc['recovered']}/{sc['true_neighbours']} neighbours, "
          f"{sc['false_positives']} false positives, "
          f"held-out R2 {sc['r2_held_out']:.6f}")

T = np.array([p[0] for p in pairs])
H = np.array([p[1] for p in pairs])
r2_coef = float(1 - ((H - T) ** 2).sum() / ((T - T.mean()) ** 2).sum())
med_err = float(np.median(np.abs(H - T) / np.maximum(np.abs(T), 1e-9)))
r2_out = float(np.mean([s["r2_held_out"] for s in scores]))
fp = [s["false_positives"] for s in scores]

fig_law(pairs, "figures/fig4_law.png")

summary = {
    "fit_interventions": len(fit), "held_out_interventions": len(held),
    "r2_law_held_out_mean": r2_out,
    "r2_law_held_out_min": float(min(s["r2_held_out"] for s in scores)),
    "neighbours_recovered": sum(s["recovered"] for s in scores),
    "neighbours_true": sum(s["true_neighbours"] for s in scores),
    "false_positives_min_max_median": [int(min(fp)), int(max(fp)), float(np.median(fp))],
    "r2_coefficients_vs_Ybus": r2_coef,
    "median_relative_coef_error": med_err,
    "coefficients_scored": len(T),
    "ground_truth_check_pu": check_ground_truth(D["pool"], Y, 20),
}
print("\n" + json.dumps(summary, indent=2))

r = json.load(open("results.json"))
r["distillation"] = scores
r["distillation_summary"] = summary
json.dump(r, open("results.json", "w"), indent=2, default=float)
print("\nresults.json and figures/fig4_law.png updated")
