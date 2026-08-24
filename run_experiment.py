"""
LAPLACE prototype -- the full experiment.

Loop: world model -> uncertainty -> curiosity (under a safety constraint)
      -> experiment -> revision -> symbolic distillation -> verification.

Produces the figures and results.json.

Warning: running this overwrites results.json. If you want to keep the stored run,
copy it aside first.
"""

import json, pickle, time
import numpy as np

from laplace.curiosity import active_loop
from laplace.distill import discover_law, score_against_truth, check_ground_truth
from laplace.plots import fig_efficiency, fig_calibration, fig_frontier, fig_law

CFG = dict(n_init=10, n_rounds=8, batch=8, k=4, hidden=56, n_layers=5, epochs=500)

def main():
    t0 = time.time()
    with open("cache_pool.pkl", "rb") as f:
        D = pickle.load(f)
    pool, test_id, test_ood, Ybus = D["pool"], D["test_id"], D["test_ood"], D["Ybus_true"]
    print(f"pool={len(pool)}  test_id={len(test_id)}  test_ood={len(test_ood)}\n")

    # ---------------------------------------------------------------- 0. verification
    gt_err = check_ground_truth(pool, Ybus)
    print(f"[0] Ground-truth check: equations vs simulator = {gt_err:.2e} p.u.\n")

    # --------------------------------------------------- 1-3. active-learning loop
    hists, models = {}, {}
    for st in ("random", "curiosity", "curiosity_safe"):
        print(f"[1] Active loop -- strategy: {st}")
        h, ens = active_loop(pool, test_id, test_ood, strategy=st, seed=0, **CFG)
        hists[st], models[st] = h, ens
        print()

    fig_efficiency(hists, "figures/fig1_efficiency.png")
    mean, std = models["curiosity"].predict(test_id)
    y_id = np.stack([s["y"] for s in test_id])
    corr = fig_calibration(std[..., 0].ravel(),
                           np.abs(mean[..., 0] - y_id[..., 0]).ravel(),
                           "figures/fig2_calibration.png")
    fig_frontier(hists, "figures/fig3_frontier.png")

    # ---------------------------------------------------------- 4. symbolic distillation
    print("[4] Symbolic distillation (sparse regression on the collected data)")
    base = [s for s in pool if s["line_out"] is None]
    print(f"    {len(base)} experiments on the nominal topology")
    scores, pairs = [], []
    degree = (np.abs(Ybus) > 0.05).sum(axis=1)
    buses = list(np.argsort(-degree)[:6])                 # most connected buses
    for b in buses:
        res = discover_law(base, int(b), threshold=0.05)
        sc = score_against_truth(res, Ybus, int(b))
        scores.append(sc)
        print(f"    bus {b:3d}: {sc['recovered']}/{sc['true_neighbours']} neighbours recovered, "
              f"{sc['false_positives']} false positives, coef error {sc['relative_coef_error']*100:.1f}%, "
              f"in-sample R2={sc['r2_in_sample_fit']:.4f}")
        sup = (np.abs(Ybus[b].real) > 0.05) | (np.abs(Ybus[b].imag) > 0.05)
        for j in np.where(sup)[0]:
            pairs.append((Ybus[b, j].real, res["G_hat"][j]))
            pairs.append((Ybus[b, j].imag, res["B_hat"][j]))
    fig_law(pairs, "figures/fig4_law.png")

    # -------------------------------------------------------------------- results
    out = {"config": CFG, "ground_truth_check_pu": gt_err,
           "history": hists, "uncertainty_error_correlation": corr,
           "distillation": scores,
           "curiosity_ood_gain_pct": 100 * (1 - hists["curiosity"][-1]["rmse_ood"]["vm"]
                                            / hists["random"][-1]["rmse_ood"]["vm"]),
           "runtime_s": time.time() - t0}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nOut-of-distribution gain from curiosity: {out['curiosity_ood_gain_pct']:.1f}% "
          f"lower error at equal budget")
    print(f"Done in {out['runtime_s']/60:.1f} min -- figures/ and results.json")


if __name__ == "__main__":
    main()
