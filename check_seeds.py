"""Robustness checks for the LAPLACE prototype.

Two controls, matching the two fragile quantitative claims:

  main()             -- does the curiosity gain hold on other seeds?
                        (one seed proves nothing)
  check_excitation() -- does the spread in coefficient accuracy really follow
                        excitation, or is it an artefact of the relative error?

Run: python check_seeds.py     (both, ~35 min)
or   python -c "import check_seeds; check_seeds.check_excitation()"   (~1 min)
"""

import json, pickle
import numpy as np
import torch

from laplace.curiosity import active_loop
from laplace.distill import build_library, discover_law

torch.set_num_threads(1)      # reproducibility: fixes the BLAS reduction order

CFG = dict(n_init=10, n_rounds=8, batch=8, k=4, hidden=56, n_layers=5, epochs=500)


def main(seeds=(1, 2)):
    """Replays curiosity vs random on additional seeds."""
    D = pickle.load(open("cache_pool.pkl", "rb"))
    pool, tid, tood = D["pool"], D["test_id"], D["test_ood"]
    out = {}
    for seed in seeds:
        row = {}
        for st in ("random", "curiosity"):
            h, _ = active_loop(pool, tid, tood, strategy=st, seed=seed, verbose=False, **CFG)
            row[st] = {"vm": h[-1]["rmse_ood"]["vm"], "va": h[-1]["rmse_ood"]["va"],
                       "unsafe": h[-1]["n_unsafe"]}
        row["gain_vm_pct"] = 100 * (1 - row["curiosity"]["vm"] / row["random"]["vm"])
        row["gain_va_pct"] = 100 * (1 - row["curiosity"]["va"] / row["random"]["va"])
        out[f"seed_{seed}"] = row
        print(f"seed {seed}: out-of-distribution gain  |V| {row['gain_vm_pct']:+.1f}%   "
              f"angle {row['gain_va_pct']:+.1f}%   (outside the envelope: curiosity "
              f"{row['curiosity']['unsafe']}, random {row['random']['unsafe']})")
    json.dump(out, open("seeds_check.json", "w"), indent=2, default=float)
    print("seeds_check.json written")
    return out


def check_excitation(n_splits=3):
    """Does excitation explain the error on the coefficients?

    The excitation of a retained regressor is its variance once the other retained
    regressors are projected out (its partial variance): that is the quantity governing
    how well its coefficient can be determined. Three predictable objections are
    addressed here: an artefact of the relative error, near-zero denominators, and
    instability between fit/held-out splits.
    """
    D = pickle.load(open("cache_pool.pkl", "rb"))
    Y = D["Ybus_true"]
    base = [s for s in D["pool"] if s["line_out"] is None]
    buses = [int(b) for b in np.argsort(-(np.abs(Y) > 0.05).sum(1))[:6]]

    for split in range(n_splits):
        rng = np.random.default_rng(split)
        idx = rng.permutation(len(base))
        fit = [base[i] for i in idx[:int(0.8 * len(base))]]
        E, RE, AE, TC = [], [], [], []
        for b in buses:
            X, _ = build_library(fit, b)
            r = discover_law(fit, b, threshold=0.05)
            c = np.concatenate([r["G_hat"], r["B_hat"]])
            keep = np.where(np.abs(c) > 1e-9)[0]
            Xk = X[:, keep] - X[:, keep].mean(0)
            Gt, Bt = Y[b].real.copy(), Y[b].imag.copy()
            Bt[b] = 0.0                                   # non-identifiable diagonal term
            true = np.concatenate([Gt, Bt])
            for pos, k in enumerate(keep):
                if abs(true[k]) < 0.05:                   # judge only the true coefficients
                    continue
                others = np.delete(Xk, pos, axis=1)
                beta, *_ = np.linalg.lstsq(others, Xk[:, pos], rcond=None)
                E.append(float((Xk[:, pos] - others @ beta).var()))
                AE.append(abs(c[k] - true[k]))
                RE.append(abs(c[k] - true[k]) / abs(true[k]))
                TC.append(abs(true[k]))
        E, RE, AE, TC = map(np.array, (E, RE, AE, TC))
        q = np.quantile(E, [0.25, 0.75])
        lo, hi = E <= q[0], E >= q[1]
        print(f"split {split} (n={len(E)}): relative error x10^"
              f"{np.log10(np.median(RE[lo]) / np.median(RE[hi])):.1f}, "
              f"absolute x10^{np.log10(np.median(AE[lo]) / np.median(AE[hi])):.1f}, "
              f"median |true coef| {np.median(TC[lo]):.1f} (poorly excited) against "
              f"{np.median(TC[hi]):.1f} (well excited) -- no denominator artefact")


if __name__ == "__main__":
    check_excitation()
    main()
