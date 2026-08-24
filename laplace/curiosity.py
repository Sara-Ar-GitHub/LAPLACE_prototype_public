"""
The curious experimenter: which experiment next, and at what price?

Three strategies are compared:
  - random          : picks at random (what passive data collection does)
  - curiosity       : picks where the ensemble disagrees most (information gain)
  - curiosity_safe  : the same, but refuses interventions the model judges unsafe

The difference between the last two measures the information given up in order to
stay inside the safety envelope: the "information-safety frontier".
"""

import numpy as np
from .world_model import Ensemble


def acquisition_scores(ens, candidates):
    """Curiosity score = mean ensemble disagreement (epistemic uncertainty)."""
    _, std = ens.predict(candidates)
    # normalise each quantity by its own scale so that the angle does not dominate
    s = std / (std.reshape(-1, std.shape[-1]).mean(0) + 1e-12)
    return s.mean(axis=(1, 2))


def safety_screen(ens, candidates, vm_min=0.92, vm_max=1.06, margin=1.0):
    """Safety screen based on the MODEL, not on the truth (unknown before the trial).

    A candidate is rejected if the pessimistic bound on the predicted voltage
    (mean -/+ margin standard deviations) falls outside the envelope.
    """
    mean, std = ens.predict(candidates)
    lo = (mean[..., 0] - margin * std[..., 0]).min(axis=1)
    hi = (mean[..., 0] + margin * std[..., 0]).max(axis=1)
    return (lo >= vm_min) & (hi <= vm_max)


def active_loop(pool, test_id, test_ood, strategy="curiosity", n_init=30, n_rounds=8,
                batch=10, k=5, hidden=64, n_layers=6, epochs=600, seed=0, verbose=True):
    """Active-learning loop: observe -> doubt -> choose -> act -> revise."""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(pool))
    labeled = list(rng.choice(idx_all, n_init, replace=False))   # same start for all strategies
    history, n_unsafe = [], 0

    for r in range(n_rounds + 1):
        train = [pool[i] for i in labeled]
        ens = Ensemble(k=k, hidden=hidden, n_layers=n_layers, seed=seed).fit(train, epochs=epochs)

        rec = {"round": r, "n_exp": len(labeled), "strategy": strategy,
               "rmse_id": ens.rmse(test_id), "rmse_ood": ens.rmse(test_ood),
               "calib_id": ens.calibration(test_id), "n_unsafe": n_unsafe}
        history.append(rec)
        if verbose:
            print(f"  [{strategy}] round {r}  {len(labeled)} experiments  "
                  f"RMSE|V| ID={rec['rmse_id']['vm']:.5f} OOD={rec['rmse_ood']['vm']:.5f}  "
                  f"RMSEangle OOD={rec['rmse_ood']['va']:.4f}  "
                  f"corr(unc,err)={rec['calib_id']['corr_uncertainty_error']:.2f}  "
                  f"violations={n_unsafe}")
        if r == n_rounds:
            break

        # --- choose the next experiments ---
        rest = np.setdiff1d(idx_all, labeled)
        cands = [pool[i] for i in rest]
        if strategy == "random":
            pick = rng.choice(len(rest), batch, replace=False)
        else:
            score = acquisition_scores(ens, cands)
            if strategy == "curiosity_safe":
                ok = safety_screen(ens, cands)
                score = np.where(ok, score, -np.inf)         # hard constraint, not a penalty
                if np.all(~np.isfinite(score)):              # nothing safe left: stop screening
                    score = acquisition_scores(ens, cands)
            pick = np.argsort(-score)[:batch]

        chosen = rest[pick]
        n_unsafe += sum(not pool[i]["safe"] for i in chosen)   # truth observed after the fact
        labeled += list(chosen)

    return history, ens
