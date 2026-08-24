"""Redraws the figures from results.json, without re-running the experiment.

Useful because run_experiment.py overwrites results.json: if you only want the figures
back (after a restyle, a relabelling, or a fresh clone), this script redraws them from
the stored run and touches nothing else.

fig1, fig3 and fig4 come from results.json alone. fig2 is a scatter of per-prediction
uncertainty against per-prediction error, which is not stored in results.json; it is
redrawn only if cache_pool.pkl is present (that path needs torch).

Run: python make_figures.py
"""

import json, os, pickle
import numpy as np

from laplace.plots import fig_efficiency, fig_frontier, fig_law, fig_calibration

res = json.load(open("results.json"))
os.makedirs("figures", exist_ok=True)

fig_efficiency(res["history"], "figures/fig1_efficiency.png")
fig_frontier(res["history"], "figures/fig3_frontier.png")

pairs = []
for sc in res["distillation"]:
    pairs += list(zip(sc["true_coefs"], sc["recovered_coefs"]))
fig_law(pairs, "figures/fig4_law.png")
print(f"fig1, fig3, fig4 redrawn from results.json ({len(pairs)} coefficients in fig4)")

if os.path.exists("cache_pool.pkl"):
    from laplace.world_model import Ensemble
    D = pickle.load(open("cache_pool.pkl", "rb"))
    pool, test_id = D["pool"], D["test_id"]
    cfg = res["config"]
    ens = Ensemble(k=cfg["k"], hidden=cfg["hidden"], n_layers=cfg["n_layers"], seed=0)
    ens.fit(pool[:60], epochs=cfg["epochs"])
    mean, std = ens.predict(test_id)
    y = np.stack([s["y"] for s in test_id])
    corr = fig_calibration(std[..., 0].ravel(), np.abs(mean[..., 0] - y[..., 0]).ravel(),
                           "figures/fig2_calibration.png")
    print(f"fig2 redrawn (correlation {corr:.2f})")
else:
    print("cache_pool.pkl absent: fig2 not redrawn (run build_pool.py first)")
