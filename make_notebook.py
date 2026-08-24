import json

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": t}

cells = [
md("""# LAPLACE — a toy prototype

**An artificial scientist for networked physical systems.**

This notebook walks, at small scale but end to end, through the loop the LAPLACE project proposes:

> world model -> uncertainty -> curiosity (under a safety constraint) -> experiment -> revision -> **symbolic law** -> **verification against ground truth**

The system studied is a 118-bus transmission network simulated with `pandapower`.
The decisive point: the equations governing that simulator (the power flow, the
admittance matrix Ybus) are **known by construction**. Any law the agent claims to have
discovered can therefore be **refuted**, not merely admired.

*This notebook is a feasibility prototype, not a research result. The limitations are
listed honestly at the end.*"""),

code("""import pickle, json
import numpy as np
import matplotlib.pyplot as plt

from laplace.env import GridEnvironment
from laplace.world_model import Ensemble
from laplace.curiosity import acquisition_scores, safety_screen, active_loop
from laplace.distill import check_ground_truth, discover_law, score_against_truth, holdout_r2

D = pickle.load(open("cache_pool.pkl", "rb"))
pool, test_id, test_ood, Ybus = D["pool"], D["test_id"], D["test_ood"], D["Ybus_true"]
print(f"experiment pool: {len(pool)}   test: {len(test_id)} (ID) / {len(test_ood)} (OOD)")"""),

md("""## 1. The interventional environment

Unlike a frozen dataset, this object lets one **act**: choose an intervention (per-zone
load factors, a line taken out of service), perform it, and observe the electrical state
of every bus.

That is the difference between watching a system and questioning it."""),

code("""env = GridEnvironment(seed=0)

action = {"scale": np.full(env.n_zones, 1.3), "line_out": env.candidate_lines[0]}
obs = env.run(action)

print(f"intervention: loads x1.3 everywhere, line {action['line_out']} opened")
print(f"observed |V|: min {obs['y'][:,0].min():.3f}  max {obs['y'][:,0].max():.3f} p.u.")
print(f"inside the safety envelope [0.92, 1.06]: {env.is_safe(obs['y'][:,0])}")"""),

md("""## 2. Check the ground truth BEFORE relying on it

Comparing a "discovered" law against a false ground truth would be meaningless. So the
first step is to verify that the power-flow equations do describe the simulator, using
its admittance matrix:

$$P_i = \\sum_j |V_i||V_j| \\left( G_{ij}\\cos(\\theta_i-\\theta_j) + B_{ij}\\sin(\\theta_i-\\theta_j) \\right)$$"""),

code("""err = check_ground_truth(pool, Ybus, n_check=20)
print(f"maximum discrepancy equations / simulator: {err:.2e} p.u.   (machine precision)")
print("the ground truth is therefore usable: a discovered law can be refuted.")"""),

md("""## 3. The world model and its uncertainty

A message-passing GNN learns the map (injections, topology) -> (|V|, angle). A **deep
ensemble** of several models supplies the epistemic uncertainty: where the members
diverge, the model does not know.

That signal is what will drive curiosity."""),

code("""ens = Ensemble(k=4, hidden=56, n_layers=5, seed=0).fit(pool[:60], epochs=500)

print("error      :", {k: round(v, 5) for k, v in ens.rmse(test_id).items()})
print("calibration:", {k: round(v, 3) for k, v in ens.calibration(test_id).items()})

mean, std = ens.predict(test_id)
y = np.stack([s["y"] for s in test_id])
plt.figure(figsize=(5, 4))
plt.scatter(std[..., 0].ravel(), np.abs(mean[..., 0] - y[..., 0]).ravel(),
            s=8, alpha=.25, color="#2a78d6", edgecolors="none")
plt.xlabel("stated uncertainty"); plt.ylabel("observed error")
plt.title("Does the model know what it does not know?"); plt.grid(color="#e6e5e1"); plt.show()"""),

md("""## 4. Curiosity, and what it costs

The agent scores each candidate intervention by ensemble disagreement (expected
information gain), then screens out those the model judges unsafe.

The important observation below: **the most informative experiments are often the least
safe**. That is the tension WP2 of the project sets out to study."""),

code("""cands = pool[200:500]
score = acquisition_scores(ens, cands)
safe_pred = safety_screen(ens, cands)
safe_true = np.array([c["safe"] for c in cands])

top = np.argsort(-score)[:30]
print(f"among the 30 most informative interventions: "
      f"{(~safe_true[top]).sum()}/30 in fact leave the envelope")
print(f"among 30 interventions taken at random:      "
      f"{(~safe_true[:30]).sum()}/30")
print(f"\\nthe safety screen (model-based) accepts "
      f"{safe_pred.sum()}/{len(cands)}")"""),

md("""## 5. Results of the full loop

The three strategies (`random`, `curiosity`, `curiosity + safety`) start from the same
10 initial experiments and share the same budget (74 experiments).

Running `python run_experiment.py` reproduces these figures in ~19 min."""),

code("""res = json.load(open("results.json"))
for st, h in res["history"].items():
    print(f"{st:16s} final OOD RMSE|V| = {h[-1]['rmse_ood']['vm']:.5f}   "
          f"angle = {h[-1]['rmse_ood']['va']:.4f}   "
          f"interventions outside the envelope = {h[-1]['n_unsafe']}")
print(f"\\nout-of-distribution gain from curiosity: "
      f"{res['curiosity_ood_gain_pct']:.1f} % lower error at equal budget")"""),

code("""from IPython.display import Image, display
for f in ["fig1_efficiency", "fig3_frontier", "fig4_law"]:
    display(Image(f"figures/{f}.png"))"""),

md("""## 6. From weights to law

We do not ask the neural network to report what it knows. We ask the **collected data**
which algebraic relation governs it, using a library of candidate terms spanning **every**
bus in the network and a sparse regression (STLSQ, the SINDy optimiser).

**An important distinction.** The world model *does* receive the adjacency of the current
topology — it is a graph-structured model. The regression **does not**: it must discover
for itself which buses enter the equation — that is, the edge set — and with which
coefficients.

**Three R² must be kept carefully apart**: the in-sample fit (never quoted as a result),
the out-of-sample predictive power, and the fidelity of the recovered coefficients
against the true admittance matrix. These are three different claims, and conflating
them is indefensible in front of a reviewer."""),

code("""base = [s for s in pool if s["line_out"] is None]
degree = (np.abs(Ybus) > 0.05).sum(axis=1)
bus = int(np.argsort(-degree)[0])

rng = np.random.default_rng(0); idx = rng.permutation(len(base)); cut = int(.8 * len(base))
fit, held = [base[i] for i in idx[:cut]], [base[i] for i in idx[cut:]]

r = discover_law(fit, bus, threshold=0.05)
sc = score_against_truth(r, Ybus, bus)
print(f"bus {bus}: {sc['recovered']}/{sc['true_neighbours']} electrical neighbours recovered, "
      f"{sc['false_positives']} false positives")
print(f"R2 on {len(held)} HELD-OUT interventions: {holdout_r2(r, held, bus):.6f}")
print(f"(the in-sample fit R2, {sc['r2_in_sample_fit']:.4f}, is not a result:")
print(" it only measures reconstruction of the fitting data)")

sup = [j for j in np.where((np.abs(Ybus[bus].real) > .05) |
                           (np.abs(Ybus[bus].imag) > .05))[0] if j != bus]
print("\\n  j   B_true    B_recovered")
for j in sup[:8]:
    print(f"{j:4d} {Ybus[bus,j].imag:9.3f} {r['B_hat'][j]:12.3f}")"""),

md("""## 7. What this prototype shows — and what it does not

**What works.**

- The full loop runs end to end on a 118-bus network.
- Curiosity beats random collection out of distribution (~16 % lower error at equal
  budget on this seed) — a modest margin, and on one seed. Across three seeds the gain
  is 8–17 % on voltage magnitude, and it does not replicate on angles.
- The curious agent walks straight into danger: that is where it is ignorant. The safety
  constraint cuts out-of-envelope interventions roughly fivefold for a near-zero accuracy
  cost **in this regime** — an encouraging but local result.
- The network structure is recovered **without the true adjacency being given to the
  regression**: across the six most connected buses, all 49 true electrical neighbours
  are identified. The recovered law predicts active-power injection with R² > 0.9999 on
  127 held-out interventions, and the median coefficient is recovered to six significant
  figures (R² of the coefficients against the true Ybus: 0.91 over 98 coefficients — the
  conductance and susceptance of those 49 edges — with a few poorly excited coefficients
  explaining the gap).
- A **non-identifiability** shows up and can be read off the result: the diagonal term
  $B_{ii}$ cannot be recovered from the active-power equation alone (its candidate term
  is $V_i^2\\sin 0 = 0$); it appears only in the reactive-power equation. This is exactly
  the kind of result — stating precisely what *cannot* be identified — that the project
  claims as a contribution.

**What does not work yet — and that is the research programme.**

- The sparse regression produces many **false positives**: it finds the true neighbours
  but also keeps spurious terms (7 to 93 depending on the bus, median 21). The sparsity
  threshold is not the right instrument; a principled selection criterion is needed.
- **Excitation is insufficient on a few edges**: the median coefficient is exact to six
  figures, but a handful of poorly probed ones bring the coefficient R² down to 0.91.
  **This argues precisely for coupling distillation to experiment design** — what LAPLACE
  proposes, and what this prototype does not yet do: here curiosity serves the world
  model, not the law.
- The world model plateaus: a five-layer GNN cannot represent a global operation such as
  a power flow. The architecture needs rethinking.
- One physical domain, one network, three seeds. Nothing here is a scientific result: it
  is a feasibility proof of the loop."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("LAPLACE_prototype.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("notebook written")
