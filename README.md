# LAPLACE — a toy prototype
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081404.svg)](https://doi.org/10.5281/zenodo.22081404)

**Feasibility prototype of a closed discovery loop on a 118-bus network**

A feasibility demonstration of a full scientific-discovery loop:

```
world model → uncertainty → curiosity (under a safety constraint)
            → experiment → revision → symbolic law → verification
```

The system studied is a **118-bus** transmission network simulated with `pandapower`.
The decisive point: the equations governing that simulator — the power flow, the
admittance matrix `Ybus` — are **known by construction**. Any law the agent claims to
have discovered can therefore be **refuted**, not merely admired.



## Results

| Measurement | Result |
|---|---|
| Ground-truth verification | equations vs simulator: `2.8e-10` p.u. |
| Curiosity vs random collection (out of distribution, equal budget) | lower error on \|V\| on **all three seeds**: 16.5 / 15.6 / 7.8 %. On angles the gain **does not replicate** (+22 / +8 / −1 %). |
| Interventions outside the safety envelope (74 experiments) | curiosity **33** · random 8 · curiosity + safety **6** (seed 0). The asymmetry **replicates**: 23–33 against 8–11 across three seeds. |
| Cost of caution | OOD RMSE 0.00297 vs 0.00279 — near zero *in this regime* |
| Structure recovered without giving the regression the true adjacency | **49/49** true electrical neighbours across the 6 most connected buses; 7 to 93 false positives per bus (median 21) |
| Predictive power of the recovered law | R² > 0.9999 on **127 interventions held out** from fitting |
| Fidelity of the admittance coefficients | R² = **0.91** against the true Ybus over the 98 coefficients (conductance and susceptance of 49 edges); median relative error 2 × 10⁻⁶ |

The most instructive result was not planned: **the curious agent walks straight into
danger**, because that is where it is ignorant. Of its 30 highest-scoring interventions,
roughly half (15 to 18 depending on the run) leave the voltage envelope, against 4 out
of 30 drawn at random.

> Every figure in this table is recomputed from the raw artefacts by
> `python audit_claims.py` — 13/13 verified.

## Contents

```
laplace/env.py           interventional environment (pandapower, case118)
laplace/world_model.py   message-passing GNN + deep ensemble (uncertainty)
laplace/curiosity.py     acquisition, safety screen, active-learning loop
laplace/distill.py       symbolic distillation (STLSQ) + scoring against ground truth
laplace/plots.py         the figures (no torch dependency)
build_pool.py            pre-computes the experiment pool (~10 min)
fix_ood.py               (re)generates the out-of-distribution test set
run_experiment.py        full experiment → figures/ + results.json (~19 min)
redo_distill.py          recomputes the distillation with the three R² reported separately
audit_claims.py          recomputes every reported figure from raw data (13/13 verified)
check_seeds.py           robustness: further seeds, and the excitation test
make_figures.py          redraws the figures from results.json, without re-running
make_notebook.py         regenerates the notebook
LAPLACE_prototype.ipynb  commented notebook (run it to populate the outputs)
results.json             the stored run: every raw measurement quoted above
```

## Install and run

```bash
pip install -r requirements.txt
python build_pool.py        # experiment pool (~10 min)
python fix_ood.py           # out-of-distribution test set (~2 min)
python run_experiment.py    # full loop + figures (~19 min)
jupyter notebook LAPLACE_prototype.ipynb
```

About thirty minutes in total on an ordinary CPU. No GPU required.
`cache_pool.pkl` (149 MB) is not tracked in git — `build_pool.py` regenerates it.

Note that `run_experiment.py` overwrites `results.json`, which is the stored run every
figure quoted above comes from; copy it aside first if you want to keep it. To redraw the
figures without re-running anything, use `python make_figures.py`. One figure ships with
French axis labels (`figures/fig2_calibration_fr.png`): it is a per-prediction scatter
that is not stored in `results.json`, so it can only be redrawn once `cache_pool.pkl`
exists, which `make_figures.py` then does.

## The four components

**1. Interventional environment.** An intervention = per-zone load scaling factors plus
taking a line out of service. One can *act*, not only observe.

**2. World model and uncertainty.** A GNN learns (injections, topology) → (|V|, angle).
A deep ensemble supplies epistemic uncertainty: where its members disagree, the model
does not know.

**3. Curiosity under constraint.** The agent scores each candidate intervention by
ensemble disagreement, then discards those the model judges to fall outside the voltage
envelope. The constraint is hard: a candidate it rejects cannot be redeemed by a high
information gain.

**4. From weights to law.** Sparse regression (STLSQ) over a library spanning **every**
bus in the network. Note the distinction: the world model *does* receive the adjacency of
the current topology — it is a graph-structured model. The regression does not: it must
select the electrical neighbours itself, and their coefficients.

*Three distinct R², never to be conflated*: the in-sample fit (not quoted as a result),
the out-of-sample predictive power (> 0.9999), and the fidelity of the coefficients
against Ybus (0.91).

## What the prototype does not do yet

- **False positives.** The regression finds every true neighbour and also keeps spurious
  terms (7 to 93 depending on the bus, median 21). The sparsity threshold is not the
  right instrument.
- **Insufficient excitation on a few edges.** The median edge is recovered to six
  significant figures, but a handful of poorly excited edges bring the coefficient R²
  down to 0.91.
- **A stated non-identifiability.** The diagonal term `B[i,i]` cannot be recovered from
  the active-power equation alone (its candidate term is `V²·sin 0 = 0`); it appears only
  in the reactive-power equation. Saying precisely what *cannot* be identified is a
  result, not a failure.
- **Curiosity serves the world model, not the law.** Coupling experiment design to
  symbolic discovery is exactly the open problem this prototype was built to expose.
- **The world model plateaus.** A five-layer GNN cannot represent a global operation such
  as a power flow. The architecture needs rethinking.
- One network, one physical domain, **three seeds**. The safety asymmetry replicates on
  all three; the accuracy gain is positive on all three but varies by a factor of two,
  and on angles it does not replicate at all. `check_seeds.py` reproduces that check.

## Reproducibility

Results are deterministic within a fixed environment, but small quantities shift with the
BLAS thread configuration — the count of unsafe interventions among the top 30, for
instance, moves between 15 and 18. `check_seeds.py` sets `torch.set_num_threads(1)`; do
the same before comparing runs. If a figure you obtain differs slightly from the table
above, this is the first thing to check.

## Citing this prototype

This archive is citable. If you use it, cite the concept DOI, which always resolves to the
latest version:

> Araar, S. (2026). *LAPLACE prototype: a closed discovery loop on a networked physical
> system* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.22081404

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). The `case118` network data ships with `pandapower` and is
subject to that project's own terms.
