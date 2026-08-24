"""
From weights to law: symbolic distillation by sparse regression.

We do not ask the neural network to report what it knows. We ask the DATA the agent
collected which algebraic relation governs it, using a library of candidate terms and
a sparse regression (STLSQ, the SINDy optimiser).

The ground truth is known by construction: the active-power flow equation
    P_i = sum_j |V_i||V_j| ( G_ij cos(theta_i - theta_j) + B_ij sin(theta_i - theta_j) )
whose coefficients (G_ij, B_ij) are the entries of the simulator's admittance matrix
Ybus. A discovered law can therefore be CHECKED, not merely admired.
"""

import numpy as np
from pysindy.optimizers import STLSQ


def build_library(samples, bus):
    """Library of candidate terms for bus `bus`.

    For every other bus j in the network, two terms are offered:
        cos_ij = |V_i||V_j| cos(theta_i - theta_j)
        sin_ij = |V_i||V_j| sin(theta_i - theta_j)
    No topology prior is given to the regression: it must discover for itself WHICH
    buses enter the equation (hence the network structure) and with which coefficients.
    """
    V = np.stack([s["y"][:, 0] for s in samples])       # (N, n_bus)
    th = np.stack([s["y"][:, 1] for s in samples])
    n = V.shape[1]
    Vi, thi = V[:, bus:bus + 1], th[:, bus:bus + 1]
    cos = Vi * V * np.cos(thi - th)                     # (N, n_bus)
    sin = Vi * V * np.sin(thi - th)
    X = np.concatenate([cos, sin], axis=1)              # (N, 2*n_bus)
    names = [f"G[{bus},{j}]" for j in range(n)] + [f"B[{bus},{j}]" for j in range(n)]
    return X, names


def discover_law(samples, bus, threshold=0.05, alpha=1e-6):
    """Recover row `bus` of the admittance matrix from the observations alone."""
    X, names = build_library(samples, bus)
    y = np.stack([s["p_inj"] for s in samples])[:, bus]
    opt = STLSQ(threshold=threshold, alpha=alpha)
    opt.fit(X, y)
    coef = np.asarray(opt.coef_).ravel()
    n = X.shape[1] // 2
    return {"G_hat": coef[:n], "B_hat": coef[n:], "names": names,
            "r2": 1 - ((X @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()}


def holdout_r2(res, samples_test, bus):
    """R2 of the discovered law on interventions HELD OUT from fitting.

    Distinct from the in-sample fit R2 returned by discover_law. This is the figure to
    quote.
    """
    X, _ = build_library(samples_test, bus)
    y = np.stack([s["p_inj"] for s in samples_test])[:, bus]
    coef = np.concatenate([res["G_hat"], res["B_hat"]])
    return float(1 - ((X @ coef - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def score_against_truth(res, Ybus_true, bus, tol=0.05):
    """Compare the discovered law with the simulator's ground truth."""
    G_true, B_true = Ybus_true[bus].real, Ybus_true[bus].imag
    G_hat, B_hat = res["G_hat"], res["B_hat"]

    # B[i,i] is structurally non-identifiable from the active-power equation alone:
    # its candidate term is V_i^2 sin(0) = 0. It appears only in the reactive-power
    # equation. This is not a failure of the method but an identifiability limit --
    # exactly the kind of result the project sets out to characterise.
    B_true, B_hat = B_true.copy(), B_hat.copy()
    B_true[bus] = 0.0
    B_hat[bus] = 0.0

    true_sup = (np.abs(G_true) > tol) | (np.abs(B_true) > tol)      # true electrical neighbours
    hat_sup = (np.abs(G_hat) > tol) | (np.abs(B_hat) > tol)         # recovered neighbours

    tp = int((true_sup & hat_sup).sum())
    fp = int((~true_sup & hat_sup).sum())
    fn = int((true_sup & ~hat_sup).sum())
    both = true_sup & hat_sup
    err = 0.0
    if both.any():
        num = np.hypot(G_hat[both] - G_true[both], B_hat[both] - B_true[both])
        den = np.hypot(G_true[both], B_true[both])
        err = float(np.mean(num / den))
    return {"bus": bus, "true_neighbours": int(true_sup.sum()),
            "recovered": tp, "false_positives": fp, "missed": fn,
            "relative_coef_error": err, "r2_in_sample_fit": res["r2"],
            "true_coefs": [float(v) for v in np.concatenate([G_true[both], B_true[both]])],
            "recovered_coefs": [float(v) for v in np.concatenate([G_hat[both], B_hat[both]])]}


def check_ground_truth(samples, Ybus_true, n_check=5):
    """Prior control: do the assumed equations actually describe the simulator?

    Without this check, comparing a "discovered" law against a false ground truth would
    be meaningless. It is exactly the control the field most often omits.
    """
    # The supplied Ybus is that of the NOMINAL topology: only check on those runs.
    samples = [s for s in samples if s.get("line_out") is None][:n_check]
    V = np.stack([s["y"][:, 0] for s in samples])
    th = np.stack([s["y"][:, 1] for s in samples])
    P = np.stack([s["p_inj"] for s in samples])
    G, B = Ybus_true.real, Ybus_true.imag
    err = []
    for k in range(len(V)):
        c = np.cos(th[k][:, None] - th[k][None, :])
        s = np.sin(th[k][:, None] - th[k][None, :])
        VV = V[k][:, None] * V[k][None, :]
        P_eq = (VV * (G * c + B * s)).sum(axis=1)
        err.append(np.abs(P_eq - P[k]).max())
    return float(np.mean(err))
