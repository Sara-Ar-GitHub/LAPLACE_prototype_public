"""
World model: a neural simulator of the network, and its epistemic uncertainty.

The model learns the map (injections, topology) -> (electrical state), that is, what
the simulator does, without being given the equations.

Uncertainty comes from a deep ensemble: several models trained independently. Where
they agree, the model "knows"; where they diverge, it does not. That disagreement is
the signal that will drive curiosity.
"""

import numpy as np
import torch
import torch.nn as nn


class GraphSim(nn.Module):
    """Message-passing GNN: H <- tanh(H W_self + A H W_neigh).

    Deliberately simple and readable: neighbourhood aggregation is enough to capture
    the fact that a bus depends on its electrical neighbours.
    """

    def __init__(self, in_dim=4, hidden=48, out_dim=2, n_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.self_w = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.neigh_w = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)])
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, A):
        # x: (B, n, f)   A: (B, n, n)
        h = x
        for sw, nw in zip(self.self_w, self.neigh_w):
            h = torch.tanh(sw(h) + nw(torch.bmm(A, h)))
        return self.head(h)


class Ensemble:
    """Deep ensemble: K world models trained on different bootstrap resamples."""

    def __init__(self, k=5, hidden=48, n_layers=3, seed=0):
        self.k, self.hidden, self.n_layers, self.seed = k, hidden, n_layers, seed
        self.models, self.mu, self.sd = [], None, None

    # --------------------------------------------------------------------- data
    @staticmethod
    def pack(samples):
        x = torch.tensor(np.stack([s["x"] for s in samples]), dtype=torch.float32)
        A = torch.tensor(np.stack([s["A"] for s in samples]), dtype=torch.float32)
        y = torch.tensor(np.stack([s["y"] for s in samples]), dtype=torch.float32)
        return x, A, y

    def fit(self, samples, epochs=200, lr=5e-3, verbose=False):
        x, A, y = self.pack(samples)
        # standardise the targets (|V| ~ 1.0, angle in radians), otherwise the angle
        # dominates the loss
        self.mu = y.reshape(-1, y.shape[-1]).mean(0)
        self.sd = y.reshape(-1, y.shape[-1]).std(0) + 1e-8
        yn = (y - self.mu) / self.sd

        self.models = []
        g = np.random.default_rng(self.seed)
        for m in range(self.k):
            torch.manual_seed(self.seed * 100 + m)
            net = GraphSim(x.shape[-1], self.hidden, y.shape[-1], self.n_layers)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            # bootstrap: each member sees a different resample
            idx = torch.tensor(g.integers(0, len(samples), len(samples)))
            xb, Ab, yb = x[idx], A[idx], yn[idx]
            for ep in range(epochs):
                opt.zero_grad()
                loss = ((net(xb, Ab) - yb) ** 2).mean()
                loss.backward()
                opt.step()
            if verbose:
                print(f"    member {m}: final loss {loss.item():.5f}")
            self.models.append(net)
        return self

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, samples):
        """Return (mean, standard deviation) over the ensemble, in physical units."""
        x, A, _ = self.pack(samples)
        preds = torch.stack([net(x, A) for net in self.models])       # (K, B, n, 2)
        mean = preds.mean(0) * self.sd + self.mu
        std = preds.std(0) * self.sd
        return mean.numpy(), std.numpy()

    def rmse(self, samples):
        mean, _ = self.predict(samples)
        y = np.stack([s["y"] for s in samples])
        return {
            "vm": float(np.sqrt(((mean[..., 0] - y[..., 0]) ** 2).mean())),
            "va": float(np.sqrt(((mean[..., 1] - y[..., 1]) ** 2).mean())),
        }

    def calibration(self, samples):
        """Does the ensemble spread actually predict the error?

        Two measurements: the correlation between stated uncertainty and observed
        error, and the coverage of a +/- 2 standard-deviation interval.
        """
        mean, std = self.predict(samples)
        y = np.stack([s["y"] for s in samples])
        err = np.abs(mean - y)
        corr = float(np.corrcoef(std[..., 0].ravel(), err[..., 0].ravel())[0, 1])
        cover = float((err <= 2 * std + 1e-9).mean())
        return {"corr_uncertainty_error": corr, "coverage_2sigma": cover}
