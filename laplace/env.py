"""
Interventional environment built on a validated engineering simulator (pandapower).

The central idea of LAPLACE: unlike a frozen dataset, this object lets one ACT on the
system (change the loads, open a line) and observe the outcome. The generating
structure is known by construction: the power-flow equations and the simulator's
admittance matrix Ybus serve as ground truth.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn


class GridEnvironment:
    """Interventional environment over a transmission network.

    An intervention = (per-zone load scaling factors, line taken out of service).
    An experiment   = apply the intervention, solve the power flow, and return the
                      electrical state of every bus.
    """

    def __init__(self, n_zones=8, n_candidate_lines=12, seed=0):
        self.rng = np.random.default_rng(seed)
        self.net0 = pn.case118()
        pp.runpp(self.net0, numba=False)

        self.n_bus = len(self.net0.bus)
        self.base_mva = self.net0.sn_mva

        # --- group the loads into zones (one lever per zone, not per load) ---
        load_buses = self.net0.load.bus.values
        self.n_zones = n_zones
        self.load_zone = np.array([b % n_zones for b in load_buses])
        self.p_load0 = self.net0.load.p_mw.values.copy()
        self.q_load0 = self.net0.load.q_mvar.values.copy()

        # --- base topology: normalised adjacency matrix for the GNN ---
        self.edges0 = self._edges(self.net0)

        # --- lines eligible for outage (pre-screened: the power flow still converges) ---
        self.candidate_lines = self._screen_lines(n_candidate_lines)

        # --- ground truth: admittance matrix of the nominal topology ---
        Ybus = self.net0._ppc["internal"]["Ybus"].toarray()
        lookup = self.net0._pd2ppc_lookups["bus"]
        idx = lookup[self.net0.bus.index.values]
        self.Ybus_true = Ybus[np.ix_(idx, idx)]

        self.n_experiments = 0  # counter: the real cost of an experiment

    # ------------------------------------------------------------------ topology
    def _edges(self, net):
        """Edge list (from_bus, to_bus) of the in-service lines and transformers."""
        e = []
        for df, a, b in [(net.line, "from_bus", "to_bus"), (net.trafo, "hv_bus", "lv_bus")]:
            if len(df) == 0:
                continue
            ok = df.in_service.values
            e += list(zip(df[a].values[ok], df[b].values[ok]))
        return np.array(e, dtype=int)

    def adjacency(self, line_out=None):
        """Normalised adjacency (A + I) D^-1 of the current topology."""
        net = self.net0
        edges = self.edges0 if line_out is None else self._edges_without(line_out)
        A = np.eye(self.n_bus)
        for u, v in edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
        return A / A.sum(axis=1, keepdims=True)

    def _edges_without(self, line_out):
        mask = np.ones(len(self.net0.line), dtype=bool)
        mask[line_out] = False
        e = list(zip(self.net0.line.from_bus.values[mask], self.net0.line.to_bus.values[mask]))
        if len(self.net0.trafo):
            e += list(zip(self.net0.trafo.hv_bus.values, self.net0.trafo.lv_bus.values))
        return np.array(e, dtype=int)

    def _screen_lines(self, k):
        """Keep the lines whose outage still lets the power flow converge."""
        ok = []
        for li in self.rng.permutation(len(self.net0.line)):
            if len(ok) >= k:
                break
            net = pn.case118()
            net.line.at[li, "in_service"] = False
            try:
                pp.runpp(net, numba=False)
                if net.converged:
                    ok.append(int(li))
            except Exception:
                pass
        return sorted(ok)

    # --------------------------------------------------------------- interventions
    def sample_interventions(self, n, scale_range=(0.6, 2.0), p_outage=0.35):
        """Draw n candidate interventions (not yet performed: nothing is computed here)."""
        scales = self.rng.uniform(*scale_range, size=(n, self.n_zones))
        lines = []
        for _ in range(n):
            if self.rng.random() < p_outage:
                lines.append(int(self.rng.choice(self.candidate_lines)))
            else:
                lines.append(None)
        return [{"scale": scales[i], "line_out": lines[i]} for i in range(n)]

    def node_features(self, action):
        """Input to the world model: per-bus injections plus the bus type.

        These quantities are KNOWN before the experiment (they are setpoints), unlike
        the electrical state, which has to be predicted or measured. This rules out any
        leakage from the outcome.
        """
        p = np.zeros(self.n_bus)
        q = np.zeros(self.n_bus)
        s = action["scale"][self.load_zone]
        np.add.at(p, self.net0.load.bus.values, -self.p_load0 * s / self.base_mva)
        np.add.at(q, self.net0.load.bus.values, -self.q_load0 * s / self.base_mva)
        if len(self.net0.gen):
            np.add.at(p, self.net0.gen.bus.values, self.net0.gen.p_mw.values / self.base_mva)
        is_gen = np.zeros(self.n_bus)
        is_gen[self.net0.gen.bus.values] = 1.0
        is_slack = np.zeros(self.n_bus)
        is_slack[self.net0.ext_grid.bus.values] = 1.0
        return np.stack([p, q, is_gen, is_slack], axis=1)

    def run(self, action):
        """PERFORM the experiment: apply the intervention and solve the power flow.

        This is the only expensive operation in the chain, and therefore the resource
        the curious agent has to spend wisely.
        """
        net = pn.case118()
        net.load.p_mw = self.p_load0 * action["scale"][self.load_zone]
        net.load.q_mvar = self.q_load0 * action["scale"][self.load_zone]
        if action["line_out"] is not None:
            net.line.at[action["line_out"], "in_service"] = False
        try:
            pp.runpp(net, numba=False)
        except Exception:
            return None
        if not net.converged:
            return None
        vm = net.res_bus.vm_pu.values.copy()
        va = np.deg2rad(net.res_bus.va_degree.values.copy())
        # an open line can isolate a bus: pandapower "converges" but returns NaN
        if np.isnan(vm).any() or np.isnan(va).any():
            return None
        self.n_experiments += 1
        return {
            "x": self.node_features(action),
            "A": self.adjacency(action["line_out"]),
            "y": np.stack([vm, va], axis=1),          # observed state: |V| and angle
            "p_inj": -net.res_bus.p_mw.values / self.base_mva,
            "q_inj": -net.res_bus.q_mvar.values / self.base_mva,
            "line_out": action["line_out"],
        }

    # --------------------------------------------------------------------- safety
    VM_MIN, VM_MAX = 0.92, 1.06

    def is_safe(self, vm):
        """Safety envelope: all voltages within [0.92, 1.06] p.u.

        A real engineering constraint, not a tunable penalty term.
        """
        return bool(np.all((vm >= self.VM_MIN) & (vm <= self.VM_MAX)))
