"""Regenerates the out-of-distribution test set.

The shift used here is on the TOPOLOGY (lines opened that were never seen during
training) and on the load regime (the upper part of the range). A more violent shift
(loads beyond 2.0) makes the power flow diverge: that is no longer a generalisation
test, it is an operating point that does not physically exist.
"""

import pickle
import numpy as np
from laplace.env import GridEnvironment
from build_pool import collect

env = GridEnvironment(seed=0)
ood_lines = env.candidate_lines[8:]
rng = np.random.default_rng(1234)

acts = []
for _ in range(200):
    s = rng.uniform(1.0, 1.9, size=env.n_zones)
    lo = int(rng.choice(ood_lines)) if rng.random() < 0.8 else None
    acts.append({"scale": s, "line_out": lo})

test_ood = collect(env, acts, "test_ood")
for r in test_ood:
    r["safe"] = env.is_safe(r["y"][:, 0])

with open("cache_pool.pkl", "rb") as f:
    D = pickle.load(f)
D["test_ood"] = test_ood
with open("cache_pool.pkl", "wb") as f:
    pickle.dump(D, f)

n_out = sum(r["line_out"] is not None for r in test_ood)
print(f"\ntest_ood = {len(test_ood)} samples, {n_out} of them on a never-seen topology")
