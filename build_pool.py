"""
Pre-computes a pool of experiments and the test sets, then caches them.

Methodological note: the curious agent never "sees" an outcome before choosing it.
Pre-computing the pool therefore changes nothing scientifically -- this is pool-based
active learning, which is standard -- it only avoids re-invoking the simulator on
every trial.
"""

import pickle, time
import numpy as np
from laplace.env import GridEnvironment


def collect(env, actions, label):
    out, t0 = [], time.time()
    for i, a in enumerate(actions):
        r = env.run(a)
        if r is not None:
            r["safe"] = env.is_safe(r["y"][:, 0])
            out.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i+1}/{len(actions)} ({time.time()-t0:.0f}s)")
    return out


def main(n_pool=1000, n_test=150, seed=0):
    env = GridEnvironment(seed=seed)
    lines = env.candidate_lines
    train_lines, ood_lines = lines[:8], lines[8:]          # seen / never-seen topologies
    rng = np.random.default_rng(seed + 1)

    def make(n, scale_rng, line_choices, p_out=0.35):
        acts = []
        for _ in range(n):
            s = rng.uniform(*scale_rng, size=env.n_zones)
            lo = int(rng.choice(line_choices)) if (line_choices and rng.random() < p_out) else None
            acts.append({"scale": s, "line_out": lo})
        return acts

    print("Experiment pool (candidate interventions)...")
    pool = collect(env, make(n_pool, (0.6, 2.0), train_lines), "pool")
    print("In-distribution test set...")
    test_id = collect(env, make(n_test, (0.6, 2.0), train_lines), "test_id")
    print("Out-of-distribution test set (unseen topologies and regimes)...")
    test_ood = collect(env, make(n_test, (2.0, 2.4), ood_lines, p_out=0.6), "test_ood")

    with open("cache_pool.pkl", "wb") as f:
        pickle.dump({"pool": pool, "test_id": test_id, "test_ood": test_ood,
                     "Ybus_true": env.Ybus_true, "n_bus": env.n_bus}, f)
    print(f"\nOK -- pool={len(pool)}  test_id={len(test_id)}  test_ood={len(test_ood)}")
    print(f"     safe interventions in the pool: {sum(s['safe'] for s in pool)}/{len(pool)}")


if __name__ == "__main__":
    main()
