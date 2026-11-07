"""Independent check of tail_opt.py's optimized tails, with the pre-existing functions only.

For each pool record in a tail_opt npz (run pools, WildChat bank pools, and grid pools redrawn with
cert_scale's population and seed and checked against cert_scale.csv's base MSE): rebuild the pool's base
values, scoring permutations and certificate from the loaders feasibility.py and cert_scale.py use;
take the optimized tail as stored (prompt order); check that no prompt is raised and that the
involved-set sigma lies in the certificate's band (cert_sq_strategy.involved_sigma; certify's
sigma_ref); score base, certificate and optimized tails with poolscale_readout.split_errors on the
pool's 200 scoring splits and on FRESH uniform partitions (own seed, none shared with tail_opt).
Prints per pool, then pooled folds per run (sum of base MSE over sum of tail MSE).

  python verify_tail_opt.py NPZ [--fresh N] [--only RUN_SUBSTRING]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cert_scale import MARGIN, certify, pools_from_bank, smoke  # noqa: E402
from fleet_table import READOUTS, RUNS, base_errors  # noqa: E402
from cert_sq_strategy import involved_sigma  # noqa: E402
from heldout_spread import pools  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402


def main():
    args = sys.argv[1:]
    npz = args[0]
    n_fresh = int(args[args.index("--fresh") + 1]) if "--fresh" in args else 10000
    only = args[args.index("--only") + 1] if "--only" in args else None
    z = np.load(npz, allow_pickle=True)
    meta = json.loads(str(z["metadata"]))
    run_cache, bank, pop, refs = {}, None, None, None
    pooled = defaultdict(lambda: np.zeros(6))
    bad = 0
    for det in meta["pool_details"]:
        ent = det["entry"]
        run, name, key, k, M = ent["run"], ent["pool"], ent["key"], int(ent["k"]), int(ent["M"])
        if only and only not in run:
            continue
        ref_note = ""
        if run == "wildchat_bank":
            if bank is None:
                bank = pools_from_bank(smoke())
            vb = bank[name]
            e = count_extrapolated_eval_points(M, vb.size - M, plotting="weibull")
            rng = np.random.default_rng(int(ent["legacy_seed"]))
            perms = np.stack([rng.permutation(vb.size) for _ in range(200)])
        elif ent.get("grid"):
            # cert_scale.main's population and cert_scale.run_pool's draw, rebuilt here rather than
            # taken from tail_opt_grid, then checked against cert_scale.csv's base MSE.
            if pop is None:
                if bank is None:
                    bank = pools_from_bank(smoke())
                pop = np.concatenate([x for x in bank.values() if x.size == 2300])
                refs = {(r["cell"], int(r["k"]), r["pool"]): r for r in csv.DictReader(open(READOUTS / "cert_scale.csv"))
                        if r["source"] == "random"}
            g = ent["grid"]
            N, pid = int(g["N"]), int(g["pool_id"])
            rng = np.random.default_rng([M, N, pid, 20261104])
            vb = pop[rng.choice(pop.size, size=M + N, replace=False)]
            e = count_extrapolated_eval_points(M, N, plotting="weibull")
            rng = np.random.default_rng(int(ent["legacy_seed"]))
            perms = np.stack([rng.permutation(vb.size) for _ in range(200)])
            ref = refs[(g["cell"], k, f"rand{pid:02d}")]
            dev = abs(float((split_errors(vb, perms, M, k=k)[0] ** 2).mean()) - float(ref["base_mse"]))
            ref_note = f" | cert_scale.csv base deviation {dev:.1e}"
            bad += dev > 1e-9
        else:
            if run not in run_cache:
                base, kk = pools(RUNS / run / "snapshot_evals" / "base_test_forecast_traces.npz")
                run_cache[run] = base_errors(base, kk)
            vb, M2, e, perms, eb, sb, _ = run_cache[run][name]
            assert M2 == M
        out, tgt, e2, L, sref, info = certify(vb, M, k)
        lo, hi = sref - MARGIN, sref + MARGIN
        v = z[f"{key}_optimized_values"]
        assert v.shape == vb.shape
        raise_max = float((v - vb).max())
        sig = involved_sigma(v, M, k, e)
        ok = raise_max <= 1e-9 and lo - 1e-9 <= sig <= hi + 1e-9 and np.all(np.isfinite(v))
        bad += not ok
        # Fresh partitions in chunks (the same draw order as one block, so earlier results are
        # unchanged), scored paired across the three tails; at U = 72,750 one block of 10,000
        # permutations would need 5.8 GB.
        frng = np.random.default_rng(424242 + int(ent["legacy_seed"]) % 100000)
        sq = [[], [], []]
        left = n_fresh
        while left > 0:
            c = min(250, left)
            fresh = np.stack([frng.permutation(vb.size) for _ in range(c)])
            for i, t in enumerate((vb, out, v)):
                sq[i].append(split_errors(t, fresh, M, k=k)[0] ** 2)
            left -= c
        sq = [np.concatenate(s) for s in sq]
        vals = []
        for t, s in zip((vb, out, v), sq):
            vals.append(float((split_errors(t, perms, M, k=k)[0] ** 2).mean()))
            vals.append(float(s.mean()))
        top = np.sort(sq[2])[::-1]
        share = float(top[: max(1, top.size // 100)].sum() / top.sum())
        pooled[(run, ent.get("step"), k)] += np.array(vals)
        print(f"{run}:{ent.get('step')} k{k} {name}: feasible {ok} (raise {raise_max:.1e}, sigma {sig:.4f} in [{lo:.4f}, {hi:.4f}]) | "
              f"legacy base {vals[0]:.3f} cert {vals[2]:.3f} opt {vals[4]:.3f} | fresh base {vals[1]:.3f} cert {vals[3]:.3f} opt {vals[5]:.3f}"
              f" | opt worst-1% share {share:.2f}{ref_note}", flush=True)
    print(f"\ninfeasible or non-finite optimized tails: {bad}")
    print("pooled folds over base: legacy (cert, optimized) | fresh (cert, optimized, optimized/cert)")
    for key, s in sorted(pooled.items(), key=lambda kv: str(kv[0])):
        print(f"  {key[0]:<26s} step {str(key[1]):>4s} k{key[2]:<3d} | legacy {s[0] / s[2]:6.2f} {s[0] / s[4]:6.2f} | "
              f"fresh {s[1] / s[3]:6.2f} {s[1] / s[5]:6.2f} {s[3] / s[5]:5.2f}")


if __name__ == "__main__":
    main()
