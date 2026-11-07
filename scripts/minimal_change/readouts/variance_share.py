"""How much of each held-out read's squared error is variance across splits, and how much is bias.

This gives the shares in the paragraph "Room for improvement": on the held-out task sets of our
minimal-change training at step 200, the share of the best straight tail's squared error that is
variance across the scoring splits, on each bank. It repeats strict_table.read() and keeps each task
set's signed per-split errors for the base model, the best straight tail (the certificate), the
trained model after the band clamp and projection (the spread-normalized read) and the trained model
as read. Each task set's squared error is bias^2 + variance over its splits; the shares sum both over
every seed's task sets, and the improvement factors are checked against strict_table_200.csv.

  python variance_share.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strict_table as st  # noqa: E402

STEP = 200
MAIN = "anchpen + band (main)"
TAILS = ("base", "best straight tail", "trained (spread-normalized)", "trained (as read)")


def read(run, step):
    """Per task set and tail: (squared error, bias^2, variance) over the scoring splits."""
    ev = st.RUNS / run / "snapshot_evals"
    base, k = st.pools(ev / "base_test_forecast_traces.npz")
    trained, _ = st.pools(ev / f"step{step:04d}_test_forecast_traces.npz")
    out = []
    for name, (vb, M, e, perms, eb, sb, _sref) in st.base_errors(base, k).items():
        vt = trained[name][0]
        cert, tgt, e2, L, sref, info = st.certify(vb, M, k)
        lo, hi = sref - st.MARGIN, sref + st.MARGIN
        s = st.involved_sigma(vt, M, k, e)
        vc = vt * (min(max(s, lo), hi) / s)
        vf, ok = st.project(vc, vb, M, k, e, lo, hi)
        ec, _ = st.split_errors(cert, perms, M, k=k)
        ef, _ = st.split_errors(vf, perms, M, k=k)
        et, _ = st.split_errors(vt, perms, M, k=k)
        out.append([(float((x ** 2).mean()), float(x.mean()) ** 2, float(x.var())) for x in (eb, ec, ef, et)])
    return np.array(out)  # task sets x tails x (squared error, bias^2, variance)


def main():
    ref = {r["dataset"]: r for r in csv.DictReader(open(st.READOUTS / "strict_table_200.csv"))
           if r["seed"] == "pooled" and r["arm"] == MAIN and r["step"] == str(STEP)}
    for dataset, label, name in st.ARMS:
        if label != MAIN:
            continue
        a = np.concatenate([read(name(s, STEP), STEP) for s in st.SEEDS])
        sums = a.sum(axis=0)  # tails x (squared error, bias^2, variance)
        factor, cert = sums[0, 0] / sums[2, 0], sums[0, 0] / sums[1, 0]
        assert abs(factor - float(ref[dataset]["fold"])) < 1e-6 * factor, (dataset, factor, ref[dataset]["fold"])
        assert abs(cert - float(ref[dataset]["cert_fold"])) < 1e-6 * cert, (dataset, cert, ref[dataset]["cert_fold"])
        print(f"\n{dataset}, step {STEP}: {a.shape[0]} held-out task sets over {len(st.SEEDS)} seeds "
              f"(improvement factors {factor:.3f} and {cert:.3f}, as in strict_table_200.csv)")
        print(f"  {'tail':28s} {'mean sq. error':>14s} {'bias^2':>7s} {'variance':>8s} {'factor':>7s}")
        for i, t in enumerate(TAILS):
            mse, b2, v = sums[i]
            print(f"  {t:28s} {mse / a.shape[0]:14.4f} {100 * b2 / mse:6.1f}% {100 * v / mse:7.1f}% "
                  f"{sums[0, 0] / mse:7.2f}")


if __name__ == "__main__":
    main()
