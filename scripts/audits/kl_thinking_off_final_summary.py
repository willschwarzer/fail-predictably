"""Means over seeds of the KL files that kl_thinking_off_final.py writes, to six significant digits.

kl_thinking_off_final.py prints each label's mean KL per mode at the end of a run, to four significant digits.
Four are too few to check the paper's rounding: a mean of 10.0496 prints as 10.05, which would round to 10.1
rather than the paper's 10.0. For each CSV and label, this script prints the number of adapters; the mean,
minimum and maximum of each KL column (the trainer's own KL, then the three modes); the trained model's lowest
mean top-1 probability in each mode; the prompt and cut counts; and every other label's ratio of means to the
first label's (SFT / Ours for Figure 4's file). It needs only a CPU and the Python standard library.

  python scripts/audits/kl_thinking_off_final_summary.py CSV [CSV ...]
"""
import argparse
import csv
import statistics

KL_COLS = ["kl_on_trainer", "kl_on_right", "kl_off_right", "kl_off_left"]
TOP1_COLS = ["trained_top1_on_right", "trained_top1_off_right", "trained_top1_off_left"]
COUNT_COLS = ["n_prompts", "n_cut_on_right", "n_cut_off_right", "n_cut_off_left"]


def main():
    ap = argparse.ArgumentParser(description="Means over seeds of kl_thinking_off_final.py's CSVs.")
    ap.add_argument("csvs", nargs="+", metavar="CSV")
    a = ap.parse_args()
    for path in a.csvs:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        by = {}
        for r in rows:
            by.setdefault(r["label"], []).append(r)
        print(f"== {path}")
        means = {}
        for lab, rs in by.items():
            print(f"{lab}: {len(rs)} adapters")
            for c in KL_COLS:
                v = [float(r[c]) for r in rs if r[c] not in ("", None)]
                if v:
                    means[(lab, c)] = statistics.mean(v)
                    print(f"  {c:14s} mean {means[(lab, c)]:.6g}  min {min(v):.6g}  max {max(v):.6g}")
                else:
                    print(f"  {c:14s} empty")
            print("  lowest trained top-1: " + ", ".join(f"{c} {min(float(r[c]) for r in rs):.4f}" for c in TOP1_COLS))
            print("  counts: " + ", ".join(f"{c} {' '.join(sorted({r[c] for r in rs}))}" for c in COUNT_COLS))
        labs = list(by)
        for lab in labs[1:]:
            ratios = [f"{c} {means[(lab, c)] / means[(labs[0], c)]:.4g}" for c in KL_COLS
                      if (lab, c) in means and (labs[0], c) in means]
            print(f"{lab} / {labs[0]}, ratio of means: " + ", ".join(ratios))


if __name__ == "__main__":
    main()
