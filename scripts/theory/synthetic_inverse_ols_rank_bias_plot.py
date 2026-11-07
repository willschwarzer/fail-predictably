"""Plot the synthetic rank-bias validation: theory vs. empirical.

Input: the JSON of synthetic_inverse_ols_rank_bias.py; the default file came from
  synthetic_inverse_ols_rank_bias.py --n-trials 1000000 --chunk 2000 \
      --out-json "$RESULTS_ROOT/synthetic_inverse_ols_rank_bias_1M.json"
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
FIGURES_ROOT = Path(os.environ.get("FIGURES_ROOT", REPO_ROOT / "artifacts" / "figures"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-json",
        type=str,
        default=str(RESULTS_ROOT / "synthetic_inverse_ols_rank_bias_1M.json"),
    )
    parser.add_argument("--out", type=str, default=str(FIGURES_ROOT / "synthetic_rank_bias_validation.pdf"))
    args = parser.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)
    cfg = data["config"]
    rows = data["rows"]
    Rs = np.array([r["R"] for r in rows])
    log_Rs = np.log(Rs)
    thy_real = np.array([r["theory_b_realized"] for r in rows])
    thy_quan = np.array([r["theory_b_quantile"] for r in rows])
    emp_real = np.array([r["empirical_b_realized"] for r in rows])
    emp_quan = np.array([r["empirical_b_quantile"] for r in rows])
    se_real = np.array([r["se_realized"] for r in rows])
    se_quan = np.array([r["se_quantile"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

    for ax, thy, emp, se, title in (
        (axes[0], thy_real, emp_real, se_real, "vs. realized deployment maximum"),
        (axes[1], thy_quan, emp_quan, se_quan, "vs. population one-in-$N$ quantile"),
    ):
        ax.plot(log_Rs, thy, "-", color="C0", linewidth=1.5, label="theory (limiting law)")
        ax.errorbar(
            log_Rs,
            emp,
            yerr=1.96 * se,
            fmt="o",
            color="C3",
            markersize=5,
            capsize=3,
            label=rf"empirical ($m={cfg['m']}$, ${cfg['n_trials']:,}$ trials)",
        )
        ax.set_xticks(log_Rs)
        ax.set_xticklabels([str(r) for r in Rs])
        ax.set_xlabel("$R = N/m$")
        ax.set_ylabel(r"mean error $/\, q'(y)$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        r"Synthetic validation of the inverse-OLS rank-bias coefficients on Exp(1) tails ($k=10$)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
