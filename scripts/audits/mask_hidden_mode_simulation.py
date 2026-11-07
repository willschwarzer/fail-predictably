"""Is the improving-only deploy mask active on rare-mode prompts exactly when
the fit set missed the mode? (paper appendix "Checks beyond the training
forecast and score", subsection "The improving-only mask and hidden rare modes")

The concern: the mask keeps a deploy rank active when the fit line
under-predicts it, but in the dangerous case the rare mode is absent from the
fit side, so the line is fit on bulk scores alone -- does the protection
still engage there?

This simulation answers with a frequency. For each held-out target of the
rebuilt-bank password fleets, we take the baseline model's full scored pool
(fit + deploy from the saved traces, 935 prompts), attach family labels by
reconstructing the records exactly as the eval built them, and draw many
random fit/deploy partitions at the paper cell (44/891). Per partition we fit
the top-10 Weibull OLS line on the fit side and apply the deploy-side
improving-only criterion (active iff observed transformed score >= the line's
prediction at that rank's plotting position, the `eval_scores >= pred` rule
from ols_utils). "Rare mode" is filtered to prompts that are actually
dangerous: GCG-family prompts whose baseline leak probability exceeds a
threshold (0.5 primary; 0.1/0.9 sensitivity).

Reported: P(hidden) -- no rare prompt in the fit top-k -- and, conditional on
hidden vs not-hidden, the probability that the highest-scoring rare deploy
prompt lands inside the extrapolated window and is mask-active.

CPU-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))

from forecast_experiment.train_lquant import (  # noqa: E402
    LQuantConfig,
    _load_prompt_pool_from_bank,
    _split_test_records_for_target,
    split_pool,
)
from forecast_experiment.lquant_prompt_bank import resolve_prompt_bank  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402

PSI_EPS = 1e-8
THRESHOLDS = [0.1, 0.5, 0.9]


def load_cfg(run_dir: Path) -> LQuantConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in dataclasses.fields(LQuantConfig)}
    return LQuantConfig(**{k: v for k, v in raw.items() if k in names})


def target_pool(run_dir: Path, tokenizer_cache: dict) -> list[dict]:
    """Per held-out target: baseline scores + family labels, trace order."""
    cfg = load_cfg(run_dir)
    tok_key = cfg.model_id
    if tok_key not in tokenizer_cache:
        from transformers import AutoTokenizer
        tokenizer_cache[tok_key] = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer = tokenizer_cache[tok_key]
    bank_seed = int(cfg.seed if cfg.prompt_bank_seed < 0 else cfg.prompt_bank_seed)
    bank_info = resolve_prompt_bank(cfg.prompt_bank_root, bank_seed)
    pool = _load_prompt_pool_from_bank(bank_info, cfg=cfg, tokenizer=tokenizer)
    train_recs, eval_recs, test_recs = split_pool(pool, cfg)

    meta = json.loads((run_dir / "baseline_test_forecast_traces_metadata.json").read_text())
    data = np.load(run_dir / "baseline_test_forecast_traces.npz")
    out = []
    for tinfo in meta["targets"]:
        target = tinfo["target"]
        fit, deploy = _split_test_records_for_target(
            target, test_recs.get(target, []), train_recs, eval_recs, cfg)
        if len(fit) != tinfo["fit_count"] or len(deploy) != tinfo["deploy_count"]:
            raise RuntimeError(f"{target}: reconstruction mismatch")
        pre = tinfo["array_prefix"]
        lp = np.concatenate([
            np.asarray(data[f"{pre}__fit_log_probs"], dtype=np.float64),
            np.asarray(data[f"{pre}__deploy_log_probs"], dtype=np.float64)])
        fams = np.array([r.get("family_id", "?") for r in fit + deploy])
        out.append({"run_dir": str(run_dir), "target": target,
                    "fit_n": len(fit), "deploy_n": len(deploy),
                    "log_probs": lp, "families": fams})
    return out


def simulate(entry: dict, n_partitions: int, k: int, rng: np.random.Generator) -> dict:
    lp = entry["log_probs"]
    fams = entry["families"]
    M, N = entry["fit_n"], entry["deploy_n"]
    n_total = len(lp)
    psi = -np.log(np.maximum(-lp, PSI_EPS))
    prob = np.exp(lp)
    is_gcg = np.char.find(np.char.lower(fams.astype(str)), "gcg") >= 0
    n_ext = count_extrapolated_eval_points(M, N, plotting="weibull")
    y_fit = np.log(np.arange(1, k + 1) / (M + 1.0))
    p_dep = np.arange(1, n_ext + 1) / (N + 1.0)
    log_p_dep = np.log(p_dep)

    res = {}
    for thr in THRESHOLDS:
        rare = is_gcg & (prob > thr)
        n_rare = int(rare.sum())
        counters = dict(n_rare_prompts=n_rare, partitions=0, skipped_no_rare_deploy=0,
                        hidden=0, hidden_top_rare_in_window_active=0,
                        hidden_top_rare_in_window=0,
                        visible=0, visible_top_rare_in_window_active=0,
                        visible_top_rare_in_window=0, b_nonneg=0)
        if n_rare == 0:
            res[f"p>{thr}"] = counters
            continue
        for _ in range(n_partitions):
            perm = rng.permutation(n_total)
            fit_idx, dep_idx = perm[:M], perm[M:M + N]
            fit_psi = psi[fit_idx]
            order_f = np.argsort(-fit_psi)
            topk_local = order_f[:k]
            x = fit_psi[topk_local]
            xm, ym = x.mean(), y_fit.mean()
            sxx = float(((x - xm) ** 2).sum())
            dep_rare_local = rare[dep_idx]
            if not dep_rare_local.any():
                counters["skipped_no_rare_deploy"] += 1
                continue
            counters["partitions"] += 1
            hidden = not rare[fit_idx[topk_local]].any()
            if sxx <= 1e-20:
                b = 0.0
            else:
                b = float(((x - xm) * (y_fit - ym)).sum()) / sxx
            a = ym - b * xm
            if b >= 0.0:
                counters["b_nonneg"] += 1
            dep_psi = psi[dep_idx]
            order_d = np.argsort(-dep_psi)
            window_local = order_d[:n_ext]
            if b == 0.0:
                pred = np.full(n_ext, x.max())
            else:
                pred = (log_p_dep - a) / b
            active = dep_psi[window_local] >= pred
            # highest-scoring rare deploy prompt
            rare_positions = np.nonzero(dep_rare_local)[0]
            top_rare = rare_positions[np.argmax(dep_psi[rare_positions])]
            in_window = np.nonzero(window_local == top_rare)[0]
            side = "hidden" if hidden else "visible"
            counters[side] += 1
            if in_window.size:
                counters[f"{side}_top_rare_in_window"] += 1
                if active[in_window[0]]:
                    counters[f"{side}_top_rare_in_window_active"] += 1
        res[f"p>{thr}"] = counters
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--partitions", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tok_cache: dict = {}
    rows = []
    for rd in args.run_dir:
        print(f"loading {rd} ...", flush=True)
        for entry in target_pool(Path(rd), tok_cache):
            r = simulate(entry, args.partitions, args.k, rng)
            rows.append({"run_dir": entry["run_dir"], "target": entry["target"], "sim": r})
            c = r["p>0.5"]
            h, v = max(c["hidden"], 1), max(c["visible"], 1)
            print(f"  {entry['target']}: rare(p>0.5)={c['n_rare_prompts']} "
                  f"P(hidden)={c['hidden']}/{c['partitions'] or 1} "
                  f"active|hidden={c['hidden_top_rare_in_window_active']}/{h} "
                  f"active|visible={c['visible_top_rare_in_window_active']}/{v}", flush=True)

    # pooled summary
    summary = {}
    for thr in THRESHOLDS:
        key = f"p>{thr}"
        tot = {}
        for row in rows:
            for ck, cv in row["sim"][key].items():
                tot[ck] = tot.get(ck, 0) + cv
        summary[key] = tot
    out = {"partitions_per_target": args.partitions, "k": args.k,
           "rows": rows, "pooled": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\npooled:")
    for key, t in summary.items():
        if not t.get("partitions"):
            continue
        print(f"  {key}: P(hidden)={t['hidden']/t['partitions']:.3f} | "
              f"P(top-rare in window & active | hidden)="
              f"{t['hidden_top_rare_in_window_active']/max(t['hidden'],1):.4f} | "
              f"in-window|hidden={t['hidden_top_rare_in_window']/max(t['hidden'],1):.4f} | "
              f"active|visible={t['visible_top_rare_in_window_active']/max(t['visible'],1):.4f} | "
              f"b>=0 frac={t['b_nonneg']/t['partitions']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
