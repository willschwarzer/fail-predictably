"""Checkpoint selection for the natural-prompt runs (paper appendix "Forecastability on natural
user prompts").

The paper's rule: select each run's checkpoint by its training-side objective, among saved
checkpoints whose single-token KL on the 256-prompt evaluation set is at most 0.7. Held-out
errors never enter selection.

Both trainers write `best/` and `best_step.json` whenever their training-side objective
improves. The meta-trainer (run_train_lquant) uses the per-step `train_forecast_sq_log10_error`,
on a strict new minimum. The SFT trainer (run_train_lquant_sft) uses the mean SFT loss over each
`save_every` window, at the window's last step. `final/` is the endpoint. This script rebuilds
that chain of best-so-far checkpoints from `history.json`. It matches each evaluation of a saved
checkpoint (the `..._besteval_<arm>` runs of run_lquant_ldnoobw_eval_adapter.sbatch) to a chain
step: its adapter directory sits next to a `best_step.json` whose step and objective value
equal that chain step exactly. It then applies the rule. The endpoint counts as a saved
checkpoint too; its KL is the training run's own end-of-run `kl_eval_results.json`. On a tie in
the objective, a best/ checkpoint wins over the endpoint.

The trainer keeps only the latest best/. When its KL fails the bound, an earlier best-so-far
checkpoint can be chosen only if a copy of it was saved and evaluated; otherwise the rule falls
back to the endpoint. The script prints every chain step after the chosen one that has no
evaluation, since evaluating it could change the choice.

Usage:
  python scripts/wildchat/natural_prompts/select_natural_checkpoint.py RUN_DIR [RUN_DIR ...]
RUN_DIR is a training run's seed directory, e.g.
$RESULTS_ROOT/lquant_wildchat_ldnoobw_qwen_qwen3-8b_iog_full_boot1_cache296_seed17_mb8klnf05/seed_17.
Evaluations are found under --results-root (default $RESULTS_ROOT) as
lquant_wildchat_ldnoobw_<model tag>_iog_full_boot1_cache296_seed<S>_besteval_*/seed_<S>.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts" / "results"))
KL_MAX = 0.7


@dataclass
class Candidate:
    label: str          # "<arm>at<step>" (the eval directory's suffix) or "final"
    step: int | None    # None for the endpoint
    objective: float    # training-side objective (lower is better)
    kl: float
    result_dir: Path    # directory holding test_results.json / kl_eval_results.json


def _load(path: Path):
    return json.loads(Path(path).read_text())


def best_chain(run_dir: Path) -> tuple[str, list[tuple[int, float]], float]:
    """(metric name, [(step, objective)] at each best/ rewrite, endpoint objective)."""
    cfg = _load(run_dir / "config.json")
    hist = sorted(_load(run_dir / "history.json"), key=lambda h: h["step"])
    chain: list[tuple[int, float]] = []
    best = None
    if "sft_batch_per_step" in cfg:  # SFT trainer: interval mean of the loss per save window
        metric = "interval_mean_sft_loss"
        se = int(cfg["save_every"])
        means = []
        for i, h in enumerate(hist):
            if se > 0 and (h["step"] + 1) % se == 0:
                iv = hist[max(0, i - se + 1): i + 1]
                m = sum(x["loss"] for x in iv) / len(iv)
                means.append(m)
                if best is None or m < best:
                    best = m
                    chain.append((h["step"], m))
        endpoint = means[-1]
    else:  # meta-trainer: per-step training-side squared log10 forecast error
        metric = "train_forecast_sq_log10_error"
        min_step = int(cfg.get("best_ckpt_min_step", 0))
        vals = [(h["step"], h.get(metric)) for h in hist if h.get(metric) is not None]
        for step, v in vals:
            if step >= min_step and (best is None or v < best):
                best = v
                chain.append((step, v))
        endpoint = vals[-1][1]
    bs = _load(run_dir / "best_step.json")
    assert chain and chain[-1][0] == bs["step"] and chain[-1][1] == bs[metric], \
        (run_dir, "history.json does not reproduce best_step.json", chain[-1:], bs)
    return metric, chain, endpoint


@functools.lru_cache(maxsize=None)
def _listing(results_root: Path) -> tuple[str, ...]:
    """Names in results_root, read once (the directory is large)."""
    return tuple(sorted(os.listdir(results_root)))


def _resolve(p: str, results_root: Path) -> Path:
    """An adapter path as recorded in an evaluation's config.json. Launchers record it as given,
    often relative to the repo root (artifacts/results/...); that part is re-rooted at
    results_root, so the results tree can live anywhere."""
    q = Path(p)
    if q.exists():
        return q
    if "artifacts/results/" in p:
        r = Path(results_root) / p.split("artifacts/results/", 1)[1]
        if r.exists():
            return r
    if not q.is_absolute() and (REPO_ROOT / q).exists():
        return REPO_ROOT / q
    return q


def candidates(run_dir: Path, results_root: Path = RESULTS_ROOT) -> tuple[list[Candidate], list[tuple[int, float]]]:
    run_dir = Path(run_dir)
    cfg = _load(run_dir / "config.json")
    metric, chain, endpoint = best_chain(run_dir)
    by_step = dict(chain)
    seed = int(cfg["seed"])
    model_tag = cfg["model_id"].replace("/", "_").replace(".", "_").lower()
    out = []
    prefix = f"lquant_wildchat_ldnoobw_{model_tag}_iog_full_boot1_cache296_seed{seed}_besteval_"
    for name in _listing(Path(results_root)):
        if not name.startswith(prefix):
            continue
        ev = Path(results_root) / name
        ev_seed = ev / f"seed_{seed}"
        try:
            ecfg = _load(ev_seed / "config.json")
            kl = _load(ev_seed / "kl_eval_results.json")["single_token"]
            _load(ev_seed / "test_results.json")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue  # unfinished or foreign evaluation
        adapter = ecfg.get("eval_adapter_from") or ""
        if not adapter:
            continue
        bs_path = _resolve(adapter, results_root).parent / "best_step.json"
        if not bs_path.exists():
            continue
        bs = _load(bs_path)
        step = bs.get("step")
        if step in by_step and bs.get(metric) == by_step[step]:
            out.append(Candidate(ev.name.split("_besteval_", 1)[1], step, by_step[step], float(kl), ev_seed))
    kl_end = run_dir / "kl_eval_results.json"
    if kl_end.exists() and (run_dir / "test_results.json").exists():
        out.append(Candidate("final", None, endpoint, float(_load(kl_end)["single_token"]), run_dir))
    return out, chain


def select(run_dir: Path, results_root: Path = RESULTS_ROOT, kl_max: float = KL_MAX):
    """(chosen Candidate or None, all candidates, chain, chain steps after the choice with no eval)."""
    cands, chain = candidates(run_dir, results_root)
    ok = [c for c in cands if c.kl <= kl_max]
    chosen = min(ok, key=lambda c: (c.objective, c.step is None)) if ok else None
    evaluated = {c.step for c in cands if c.step is not None}
    floor = chosen.step if chosen is not None and chosen.step is not None else -1
    unevaluated_later = [s for s, _ in chain if s > floor and s not in evaluated]
    return chosen, cands, chain, unevaluated_later


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", type=Path, help="training run seed directories")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--kl-max", type=float, default=KL_MAX)
    args = ap.parse_args()
    for rd in args.run_dirs:
        chosen, cands, chain, later = select(rd, args.results_root, args.kl_max)
        print(f"{rd}")
        print(f"  best/ chain (last 5 of {len(chain)}): {[s for s, _ in chain[-5:]]}")
        for c in sorted(cands, key=lambda c: c.objective):
            flag = "ok " if c.kl <= args.kl_max else "KL>"
            print(f"  {flag} {c.label:24s} objective={c.objective:.6g} kl={c.kl:.4g}")
        print(f"  chosen: {chosen.label if chosen else None}")
        if later:
            print(f"  note: best/ steps after the choice without an evaluation: {later}")


if __name__ == "__main__":
    main()
