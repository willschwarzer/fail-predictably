from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from . import train_rlquant as tr


def _load_cfg(seed_dir: Path, *, device: str) -> tr.RLQuantConfig:
    with (seed_dir / "config.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    valid = {field.name for field in fields(tr.RLQuantConfig)}
    cfg_kwargs = {key: value for key, value in payload.items() if key in valid}
    cfg = tr.RLQuantConfig(**cfg_kwargs)
    cfg.outdir = str(seed_dir)
    cfg.device = device or str(cfg.device or "cpu")
    return cfg


def _load_manifest_suite(cfg: tr.RLQuantConfig) -> tuple[tr.TaskManifestSuite, tr.TaskMaterializationCache | None]:
    if tr._bank_mode_enabled(cfg):
        print("Loading task bank for eval refresh", flush=True)
        task_bank_info = tr._resolve_task_bank_info(cfg)
        bank_check = tr._validate_task_bank_compatibility(cfg, task_bank_info)
        if not bank_check["passed"]:
            raise RuntimeError("Task bank compatibility check failed: " + "; ".join(bank_check["failures"]))
        if getattr(cfg, "manifest_grouping", "iid") == "slip_level":
            manifest_suite = tr._build_manifest_suite_from_bank_slip_grouped(
                cfg, seed=cfg.seed, bank_info=task_bank_info
            )
        else:
            manifest_suite = tr._build_manifest_suite_from_bank(cfg, seed=cfg.seed, bank_info=task_bank_info)
        if cfg.bank_materialization == "eager":
            manifest_suite, _counts = tr._materialize_manifest_suite(manifest_suite, cfg=cfg)
            materialization_cache: tr.TaskMaterializationCache | None = None
        else:
            materialization_cache = {}
    else:
        print("Rebuilding manifest suite online for eval refresh", flush=True)
        manifest_suite = tr._build_manifest_suite(cfg, seed=cfg.seed)
        materialization_cache = None
    return manifest_suite, materialization_cache


def _make_model_factory(
    manifest_suite: tr.TaskManifestSuite,
    cfg: tr.RLQuantConfig,
):
    reference_task = tr._manifest_reference_task(manifest_suite, cfg=cfg)
    feature_channels = reference_task.feature_tensor.shape[0]
    task_dim = reference_task.task_vector.shape[0]

    def _make_model() -> tr.DensePolicyNet:
        return tr.DensePolicyNet(
            feature_channels,
            task_dim,
            rows=cfg.rows,
            cols=cfg.cols,
            horizon=cfg.horizon,
            hidden_channels=cfg.hidden_channels,
            task_hidden_dim=cfg.task_hidden_dim,
        )

    return _make_model


def rerun_eval(seed_dir: Path, *, device: str = "cpu") -> None:
    cfg = _load_cfg(seed_dir, device=device)
    tr._seed_everything(cfg.seed, deterministic_debug=cfg.deterministic_debug)
    manifest_suite, materialization_cache = _load_manifest_suite(cfg)
    make_model = _make_model_factory(manifest_suite, cfg)

    final_model = make_model().to(cfg.device)
    tr._load_checkpoint(final_model, str(seed_dir / "final_checkpoint.pt"))
    final_report, final_tail_gap, final_traces = tr._evaluate_policy_report_and_tail_gap(
        final_model,
        manifest_suite,
        cfg=cfg,
        label="rerun_final",
        materialization_cache=materialization_cache,
    )

    pretrained_model = make_model().to(cfg.device)
    tr._load_checkpoint(pretrained_model, str(seed_dir / "pretrained_checkpoint.pt"))
    pretrained_report, pretrained_tail_gap, pretrained_traces = tr._evaluate_policy_report_and_tail_gap(
        pretrained_model,
        manifest_suite,
        cfg=cfg,
        label="rerun_pretrained",
        materialization_cache=materialization_cache,
    )

    tr._save_forecast_trace_archive(
        seed_dir,
        pretrained_traces=pretrained_traces,
        meta_learned_traces=final_traces,
        cfg=cfg,
    )

    final_results_path = seed_dir / "final_results.json"
    final_results = {}
    if final_results_path.exists():
        with final_results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            final_results = payload
    final_results["pretrained"] = pretrained_report
    final_results["meta_learned"] = final_report
    final_results["tail_gap_checks"] = {
        "pretrained": pretrained_tail_gap,
        "meta_learned": final_tail_gap,
    }
    final_results["forecast_traces"] = {
        "archive": "forecast_traces.npz",
        "metadata": "forecast_traces_manifests.json",
        "dtype": "float32",
        "fields": ["fit_returns", "fit_regrets", "deploy_returns", "deploy_regrets"],
    }
    with final_results_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    pretrained_results_path = seed_dir / "pretrained_results.json"
    pretrained_results = {
        "exact_eval_phase": cfg.pretrained_exact_eval_phase,
        "quickcheck": final_results.get("pretrained_quickcheck"),
        "report": pretrained_report,
        "tail_gap_check": pretrained_tail_gap,
        "difficulty_audit": tr._deprecated_difficulty_audit_note(),
    }
    if pretrained_results_path.exists():
        with pretrained_results_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            pretrained_results["quickcheck"] = payload.get("quickcheck")
            pretrained_results["exact_eval_phase"] = payload.get(
                "exact_eval_phase",
                pretrained_results["exact_eval_phase"],
            )
            pretrained_results["difficulty_audit"] = payload.get(
                "difficulty_audit",
                pretrained_results["difficulty_audit"],
            )
    with pretrained_results_path.open("w", encoding="utf-8") as f:
        json.dump(pretrained_results, f, indent=2)

    print(f"Refreshed eval artifacts for {seed_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rerun exact RLQuant evaluation for an existing seed directory")
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    rerun_eval(Path(args.seed_dir), device=args.device)


if __name__ == "__main__":
    main()
