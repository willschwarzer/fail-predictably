"""Mid-training held-out read: score adapter snapshots on a run's held-out pools.

The lqdyn arms save LoRA snapshots every 10 steps but score their held-out
pools only at the end. This builds the held-out records with the trainer's own
code (bank loader, split_pool, target partition, fit/deploy split), loads each
snapshot into the trainer's PolicyLM the way --eval-adapter-from does, and
scores fit + deploy for every held-out pool with compute_password_logprobs_batched,
once with the adapter disabled (base) and once per snapshot, all on the same
GPU so the trained-minus-base difference carries no cross-GPU numerics. Writes
<run>/snapshot_evals/{base,stepNNNN}_test_forecast_traces{.npz,_metadata.json}
in the trainer's trace format, then prints poolscale_readout's multi-split read
(200 splits per pool, seeded by pool name; k 10 / 20 / 50, the run's own k
starred) for each snapshot against this job's base, and the base's agreement
with the run's own baseline traces.

  python snapshot_heldout_eval.py RUN_DIR STEP [STEP ...]
"""
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "readouts"))
from forecast_experiment import train_lquant as tl  # noqa: E402
from forecast_experiment.lquant_prompt_bank import resolve_prompt_bank  # noqa: E402
from importance_sampling.policy_lm import LoRASetup, PolicyLM  # noqa: E402
from poolscale_readout import S, load, split_errors  # noqa: E402

KS = (10, 20, 50)
SNAPSHOT_WAIT_SEC = 4 * 3600


def make_cfg(run_dir: Path) -> "tl.LQuantConfig":
    raw = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in dataclasses.fields(tl.LQuantConfig)}
    return tl.LQuantConfig(**{k: v for k, v in raw.items() if k in names})


def heldout_records(cfg, tokenizer, run_dir: Path):
    bank_info = resolve_prompt_bank(cfg.prompt_bank_root, tl._effective_prompt_bank_seed(cfg))
    # The same loader dispatch as the trainer: WildChat banks are token_union,
    # the password banks predate the field and default to password.
    if str(bank_info.metadata.get("task_kind", "password")) == "token_union":
        pool = tl._load_prompt_pool_from_bank_token_union(bank_info, cfg=cfg, tokenizer=tokenizer)
    else:
        pool = tl._load_prompt_pool_from_bank(bank_info, cfg=cfg, tokenizer=tokenizer)
    train_recs, eval_recs, test_recs = tl.split_pool(pool, cfg)
    part = tl._resolve_target_partition(list(pool["targets"]), cfg)
    saved = json.loads((run_dir / "target_partition.json").read_text())
    if list(part["heldout_targets"]) != list(saved["heldout_targets"]):
        raise RuntimeError(f"partition mismatch: {part['heldout_targets']} vs {saved['heldout_targets']}")
    out = []
    for t in part["heldout_targets"]:
        fit, dep = tl._split_test_records_for_target(t, test_recs[t], train_recs, eval_recs, cfg)
        out.append((t, fit, dep))
    return out


def score_all(model, tokenizer, items, cfg, mb):
    rows = []
    for t, fit, dep in items:
        with torch.no_grad():
            kw = dict(enable_thinking=(not cfg.disable_thinking), device=cfg.device, micro_batch_size=mb,
                      require_grad=False, score_max_len=cfg.score_max_len, token_budget=cfg.score_token_budget)
            f = tl.compute_password_logprobs_batched(model, tokenizer, fit, **kw)
            d = tl.compute_password_logprobs_batched(model, tokenizer, dep, **kw)
        rows.append({"target": t, "fit_log_probs": f.float().cpu().numpy(), "deploy_log_probs": d.float().cpu().numpy()})
    return rows


def readout(base, trained, k_own):
    out = {}
    # A split's OLS reads the fit top-k, so k must stay below the fit size
    # (password pools fit M = 44: k10 and k20 only).
    m_min = min(M for _, M in base.values())
    for k in (k for k in KS if k < m_min):
        sb = st = 0.0
        d = []
        for t, (vb, M) in base.items():
            vt = trained[t][0]
            rng = np.random.default_rng(int.from_bytes(t.encode(), "little") % (2**32))
            perms = np.stack([rng.permutation(vb.size) for _ in range(S)])
            eb, _ = split_errors(vb, perms, M, k=k)
            et, _ = split_errors(vt, perms, M, k=k)
            sb += (eb ** 2).mean()
            st += (et ** 2).mean()
            d.append(np.abs(et).mean() - np.abs(eb).mean())
        d = np.array(d)
        out[k] = (sb / st, d.mean(), d.std(ddof=1) / math.sqrt(d.size))
    return " | ".join(
        f"k{k}{'*' if k == k_own else ''} mse fold {f:.2f}x (abs delta {m:+.3f} +/- {s:.3f})"
        for k, (f, m, s) in out.items()
    )


def main():
    run_dir = Path(sys.argv[1]).resolve()
    steps = [int(s) for s in sys.argv[2:]]
    # A reader chained with --dependency=after:<train job> can start with its
    # run, before the trainer has written its config, target partition and
    # baseline traces (a reader that did not wait died on the missing config.json): wait for them.
    need = [run_dir / n for n in ("config.json", "target_partition.json", "baseline_test_forecast_traces.npz",
                                  "baseline_test_forecast_traces_metadata.json")]
    t0 = time.time()
    while not all(f.exists() for f in need):
        if time.time() - t0 > 3600:
            raise RuntimeError(f"{run_dir} still lacks {[f.name for f in need if not f.exists()]} after an hour")
        time.sleep(30)
    cfg = make_cfg(run_dir)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**20
    mb = int(os.environ.get("MICRO_BATCH", 16 if vram < 30000 else 48))
    print(f"{run_dir.name}: steps {steps}, GPU {torch.cuda.get_device_name(0)} ({vram:.0f} MB), micro-batch {mb}", flush=True)
    policy = PolicyLM(
        model_id=cfg.model_id, device=cfg.device,
        lora=LoRASetup(use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
                       dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules),
    )
    model, tokenizer = policy.model, policy.tokenizer
    model.eval()
    items = heldout_records(cfg, tokenizer, run_dir)
    outdir = run_dir / "snapshot_evals"
    outdir.mkdir(exist_ok=True)

    # The base pass is the same for every snapshot of a run: reuse a saved one
    # when it was scored on the same GPU model (other models shift bf16 scores
    # by ~0.06 nats median, which would leak into the trained-minus-base read).
    gpu = torch.cuda.get_device_name(0)
    stamp = outdir / "base_gpu.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("gpu") == gpu \
            and (outdir / "base_test_forecast_traces.npz").exists():
        print(f"reusing the base pass scored on {gpu}", flush=True)
    else:
        with model.disable_adapter():
            rows = score_all(model, tokenizer, items, cfg, mb)
        tl._save_lquant_test_trace_archive(outdir, prefix="base_", target_traces=rows, cfg=cfg)
        stamp.write_text(json.dumps({"gpu": gpu, "micro_batch": mb}))
    base = load(outdir, "base_")
    ref = load(run_dir, "baseline_")
    gap = np.concatenate([np.abs(base[t][0] - ref[t][0]) for t in ref])
    print(f"base vs the run's baseline traces: median |diff| {np.median(gap):.4f}, "
          f"99th pct {np.quantile(gap, 0.99):.4f}, max {gap.max():.4f} nats", flush=True)

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    for step in steps:
        prefix = f"step{step:04d}_"
        # A step already scored against this base (a requeued or repeated
        # job) is read back, not re-scored.
        if (outdir / f"{prefix}test_forecast_traces.npz").exists() and \
                (outdir / f"{prefix}test_forecast_traces.npz").stat().st_mtime >= stamp.stat().st_mtime:
            trained = load(outdir, prefix)
            top = np.mean([base[t][0].max() - trained[t][0].max() for t in base])
            med = np.mean([np.median(base[t][0]) - np.median(trained[t][0]) for t in base])
            print(f"step {step:3d}: {readout(base, trained, int(cfg.ols_top_k))} | drop top {top:.2f} "
                  f"median {med:.2f} nats (saved)", flush=True)
            continue
        snap = run_dir / "snapshots" / f"step_{step:04d}"
        # A later snapshot may not exist yet while the arm trains: wait for it,
        # and for its writer to finish (a minute without modification).
        f = snap / "adapter_model.safetensors"
        t0 = time.time()
        while not (f.exists() and time.time() - f.stat().st_mtime > 60):
            if time.time() - t0 > SNAPSHOT_WAIT_SEC:
                raise RuntimeError(f"snapshot {f} did not appear within {SNAPSHOT_WAIT_SEC} s")
            time.sleep(60)
        state = load_file(str(f))
        before = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
        res = set_peft_model_state_dict(model, state)
        if list(getattr(res, "unexpected_keys", []) or []):
            raise RuntimeError(f"unexpected adapter keys at {snap}")
        after = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
        if abs(after - before) < 1e-8:
            raise RuntimeError(f"adapter at {snap} left the trainable weights unchanged")
        model.eval()
        rows = score_all(model, tokenizer, items, cfg, mb)
        tl._save_lquant_test_trace_archive(outdir, prefix=prefix, target_traces=rows, cfg=cfg)
        trained = load(outdir, prefix)
        top = np.mean([base[t][0].max() - trained[t][0].max() for t in base])
        med = np.mean([np.median(base[t][0]) - np.median(trained[t][0]) for t in base])
        print(f"step {step:3d}: {readout(base, trained, int(cfg.ols_top_k))} | drop top {top:.2f} "
              f"median {med:.2f} nats", flush=True)


if __name__ == "__main__":
    main()
