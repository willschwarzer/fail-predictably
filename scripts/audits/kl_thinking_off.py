"""Held-out single-token KL with the chat template's thinking switched off, beside the trainer's own.

train_lquant.load_kl_dataset formats the held-out KL prompts with apply_chat_template and no
enable_thinking, so Qwen3's default (thinking on) applies: the generation prompt ends at
"<|im_start|>assistant\\n", and at that position, where the trainer measures its held-out KL, the base
model puts about 0.999 on "<think>" (the base_top1_on column below, with prompts cut from the left).
Training, its KL regularizer and the risk scores all use disable_thinking = True, whose generation prompt
ends after an empty "<think>\\n\\n</think>\\n\\n" block. For each snapshot this recomputes the held-out KL
both ways, on the same 256 prompts, tokenized as the run's config says (kl_tokenize_max_len,
kl_truncation_side):
  kl_on    thinking on: the trainer's prompts (kl_on_trainer is kl_eval_results.json, at a run's last step)
  kl_off   thinking off: the first token after the empty think block, the position scoring uses
with each mode's mean top-1 probability under the base and the trained model, and n_cut, the prompts
longer than kl_tokenize_max_len (under right truncation those lose their generation marker).
The base model is the same LoRA model with its adapter disabled; its log-probabilities are computed
once per prompt set and mode. The paper reports these values in the appendix "Forecastability training
with minimal change to the model" (paragraph "Both terms together changed the model least").

  python scripts/audits/kl_thinking_off.py --out CSV [--steps 100 200] [--dataset password|wildchat]
        every seed of the four 2x2 arms, read from the runs that
        scripts/minimal_change/readouts/strict_table.py reads (under $RESULTS_ROOT/lquant_dynamics/runs/)
  python scripts/audits/kl_thinking_off.py --out CSV RUN_DIR:STEP [RUN_DIR:STEP ...]
Rows already in CSV (same run and step) are skipped, so a requeued job resumes where it stopped.
--truncation-side left|right overrides the runs' kl_truncation_side (the password runs truncate on the
right, so a prompt longer than kl_tokenize_max_len loses its generation marker; left keeps it). The
override changes only how prompts are cut, so kl_on then differs from kl_on_trainer for the cut prompts.
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
READOUT = REPO_ROOT / "scripts/minimal_change/readouts"
sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment import train_lquant as tl  # noqa: E402
from importance_sampling.policy_lm import LoRASetup, PolicyLM  # noqa: E402

KEEP = ("anchpen + band (main)", "IOG alone", "anchpen, no band", "IOG + band")
COLS = ["dataset", "arm", "seed", "step", "run", "kl_on", "kl_on_trainer", "kl_off", "kl_on_median",
        "kl_off_median", "base_top1_on", "base_top1_off", "trained_top1_on", "trained_top1_off",
        "n_prompts", "n_cut", "max_len", "truncation_side"]
LORA_KEYS = ("model_id", "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")


class ThinkingOff:
    """Tokenizer proxy whose chat template defaults to enable_thinking=False."""

    def __init__(self, tok):
        self._tok = tok

    def apply_chat_template(self, *a, **k):
        k.setdefault("enable_thinking", False)
        return self._tok.apply_chat_template(*a, **k)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def make_cfg(run_dir):
    raw = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in dataclasses.fields(tl.LQuantConfig)}
    return tl.LQuantConfig(**{k: v for k, v in raw.items() if k in names}), raw


def arm_specs(steps, dataset):
    sys.path.insert(0, str(READOUT))
    from strict_table import ARMS, SEEDS
    out = []
    for ds, arm, name in ARMS:
        if arm not in KEEP or (dataset and ds != dataset):
            continue
        for step in steps:
            for s in SEEDS:
                rd = RUNS / name(s, step)
                f = rd / "snapshots" / f"step_{step:04d}" / "adapter_model.safetensors"
                if f.exists() and time.time() - f.stat().st_mtime > 60:
                    out.append((ds, arm, s, step, rd))
                else:
                    print(f"skip {rd.name} step {step}: no finished snapshot", flush=True)
    return out


def trainer_kl(run_dir, raw, step):
    f = run_dir / "kl_eval_results.json"
    snap = run_dir / "snapshots" / f"step_{step:04d}"
    if int(raw.get("steps", -1)) != step or not f.exists() or f.stat().st_mtime < snap.stat().st_mtime:
        return ""
    return float(json.loads(f.read_text())["single_token"])


def prompts_for(cfg, tok, off):
    t = ThinkingOff(tok) if off else tok
    ps = []
    for ds in cfg.kl_eval_datasets:
        ps.extend(tl.load_kl_dataset(ds, t, n_samples=256))
    return ps[:256]


def last_logprobs(model, tok, prompts, cfg):
    out = []
    with torch.no_grad():
        for s in range(0, len(prompts), max(1, cfg.kl_batch_size)):
            ids, mask = tl._tokenize_for_kl(tok, prompts[s:s + cfg.kl_batch_size], cfg.device,
                                            max_len=cfg.kl_tokenize_max_len,
                                            truncation_side=cfg.kl_truncation_side)
            out.append(F.log_softmax(tl._kl_last_logits(model, ids, mask), -1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 200])
    ap.add_argument("--dataset", choices=["password", "wildchat"], default=None)
    ap.add_argument("--truncation-side", choices=["left", "right"], default=None)
    ap.add_argument("specs", nargs="*")
    a = ap.parse_args()
    if a.specs:
        specs = []
        for x in a.specs:
            rd, step = x.rsplit(":", 1)
            rd = Path(rd)
            raw = json.loads((rd / "config.json").read_text())
            specs.append(("wildchat" if rd.name.startswith("lqdyn") else "password", "", raw.get("seed", ""),
                          int(step), rd))
    else:
        specs = arm_specs(a.steps, a.dataset)
    out = Path(a.out)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["run"], int(r["step"])) for r in csv.DictReader(fh)}
    todo = [s for s in specs if (s[4].name, s[3]) not in done]
    print(f"{len(specs)} snapshots, {len(specs) - len(todo)} already in {out}, {len(todo)} to read", flush=True)
    if not todo:
        return
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    model = tok = lora0 = None
    base_cache, prompt_cache = {}, {}
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new_file:
        w.writeheader()
        fh.flush()
    for ds, arm, seed, step, run_dir in todo:
        t0 = time.time()
        cfg, raw = make_cfg(run_dir)
        if a.truncation_side:
            cfg = dataclasses.replace(cfg, kl_truncation_side=a.truncation_side)
        lora = tuple(json.dumps(raw.get(k), sort_keys=True) for k in LORA_KEYS)
        if model is None:
            policy = PolicyLM(model_id=cfg.model_id, device=cfg.device,
                              lora=LoRASetup(use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                             dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules))
            model, tok, lora0 = policy.model, policy.tokenizer, lora
            model.eval()
        elif lora != lora0:
            raise RuntimeError(f"{run_dir.name}: model or LoRA settings differ from the first run's")
        f = run_dir / "snapshots" / f"step_{step:04d}" / "adapter_model.safetensors"
        before = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
        res = set_peft_model_state_dict(model, load_file(str(f)))
        if list(getattr(res, "unexpected_keys", []) or []):
            raise RuntimeError(f"unexpected adapter keys in {f}")
        after = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
        if abs(after - before) < 1e-8:
            raise RuntimeError(f"adapter {f} left the trainable weights unchanged")
        model.eval()
        row = dict(dataset=ds, arm=arm, seed=seed, step=step, run=run_dir.name,
                   kl_on_trainer=trainer_kl(run_dir, raw, step), max_len=cfg.kl_tokenize_max_len,
                   truncation_side=cfg.kl_truncation_side)
        for off in (False, True):
            key = (tuple(cfg.kl_eval_datasets), cfg.kl_tokenize_max_len, cfg.kl_truncation_side, off)
            if key not in prompt_cache:
                ps = prompts_for(cfg, tok, off)
                n_cut = sum(len(tok(p).input_ids) > cfg.kl_tokenize_max_len for p in ps)
                prompt_cache[key] = (ps, n_cut)
                print(f"  prompt set {'off' if off else 'on'} for {cfg.kl_eval_datasets[0][-60:]}: "
                      f"{len(ps)} prompts, {n_cut} longer than {cfg.kl_tokenize_max_len}; prompt 0 ends "
                      f"{ps[0][-48:]!r}", flush=True)
                with model.disable_adapter():
                    base_cache[key] = last_logprobs(model, tok, ps, cfg)
            ps, n_cut = prompt_cache[key]
            lq = base_cache[key]
            lp = last_logprobs(model, tok, ps, cfg)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            tag = "off" if off else "on"
            row[f"kl_{tag}"] = f"{kl.mean():.6g}"
            row[f"kl_{tag}_median"] = f"{kl.median():.6g}"
            row[f"base_top1_{tag}"] = f"{lq.max(-1).values.exp().mean():.4f}"
            row[f"trained_top1_{tag}"] = f"{lp.max(-1).values.exp().mean():.4f}"
            row["n_prompts"], row["n_cut"] = len(ps), n_cut
        w.writerow(row)
        fh.flush()
        print(f"{ds:8s} {arm:22s} s{seed} step {step}: KL on {row['kl_on']} (trainer {row['kl_on_trainer']}) | "
              f"off {row['kl_off']} | base top-1 on {row['base_top1_on']} off {row['base_top1_off']} | "
              f"{time.time() - t0:.0f}s", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
