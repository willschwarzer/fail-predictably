"""Held-out single-token KL with thinking on and off, for runs that kept only a final adapter (seed_*/final/).

Why: train_lquant.load_kl_dataset formats the held-out KL prompts with apply_chat_template and no
enable_thinking, so Qwen3's default (thinking on) applies, while the runs' task prompts and risk scores use
disable_thinking = True. kl_thinking_off.py recomputes the KL both ways for the minimal-change snapshots
(snapshots/step_XXXX/). The runs behind the password-game figure (Figure 4, whose KL panel
scripts/figures/plot_lquant_three_metric_bars.py draws from each seed's kl_eval_results.json) kept only
seed_*/final/adapter_model.safetensors, so this reads those. It imports kl_thinking_off.py's prompt and
logit functions unchanged. Three modes, on the same prompt sample (load_kl_dataset, seed 42):
  on_right   thinking on, right truncation: the trainer's own measurement (kl_on_trainer is the run's
             kl_eval_results.json, for checking the recomputation)
  off_right  thinking off (the position the risk scores use), the trainer's truncation
  off_left   thinking off, left truncation, which keeps the generation marker of prompts longer than
             max_len (right truncation cuts it, so for those prompts the KL is taken mid-prompt)
KL is KL(trained || base) of the next-token distribution at the last prompt position, as in the trainer.
"_right" in the first two modes' column names means the run's own kl_truncation_side and kl_tokenize_max_len
(the config's values, or the trainer defaults 512 and right when the config predates them). That is right
truncation at 512 for the Figure 4 runs, but left truncation at 1024 for the natural-prompt runs, where
all three modes then differ only in the template. Each mode also prints the three most common argmax first
tokens of the trained and the base model, and the run ends with each label's mean KL per mode
(kl_thinking_off_final_summary.py prints the same means to more digits). The paper reports these values in
the appendix "Experimental details" (paragraph "KL divergence from the base model") and the appendix
"Forecastability on natural user prompts" (paragraph "What these runs measure").

  python scripts/audits/kl_thinking_off_final.py --out CSV [--adapter-subdir best] [--generate N] [LABEL=GLOB ...]
GLOB matches run directories under $RESULTS_ROOT, each holding seed_*/. The default is the two trained
arms of Figure 4's KL panel. Rows already in CSV (same seed_dir) are skipped, so a requeued job resumes.
Read-only on the run directories.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kl_thinking_off as kto  # noqa: E402  (its prompt and logit functions)
from importance_sampling.policy_lm import LoRASetup, PolicyLM  # noqa: E402

DEFAULT = [
    "Ours=lquant_40slots_modbulkgcg_iog_full_bootstrap1_cache296_seed*_20260505",
    "SFT=lquant_40slots_modbulkgcg_sft_union_seed*_20260504",
]
MODES = (("on_right", False, None), ("off_right", True, None), ("off_left", True, "left"))
COLS = (["label", "run", "seed_dir", "kl_on_trainer"]
        + [f"{c}_{m}" for m, _, _ in MODES for c in ("kl", "kl_median", "base_top1", "trained_top1", "n_cut")]
        + ["n_prompts", "max_len", "truncation_side_trainer"])


def cfg_of(raw, truncation_side=None):
    return SimpleNamespace(
        model_id=raw["model_id"], device=raw.get("device", "cuda"),
        use_lora=raw["use_lora"], lora_r=raw["lora_r"], lora_alpha=raw["lora_alpha"],
        lora_dropout=raw["lora_dropout"], lora_target_modules=raw["lora_target_modules"],
        kl_eval_datasets=raw["kl_eval_datasets"], kl_batch_size=int(raw.get("kl_batch_size", 8)),
        kl_tokenize_max_len=int(raw.get("kl_tokenize_max_len", 512)),
        kl_truncation_side=truncation_side or raw.get("kl_truncation_side", "right"))


def gmean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(statistics.mean(math.log(x) for x in xs)) if xs else float("nan")


def summarize(out):
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    print("\nlabel  n  | trainer mean | " + " | ".join(f"{m} mean (gmean)" for m, _, _ in MODES), flush=True)
    stats = {}
    for lab, rs in by.items():
        st = {"trainer": [float(r["kl_on_trainer"]) for r in rs if r["kl_on_trainer"] not in ("", None)]}
        for m, _, _ in MODES:
            st[m] = [float(r[f"kl_{m}"]) for r in rs]
        stats[lab] = st
        tr = statistics.mean(st["trainer"]) if st["trainer"] else float("nan")
        print(f"{lab:5s} {len(rs):2d} | {tr:.4g} | "
              + " | ".join(f"{statistics.mean(st[m]):.4g} ({gmean(st[m]):.4g})" for m, _, _ in MODES), flush=True)
    if "Ours" in stats and "SFT" in stats:
        for m in ("trainer",) + tuple(m for m, _, _ in MODES):
            a, b = stats["SFT"][m], stats["Ours"][m]
            if a and b:
                print(f"SFT / Ours, {m}: ratio of means {statistics.mean(a) / statistics.mean(b):.3g}, "
                      f"ratio of geometric means {gmean(a) / gmean(b):.3g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-subdir", default="final",
                    help="seed_*/<this>/adapter_model.safetensors; e.g. best for runs that saved a selected checkpoint")
    ap.add_argument("--generate", type=int, default=0,
                    help="also greedy-decode this many tokens on the first 3 prompts (thinking off, left truncation), "
                         "from the trained and the base model; printed only")
    ap.add_argument("specs", nargs="*")
    a = ap.parse_args()
    items = []
    for sp in a.specs or DEFAULT:
        label, pat = sp.split("=", 1)
        runs = sorted(glob.glob(str(RESULTS_ROOT / pat)))
        print(f"{label}: {len(runs)} run dirs match {pat}", flush=True)
        for run in runs:
            for sd in sorted(glob.glob(run + "/seed_*")):
                sd = Path(sd)
                if (sd / a.adapter_subdir / "adapter_model.safetensors").exists():
                    items.append((label, sd))
                else:
                    print(f"skip {sd}: no {a.adapter_subdir} adapter", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["seed_dir"] for r in csv.DictReader(fh)}
    todo = [it for it in items if str(it[1]) not in done]
    print(f"{len(items)} adapters, {len(items) - len(todo)} already in {out}, {len(todo)} to read", flush=True)
    if todo:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        model = tok = lora0 = None
        base_cache, prompt_cache = {}, {}
        new_file = not out.exists()
        fh = open(out, "a", newline="")
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new_file:
            w.writeheader()
            fh.flush()
        for label, sd in todo:
            t0 = time.time()
            raw = json.loads((sd / "config.json").read_text())
            cfg = cfg_of(raw)
            lora = tuple(json.dumps(raw.get(k), sort_keys=True) for k in kto.LORA_KEYS)
            if model is None:
                policy = PolicyLM(model_id=cfg.model_id, device=cfg.device,
                                  lora=LoRASetup(use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                                 dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules))
                model, tok, lora0 = policy.model, policy.tokenizer, lora
                model.eval()
                print(f"model {cfg.model_id}, dtype {next(model.parameters()).dtype}", flush=True)
            elif lora != lora0:
                raise RuntimeError(f"{sd}: model or LoRA settings differ from the first run's")
            f = sd / a.adapter_subdir / "adapter_model.safetensors"
            sdict = load_file(str(f))
            before = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
            res = set_peft_model_state_dict(model, sdict)
            if list(getattr(res, "unexpected_keys", []) or []):
                raise RuntimeError(f"unexpected adapter keys in {f}: {list(res.unexpected_keys)[:5]}")
            missing_lora = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
            if missing_lora:
                raise RuntimeError(f"adapter {f} lacks LoRA weights: {missing_lora[:5]}")
            after = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
            if abs(after - before) < 1e-8:
                raise RuntimeError(f"adapter {f} left the trainable weights unchanged")
            model.eval()
            # kl_eval_results.json belongs to the final adapter; for another subdir, use a KL logged in
            # best_step.json if there is one, else leave the check blank.
            kf = sd / "kl_eval_results.json"
            if a.adapter_subdir == "final":
                tr = float(json.loads(kf.read_text())["single_token"]) if kf.exists() else ""
            else:
                bj = sd / "best_step.json"
                bd = json.loads(bj.read_text()) if bj.exists() else {}
                kls = [v for k, v in bd.items() if "kl" in k.lower() and isinstance(v, (int, float))]
                tr = float(kls[0]) if len(kls) == 1 else ""
            row = dict(label=label, run=sd.parent.name, seed_dir=str(sd), kl_on_trainer=tr,
                       max_len=cfg.kl_tokenize_max_len, truncation_side_trainer=cfg.kl_truncation_side)
            for m, off, trunc in MODES:
                mcfg = cfg_of(raw, trunc)
                key = (tuple(mcfg.kl_eval_datasets), mcfg.kl_tokenize_max_len, mcfg.kl_truncation_side, off)
                if key not in prompt_cache:
                    ps = kto.prompts_for(mcfg, tok, off)
                    n_cut = sum(len(tok(p).input_ids) > mcfg.kl_tokenize_max_len for p in ps)
                    prompt_cache[key] = (ps, n_cut)
                    print(f"  prompt set {m}: {len(ps)} prompts, {n_cut} longer than {mcfg.kl_tokenize_max_len}; "
                          f"prompt 0 ends {ps[0][-48:]!r}", flush=True)
                    with model.disable_adapter():
                        base_cache[key] = kto.last_logprobs(model, tok, ps, mcfg)
                ps, n_cut = prompt_cache[key]
                lq = base_cache[key]
                lp = kto.last_logprobs(model, tok, ps, mcfg)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                row[f"kl_{m}"] = f"{kl.mean():.6g}"
                row[f"kl_median_{m}"] = f"{kl.median():.6g}"
                row[f"base_top1_{m}"] = f"{lq.max(-1).values.exp().mean():.4f}"
                row[f"trained_top1_{m}"] = f"{lp.max(-1).values.exp().mean():.4f}"
                row[f"n_cut_{m}"] = n_cut
                # Print only (the CSV columns stay fixed): the most common argmax first tokens.
                fmt = lambda c: ", ".join(f"{tok.decode([t])!r} x{n}" for t, n in c)
                print(f"    {m}: trained argmax {fmt(collections.Counter(lp.argmax(-1).tolist()).most_common(3))} | "
                      f"base argmax {fmt(collections.Counter(lq.argmax(-1).tolist()).most_common(3))}", flush=True)
                row["n_prompts"] = len(ps)
            w.writerow(row)
            fh.flush()
            if a.generate:
                gcfg = cfg_of(raw, "left")
                ps = prompt_cache[(tuple(gcfg.kl_eval_datasets), gcfg.kl_tokenize_max_len, "left", True)][0]
                old_side, old_trunc = tok.padding_side, tok.truncation_side
                tok.padding_side, tok.truncation_side = "left", "left"
                try:
                    for j in range(min(3, len(ps))):
                        enc = tok(ps[j], return_tensors="pt", truncation=True,
                                  max_length=gcfg.kl_tokenize_max_len).to(gcfg.device)
                        n0 = enc["input_ids"].shape[1]
                        kw = dict(max_new_tokens=a.generate, do_sample=False, pad_token_id=tok.eos_token_id)
                        with torch.no_grad():
                            out_t = model.generate(**enc, **kw)
                            with model.disable_adapter():
                                out_b = model.generate(**enc, **kw)
                        user = ps[j].split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
                        print(f"    gen {j}: prompt ...{user[-80:]!r}", flush=True)
                        print(f"      trained: {tok.decode(out_t[0, n0:])!r}", flush=True)
                        print(f"      base:    {tok.decode(out_b[0, n0:])!r}", flush=True)
                finally:
                    tok.padding_side, tok.truncation_side = old_side, old_trunc
            print(f"{label:4s} {sd.parent.name[-40:]}/{sd.name}: trainer {row['kl_on_trainer']} | "
                  + " | ".join(f"{m} {row[f'kl_{m}']}" for m, _, _ in MODES)
                  + f" | top-1 base/trained off {row['base_top1_off_right']}/{row['trained_top1_off_right']}"
                  + f" | {time.time() - t0:.0f}s", flush=True)
        fh.close()
    summarize(out)


if __name__ == "__main__":
    main()
