"""Do the natural-prompt checkpoints still answer normally when they think first?

With thinking off (the template the risk scores use), every selected natural-prompt checkpoint puts
probability 1.000 on one first token ("**" at 0.6B), and its greedy replies are a string of "*"
(kl_thinking_off_final.py). With thinking on, their first generated token is still "<think>".
This probe lets each model think and then answer, with thinking on, on held-out ordinary prompts:

  free    the model generates with Qwen3's thinking template, sampling at Qwen3's recommended thinking
          settings (temperature 0.6, top-p 0.95, top-k 20). We record whether it opens and closes a think
          block, how long it thinks, the first reply token, how varied the reply is (distinct tokens among
          its first 100; "degenerate" = under 0.2 with at least 20 reply tokens, or over half of its first
          400 characters "*"), and the text.
  forced  every model gets the base model's own think block (prompt + "<think>...</think>" from the base
          model's free generation + "\\n\\n"). At the reply's first position we record each model's top
          tokens, the probability of "**", and KL(model || base); then the model greedy-decodes 60 reply
          tokens from there.
  off     for reference, the top-1 first token with thinking off (empty think block), the scored position.

Models per size: the base model (adapter disabled) and the seed-11 selected ("best") adapters of Ours and
SFT, under $RESULTS_ROOT. Prompts: the first N of the runs' held-out KL prompts (kl_eval_datasets, loaded as
the trainer loads them) that are at most 512 tokens long. Read-only on the run directories. One JSONL per
model in --out-dir, written when that model finishes; a requeued job skips models whose JSONL is complete.
summary.csv is rebuilt from every JSONL in --out-dir at the end. think_then_answer_classify.py sorts each
generation into one outcome. The paper reports these outcomes in the appendices "Experimental details"
(paragraph "KL divergence from the base model"), "Forecastability on natural user prompts" and
"Forecastability training with minimal change to the model".

  python scripts/audits/think_then_answer_probe.py --size 0.6b|1.7b|4b --out-dir DIR [--n 16] [--max-new 3072] [--bs B]
  python scripts/audits/think_then_answer_probe.py --group NAME --model LABEL=CONFIG_JSON:ADAPTER [--model ...] --out-dir DIR

The --group form (for the password-game and minimal-change checkpoints) probes any set of checkpoints that
share one base model, LoRA settings and held-out KL prompt set; files are {NAME}_{LABEL}.jsonl.
scripts/audits/submit_think_then_answer.sh builds the paper's groups. --dry-run checks the paths and configs
and lists what would run, without loading a model.
"""
import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kl_thinking_off as kto  # noqa: E402  (its prompt and logit functions)
from kl_thinking_off_final import cfg_of  # noqa: E402
from importance_sampling.policy_lm import LoRASetup, PolicyLM  # noqa: E402

RES = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = {
    "0.6b": ("lquant_wildchat_ldnoobw_qwen_qwen3-0_6b_iog_full_boot1_cache296_seed11_mb8klnf05/seed_11",
             "lquant_wildchat_ldnoobw_sft_union_qwen_qwen3-0_6b_klnf05_seed11/seed_11"),
    "1.7b": ("lquant_wildchat_ldnoobw_qwen_qwen3-1_7b_iog_full_boot1_cache296_seed11_mb8klnf05/seed_11",
             "lquant_wildchat_ldnoobw_sft_union_qwen_qwen3-1_7b_klnf05_seed11/seed_11"),
    "4b": ("lquant_wildchat_ldnoobw_qwen_qwen3-4b_iog_full_boot1_cache296_seed11_mb8klnf05/seed_11",
           "lquant_wildchat_ldnoobw_sft_union_qwen_qwen3-4b_klnf05_seed11/seed_11"),
}
DEFAULT_BS = {"0.6b": 16, "1.7b": 16, "4b": 8}
SAMPLE = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
# A loop: the last LOOP_WINDOW generated tokens contain at most LOOP_MAX_IDS distinct ids. Our probe of the
# natural-prompt sizes used at most 2. The value 4 also catches loops such as "****  \n**** ****", and the
# password-game and minimal-change groups ran with it.
LOOP_WINDOW, LOOP_MAX_IDS = 64, 4


def user_text(p):
    return p.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]


class EarlyStop(StoppingCriteria):
    """Per-sequence stop for the free generation: a loop (the last LOOP_WINDOW new tokens contain at most
    LOOP_MAX_IDS distinct ids), 160 reply tokens after </think>, or 200 tokens of a reply that never opened a
    think block. Our probe of the 0.6B natural-prompt checkpoints ran without it and generated to max_new."""

    def __init__(self, n0, think_id, end_id, window=LOOP_WINDOW, reply=160, direct=200, max_ids=LOOP_MAX_IDS):
        self.n0, self.think_id, self.end_id = n0, think_id, end_id
        self.window, self.reply, self.direct, self.max_ids = window, reply, direct, max_ids

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[:, self.n0:]
        n = gen.shape[1]
        done = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
        if n == 0:
            return done
        if n >= self.window:
            srt = gen[:, -self.window:].sort(dim=1).values
            done |= (1 + (srt[:, 1:] != srt[:, :-1]).sum(1)) <= self.max_ids
        has = gen == self.end_id
        first = has.int().argmax(1)
        done |= has.any(1) & ((n - 1 - first) >= self.reply)
        done |= (gen[:, 0] != self.think_id) & (n >= self.direct)
        return done


def generate(model, tok, texts, max_new, sample, seed, bs, device, eos_ids, stop_ids=None):
    """Left-padded batched generation; returns each row's new token ids, cut at the first EOS or pad."""
    outs = []
    old = tok.padding_side
    tok.padding_side = "left"
    try:
        for s in range(0, len(texts), bs):
            enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(device)
            n0 = enc["input_ids"].shape[1]
            kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=eos_ids, use_cache=True)
            kw.update(SAMPLE if sample else dict(do_sample=False))
            if stop_ids is not None:
                kw["stopping_criteria"] = StoppingCriteriaList([EarlyStop(n0, *stop_ids)])
            torch.manual_seed(seed + s)
            with torch.no_grad():
                g = model.generate(**enc, **kw)
            for row in g[:, n0:].tolist():
                cut = next((k for k, t in enumerate(row) if t in eos_ids or t == tok.pad_token_id), len(row))
                outs.append(row[:cut])
    finally:
        tok.padding_side = old
    return outs


def reply_stats(tok, ids):
    if not ids:
        return dict(reply_first="", distinct100=None, star_frac=None, degenerate=None, reply_text="")
    text = tok.decode(ids, skip_special_tokens=False)
    head = ids[:100]
    distinct = len(set(head)) / len(head)
    t400 = text[:400]
    star = t400.count("*") / max(1, len(t400))
    degenerate = (distinct < 0.2 and len(ids) >= 20) or star > 0.5
    return dict(reply_first=tok.decode(ids[:1]), distinct100=round(distinct, 3), star_frac=round(star, 3),
                degenerate=bool(degenerate), reply_text=text[:600])


def parse(tok, ids, think_id, end_id):
    opened = bool(ids) and ids[0] == think_id
    closed = opened and end_id in ids
    rec = dict(opened=opened, closed=closed, gen_tokens=len(ids))
    if opened and closed:
        p = ids.index(end_id)
        rec["think_tokens"] = p - 1
        think = tok.decode(ids[1:p], skip_special_tokens=False)
        rec["think_head"], rec["think_tail"] = think[:400], think[-300:]
        rec["think_block"] = tok.decode(ids[:p + 1], skip_special_tokens=False)
        reply = ids[p + 1:]
    elif opened:
        rec["think_tokens"] = len(ids) - 1
        think = tok.decode(ids[1:], skip_special_tokens=False)
        rec["think_head"], rec["think_tail"] = think[:400], think[-300:]
        reply = []
    else:
        rec["think_tokens"] = 0
        reply = ids
    rec["loop"] = len(ids) >= LOOP_WINDOW and len(set(ids[-LOOP_WINDOW:])) <= LOOP_MAX_IDS
    rec["loop_rule"] = f"last {LOOP_WINDOW} tokens, at most {LOOP_MAX_IDS} distinct ids"
    th = rec.get("think_head", "")
    rec["think_star_frac"] = round(th.count("*") / max(1, len(th)), 3) if th else None
    k = 0
    while k < len(reply) and tok.decode(reply[k:k + 1]).strip() == "":
        k += 1
    reply = reply[k:]
    rec["reply_tokens"] = len(reply)
    rec.update(reply_stats(tok, reply))
    return rec


def top_tokens(tok, lp, n=3):
    v, i = lp.exp().topk(n)
    return [[tok.decode([int(t)]), round(float(p), 4)] for p, t in zip(v, i)]


def summarize(out_dir):
    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not recs:
            continue
        size, label = recs[0]["size"], recs[0]["label"]
        op = [r for r in recs if r["opened"]]
        cl = [r for r in op if r["closed"]]
        star = (lambda t: t.count("*") / max(1, len(t)))
        tdeg = [r for r in op if (r.get("think_head") and star(r["think_head"]) > 0.5)
                or (not r["closed"] and r.get("loop"))]
        rep = [r for r in recs if r["degenerate"] is not None]
        fz = [r for r in recs if r.get("forced_kl_vs_base") is not None]
        fg = [r for r in fz if r.get("forced_reply_degenerate") is not None]
        mean = (lambda xs: round(statistics.mean(xs), 4) if xs else "")
        rows.append(dict(
            size=size, label=label, n_prompts=len(recs),
            opened_think=f"{len(op)}/{len(recs)}", closed_think=f"{len(cl)}/{len(op)}",
            degenerate_think=f"{len(tdeg)}/{len(op)}",
            median_think_tokens=statistics.median([r["think_tokens"] for r in cl]) if cl else "",
            replies=len(rep), degenerate_replies=sum(r["degenerate"] for r in rep),
            mean_distinct100=mean([r["distinct100"] for r in rep]),
            reply_first_top=" ".join(f"{t!r}x{c}" for t, c in
                                     collections.Counter(r["reply_first"] for r in rep).most_common(3)),
            forced_n=len(fz), forced_mean_kl=mean([r["forced_kl_vs_base"] for r in fz]),
            forced_top1_starstar=sum(r["forced_top"][0][0] == "**" for r in fz),
            forced_mean_p_starstar=mean([r["forced_p_starstar"] for r in fz]),
            forced_reply_degenerate=sum(r["forced_reply_degenerate"] for r in fg),
            off_top1=" ".join(f"{t!r}x{c}" for t, c in
                              collections.Counter(r["off_top1"][0] for r in recs).most_common(2)),
        ))
    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(" | ".join(f"{k} {v}" for k, v in r.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(RUNS), default=None,
                    help="a natural-prompt size: base, and the seed-11 best adapters of Ours and SFT")
    ap.add_argument("--group", default=None,
                    help="instead of --size: a name for the --model checkpoints (the output file prefix)")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=CONFIG_JSON:ADAPTER",
                    help="with --group, repeatable: a trained checkpoint, given by its run's config.json and its "
                         "adapter_model.safetensors. All must share model, LoRA settings and held-out KL prompts")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", default=None,
                    help="comma-separated labels: run just these (and reuse the group's finished base JSONL), so that "
                         "several jobs can share one group; each subset has its own claim file")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if (a.size is None) == (a.group is None) or (a.group is not None and not a.model):
        ap.error("give --size, or --group with at least one --model")
    tag = a.size or a.group
    only = sorted(set(a.only.split(","))) if a.only else None
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Two jobs may reach the same size or group. The first to start it writes claim_{tag}.txt with its Slurm
    # job id, and a later job skips it while that job is still running.
    me = os.environ.get("SLURM_JOB_ID", "")
    claim = out_dir / (f"claim_{tag}__{'+'.join(only)}.txt" if only else f"claim_{tag}.txt")
    holder = claim.read_text().strip() if claim.exists() else ""
    if me and holder and holder != me:
        try:
            q = subprocess.run(["squeue", "-h", "-j", holder, "-o", "%T"], capture_output=True, text=True, timeout=60)
            state = q.stdout.strip() if q.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            state = ""   # cannot tell: run the size rather than leave it to nobody
        if state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            print(f"size {tag}: skipped in job {me} (job {holder} holds {claim.name})", flush=True)
            return
    if me:
        claim.write_text(me + "\n")
    bs = a.bs or DEFAULT_BS.get(a.size, 16)
    if (a.bs is None and a.size == "4b" and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory >= 20 * 2**30):
        bs = 16   # one batch instead of two on a 24 GB card
    # models: (label, config.json, adapter file); the base model is the same LoRA model with the adapter disabled
    if a.size:
        ours, sft = (RES / r for r in RUNS[a.size])
        models = [("base", None, None), ("Ours", ours / "config.json", ours / "best" / "adapter_model.safetensors"),
                  ("SFT", sft / "config.json", sft / "best" / "adapter_model.safetensors")]
    else:
        models = [("base", None, None)]
        for spec in a.model:
            lab, _, paths = spec.partition("=")
            cj, _, ad = paths.partition(":")
            if not lab or lab == "base" or not cj or not ad or any(lab == m[0] for m in models):
                raise SystemExit(f"bad or repeated --model {spec!r}")
            models.append((lab, Path(cj), Path(ad)))
    if only:
        unknown = set(only) - {m[0] for m in models}
        if unknown:
            raise SystemExit(f"--only names labels not in the group: {sorted(unknown)}")
        models = [m for m in models if m[0] == "base" or m[0] in only]
    raws = {lab: json.loads(cj.read_text()) for lab, cj, _ in models if cj is not None}
    share = kto.LORA_KEYS + ("kl_eval_datasets",)
    lora_of = {lab: tuple(json.dumps(r.get(k), sort_keys=True) for k in share) for lab, r in raws.items()}
    if len(set(lora_of.values())) != 1:
        raise RuntimeError(f"{tag}: the checkpoints differ in model, LoRA settings or held-out KL prompts")
    for lab, _, ad in models:
        if ad is not None and not ad.exists():
            raise RuntimeError(f"{lab}: no adapter at {ad}")

    def jsonl(lab):
        return out_dir / f"{tag}_{lab}.jsonl"

    def done(lab):
        f = jsonl(lab)
        return f.exists() and len([x for x in f.read_text().splitlines() if x.strip()]) == a.n

    todo = [lab for lab, _, _ in models if not done(lab)]
    print(f"size {tag}: models to run {todo or 'none'} (n {a.n}, max_new {a.max_new}, bs {bs})", flush=True)
    if a.dry_run:
        for lab, cj, ad in models:
            print(f"  {lab}: {cj} | {ad}", flush=True)
        return
    if todo:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        cfg = cfg_of(raws[models[1][0]])
        policy = PolicyLM(model_id=cfg.model_id, device=cfg.device,
                          lora=LoRASetup(use_lora=cfg.use_lora, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                         dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules))
        model, tok = policy.model, policy.tokenizer
        model.eval()
        print(f"model {cfg.model_id}, dtype {next(model.parameters()).dtype}", flush=True)
        think_id, end_id = tok.convert_tokens_to_ids("<think>"), tok.convert_tokens_to_ids("</think>")
        eos_ids = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
        star_ids = tok.encode("**", add_special_tokens=False)
        print(f"<think> {think_id}, </think> {end_id}, eos {eos_ids}, '**' -> {star_ids}", flush=True)
        on_all = kto.prompts_for(cfg, tok, False)
        off_all = kto.prompts_for(cfg, tok, True)
        if len(on_all) != len(off_all):
            raise RuntimeError("thinking-on and thinking-off prompt sets differ in length")
        idx = [i for i, p in enumerate(on_all) if len(tok(p).input_ids) <= 512][:a.n]
        if len(idx) < a.n:
            raise RuntimeError(f"only {len(idx)} prompts of at most 512 tokens")
        for i in idx:
            if user_text(on_all[i]) != user_text(off_all[i]):
                raise RuntimeError(f"prompt {i}: thinking-on and thinking-off sets are not aligned")
        on = [on_all[i] for i in idx]
        off = [off_all[i] for i in idx]
        print(f"prompts {idx}; prompt 0 ends {on[0][-40:]!r} (on) / {off[0][-40:]!r} (off)", flush=True)
        tcfg = SimpleNamespace(kl_batch_size=max(1, bs // 4), device=cfg.device, kl_tokenize_max_len=a.max_new + 1024,
                               kl_truncation_side="left")

        base_recs = None
        if done("base"):
            base_recs = [json.loads(x) for x in jsonl("base").read_text().splitlines() if x.strip()]
        lq_forced, ctx_pos, ctx = None, None, None

        def forced_contexts(recs):
            pos = [j for j, r in enumerate(recs) if r.get("think_block")]
            return pos, [on[j] + recs[j]["think_block"] + "\n\n" for j in pos]

        for lab, _, ad in models:
            if lab not in todo:
                continue
            t0 = time.time()
            if ad is None:
                model_ctx = model.disable_adapter
            else:
                f = ad
                before = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
                res = set_peft_model_state_dict(model, load_file(str(f)))
                if list(getattr(res, "unexpected_keys", []) or []):
                    raise RuntimeError(f"unexpected adapter keys in {f}")
                if [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]:
                    raise RuntimeError(f"adapter {f} lacks LoRA weights")
                after = sum(float(p.detach().float().norm()) for p in model.parameters() if p.requires_grad)
                if abs(after - before) < 1e-8:
                    raise RuntimeError(f"adapter {f} left the trainable weights unchanged")
                model.eval()
                model_ctx = torch.no_grad
            with model_ctx():
                gens = generate(model, tok, on, a.max_new, True, a.seed, bs, cfg.device, eos_ids,
                                stop_ids=(think_id, end_id))
                recs = [dict(size=tag, label=lab, i=i, user_head=user_text(on[j])[:160], **parse(tok, g, think_id, end_id))
                        for j, (i, g) in enumerate(zip(idx, gens))]
                print(f"  {lab}: free generation done ({time.time() - t0:.0f}s)", flush=True)
                if lab == "base":
                    base_recs = recs
                if ctx is None:
                    ctx_pos, ctx = forced_contexts(base_recs)
                lo = kto.last_logprobs(model, tok, off, cfg)
                lp = kto.last_logprobs(model, tok, ctx, tcfg) if ctx else None
                fg = generate(model, tok, ctx, 60, False, a.seed, max(1, bs // 2), cfg.device, eos_ids) if ctx else []
            if lp is not None and lq_forced is None:
                if lab == "base":
                    lq_forced = lp
                else:
                    with model.disable_adapter():
                        lq_forced = kto.last_logprobs(model, tok, ctx, tcfg)
            for j, r in enumerate(recs):
                r["off_top1"] = top_tokens(tok, lo[j], 1)[0]
            for k, j in enumerate(ctx_pos or []):
                r = recs[j]
                kl = float((lp[k].exp() * (lp[k] - lq_forced[k])).sum())
                r["forced_kl_vs_base"] = round(kl, 5)
                r["forced_top"] = top_tokens(tok, lp[k], 3)
                r["forced_p_starstar"] = round(float(lp[k][star_ids[0]].exp()), 5)
                st = reply_stats(tok, fg[k])
                r["forced_reply"] = st["reply_text"][:300]
                r["forced_reply_distinct100"] = st["distinct100"]
                r["forced_reply_degenerate"] = st["degenerate"]
            tmp = jsonl(lab).with_suffix(".jsonl.tmp")
            with open(tmp, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            os.replace(tmp, jsonl(lab))
            n_deg = sum(bool(r["degenerate"]) for r in recs)
            n_tdeg = sum(1 for r in recs if r["opened"] and ((r.get("think_star_frac") or 0) > 0.5
                                                          or (not r["closed"] and r["loop"])))
            print(f"  {lab}: opened {sum(r['opened'] for r in recs)}/{len(recs)}, closed "
                  f"{sum(r['closed'] for r in recs)}, think degenerate {n_tdeg}, degenerate replies {n_deg}, "
                  f"forced degenerate {sum(bool(r.get('forced_reply_degenerate')) for r in recs)}, "
                  f"forced contexts {len(ctx_pos or [])} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            for r in recs[:3]:
                print(f"    [{r['i']}] {r['user_head'][:70]!r}\n      think {r['think_tokens']} tok; reply "
                      f"{r['reply_text'][:160]!r}\n      forced: {r.get('forced_top')} | {r.get('forced_reply', '')[:120]!r}",
                      flush=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
