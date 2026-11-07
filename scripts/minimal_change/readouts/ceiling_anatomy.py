"""What are the near-ceiling WildChat prompts, and does an input-only pool rule remove them?

The event is "the first response token is one of 38 single-token LDNOOBW words"
(resolver first_response_token_id/case_variants/v1). On the 85,601-prompt
flagged-EN full pool (Qwen3-0.6B scores + per-token log-probs, joined to the
p2300 bank's prompt texts by conversation hash):

  1. for prompts above p = 0.5 / 0.1 / 0.01: which event token carries the mass,
     and whether the prompt itself contains that word (word-boundary match);
  2. a candidate v2 population, "the prompt contains none of the 38 event
     strings" (input-only, defined before any scoring), and on random 2300-pools
     from it vs from the full population: base OLS error, the trivial
     "predict p = 1" error, and the k-matched certified floor (k10, k50; the
     g_transfer_k construction), M250/N2050, 200 splits.

Inputs: the full-pool Qwen3-0.6B scores (wildchat_harmful_scores.csv, per_token_log_p.npz) from
scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh, and the T32_p2300 bank under
$TASK_BANK_ROOT, from scripts/banks/rechunk_token_union_bank.py (T16_p4600 -> T32_p2300).
"""
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g_transfer_k import reach  # noqa: E402

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "src"))
from forecast_experiment.dynamics_probe import certified_tail_targets, involved_sigma_reference  # noqa: E402
from forecast_experiment.ols_utils import count_extrapolated_eval_points  # noqa: E402
from poolscale_readout import split_errors  # noqa: E402

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
TASK_BANK_ROOT = Path(os.environ.get("TASK_BANK_ROOT", REPO_ROOT / "artifacts/task_banks"))
FP = Path(os.environ.get("LOGIT_CACHE_ROOT", RESULTS_ROOT / "wildchat_fullpool_logits")) / "Qwen3-0.6B/harmful"
BANK = TASK_BANK_ROOT / "wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300/bank_seed_11/prompt_rows.jsonl"
M, N, S = 250, 2050, 200


def load():
    with open(FP / "wildchat_harmful_scores.csv") as f:
        rows = list(csv.DictReader(f))
    v = np.array([float(r["log_p_harmful"]) for r in rows])
    hashes = [r["conversation_hash"] for r in rows]
    z = np.load(FP / "per_token_log_p.npz", allow_pickle=True)
    per_tok = z["per_token_log_p"].astype(np.float64)
    decoded = [str(s) for s in z["decoded"]]
    text = {}
    with open(BANK) as f:
        for line in f:
            r = json.loads(line)
            text.setdefault(r.get("h"), r["p"])
    prompts = [text.get(h) for h in hashes]
    return v, per_tok, decoded, prompts


def word_re(words):
    alts = sorted({re.escape(w.lower()) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(alts) + r")(?![a-z0-9])", re.IGNORECASE)


def pool_stats(v, rng, ks=(10, 50)):
    out = {}
    perms = np.stack([rng.permutation(v.size) for _ in range(S)])
    e10, one = split_errors(v, perms, M, k=10)
    out["base_k10"] = float(np.abs(e10).mean())
    out["base_mse_k10"] = float((e10 ** 2).mean())
    out["predict1"] = float(np.abs(one).mean())
    out["top"] = float(v.max())
    out["n_above_0.1"] = int((v > math.log(0.1)).sum())
    for k in ks:
        e = count_extrapolated_eval_points(M, v.size - M, plotting="weibull")
        L = reach(v.size, M, k, e)
        sref = involved_sigma_reference(v, n_fit=M, fit_top_k=k, extrap_k=e, m=L)
        try:
            idx, tgt, info = certified_tail_targets(v, n_fit=M, fit_top_k=k, extrap_k=e, sigma_ref=sref,
                                                    sigma_margin=0.13, two_sided=True, max_rank=L)
        except RuntimeError:
            out[f"cert_k{k}"] = float("nan")
            continue
        c = v.copy()
        c[np.argsort(-v, kind="stable")[L:]] -= float(v[idx[-1]] - tgt[-1])
        c[idx] = tgt
        eb, _ = split_errors(v, perms, M, k=k)
        ec, _ = split_errors(c, perms, M, k=k)
        out[f"base_k{k}"] = float(np.abs(eb).mean())
        out[f"base_mse_k{k}"] = float((eb ** 2).mean())
        out[f"cert_k{k}"] = float(np.abs(ec).mean())
        out[f"cert_mse_k{k}"] = float((ec ** 2).mean())
    return out


def main():
    v, per_tok, decoded, prompts = load()
    have = np.array([p is not None for p in prompts])
    print(f"full pool {v.size}, with bank text {have.sum()}")
    rx_all = word_re(decoded)
    contains_any = np.array([bool(p is not None and rx_all.search(p)) for p in prompts])
    dom = per_tok.argmax(axis=1)
    for thr in (0.5, 0.1, 0.01):
        sel = (v > math.log(thr)) & have
        n = int(sel.sum())
        c = Counter(decoded[i].lower() for i in dom[sel])
        own = np.array([bool(re.search(r"(?<![a-z0-9])" + re.escape(decoded[dom[i]].lower()) + r"(?![a-z0-9])",
                                       prompts[i], re.IGNORECASE)) for i in np.flatnonzero(sel)])
        print(f"\np > {thr}: {n} prompts with text ({int((v > math.log(thr)).sum())} in all); "
              f"prompt contains the dominant word {own.mean():.2f}, contains any event string "
              f"{contains_any[sel].mean():.2f}")
        print("  dominant first token: " + ", ".join(f"{w} {k}" for w, k in c.most_common(12)))
    print(f"\npopulation: contains any event string {contains_any[have].mean():.3f} of prompts with text")
    for thr in (0.5, 0.1, 0.01, 0.001):
        a = (v > math.log(thr)) & have
        print(f"  p > {thr}: {int(a.sum())} prompts, of which {int((a & ~contains_any).sum())} contain no event string")

    rng = np.random.default_rng(0)
    pops = {"all (with text)": np.flatnonzero(have), "no event string in prompt": np.flatnonzero(have & ~contains_any)}
    for name, ids in pops.items():
        rows = []
        for _ in range(16):
            pick = rng.choice(ids, size=M + N, replace=False)
            rows.append(pool_stats(v[pick], rng))
        keys = rows[0].keys()
        m = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        print(f"\n== {name}: {ids.size} prompts, 16 random pools of {M + N}")
        print(f"  pool top {m['top']:+.2f} nats, prompts above 0.1 per pool {m['n_above_0.1']:.2f}")
        print(f"  k10: base {m['base_k10']:.3f}, predict-1 {m['predict1']:.3f}, certified {m['cert_k10']:.3f} "
              f"(mse fold {m['base_mse_k10'] / m['cert_mse_k10']:.2f}x)")
        print(f"  k50: base {m['base_k50']:.3f}, certified {m['cert_k50']:.3f} "
              f"(mse fold {m['base_mse_k50'] / m['cert_mse_k50']:.2f}x)")

    print("\n== the 25 highest-p prompts (first 90 chars, dominant token, p)")
    order = np.argsort(-v)
    shown = 0
    for i in order:
        if prompts[i] is None:
            continue
        snippet = " ".join(prompts[i].split())[:90]
        print(f"  p={math.exp(v[i]):.3f} [{decoded[dom[i]]}] {snippet}")
        shown += 1
        if shown == 25:
            break


if __name__ == "__main__":
    main()
