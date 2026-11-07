"""Held-out headroom on the v2e2 bank (12 profanity/porn first-token ids, unfiltered pools).

The 32 pools of the v2e2 bank are the p2300 bank's pools; each prompt's E2 event
log-prob comes from the full-pool per-token cache (joined by conversation hash),
so no GPU is needed. Over 20 random 24/8 partitions of the 32 pools, at k 10 / 20
/ 50 (M250/N2050, k-matched certificates as in g_transfer_k): base, each pool's
own certified floor, g fit on the 24 train pools, and a uniform shift. Also the
smoke check that the bank's per-target event ids resolve and that the recorded
38-id union still verifies under the tokenizer.

Inputs: the full-pool Qwen3-0.6B scores (wildchat_harmful_scores.csv, per_token_log_p.npz) from
scripts/wildchat/pretrained_scale/submit_wildchat_fullpool_logits.sh, and the v2e2 bank under
$TASK_BANK_ROOT, built from the T32_p2300 bank by scripts/banks/derive_wildchat_v2_bank.py.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from event_v2 import AMBIG, SEXFAM, lse  # noqa: E402
import g_transfer_k as gk  # noqa: E402

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO_ROOT / "src"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
TASK_BANK_ROOT = Path(os.environ.get("TASK_BANK_ROOT", REPO_ROOT / "artifacts/task_banks"))
FP = Path(os.environ.get("LOGIT_CACHE_ROOT", RESULTS_ROOT / "wildchat_fullpool_logits")) / "Qwen3-0.6B/harmful"
BANK = TASK_BANK_ROOT / "wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300/bank_seed_11"


def smoke():
    from transformers import AutoTokenizer
    from forecast_experiment.wildchat_token_union import resolve_target_event_token_ids, verify_token_union
    meta = json.loads((BANK / "metadata.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model_id"])
    verify_token_union(tok, meta)
    ids = resolve_target_event_token_ids(meta, list(meta["targets"]))
    sets = {tuple(v) for v in ids.values()}
    assert len(sets) == 1, "targets disagree on the event id set"
    words = [tok.decode([i]) for i in next(iter(sets))]
    print(f"smoke: union verifies; {len(ids)} targets share {len(words)} event ids: {words}")
    return meta


def pools_from_bank(meta):
    with open(FP / "wildchat_harmful_scores.csv") as f:
        hashes = [r["conversation_hash"] for r in csv.DictReader(f)]
    z = np.load(FP / "per_token_log_p.npz", allow_pickle=True)
    decoded = [str(s) for s in z["decoded"]]
    cols = [i for i, s in enumerate(decoded) if s not in AMBIG and s not in SEXFAM]
    v = lse(z["per_token_log_p"].astype(np.float64)[:, cols])
    row_of = {}
    for i, h in enumerate(hashes):
        row_of.setdefault(h, i)
    pools = {t: [] for t in meta["targets"]}
    missing = 0
    with open(BANK / "prompt_rows.jsonl") as f:
        for line in f:
            r = json.loads(line)
            i = row_of.get(r.get("h"))
            if i is None:
                missing += 1
                continue
            pools[meta["targets"][int(r["t"])]].append(v[i])
    print(f"joined bank rows to cache: {missing} rows without a cache match")
    return {t: np.asarray(x) for t, x in pools.items()}


def main():
    meta = smoke()
    pools = pools_from_bank(meta)
    sizes = {x.size for x in pools.values()}
    print(f"{len(pools)} pools, sizes {sorted(sizes)}; mean pool top {np.mean([x.max() for x in pools.values()]):+.2f} nats")
    names = [n for n, x in pools.items() if x.size == 2300]
    for M, k in [(250, 10), (250, 20), (250, 50)]:
        rng = np.random.default_rng(0)
        parts = []
        for _ in range(20):
            perm = rng.permutation(len(names))
            tr, te = [names[i] for i in perm[:24]], [names[i] for i in perm[24:]]
            g, bulk = gk.fit_g([(pools[n], gk.cert(n, pools[n], M, k)[0]) for n in tr])
            m = []
            for j, n in enumerate(te):
                v = pools[n]
                c = gk.cert(n, v, M, k)[0]
                m.append([gk.score(x, M, k, 7 + j) for x in (v, c, g(v), v - bulk)])
            parts.append(np.mean(np.array(m), axis=0))
        p = np.array(parts)
        print(f"\nv2e2 M{M} k{k}: 20 random 24/8 partitions of {len(names)} pools")
        for j, lab in enumerate(["base", "own certified", "g from 24", "uniform shift"]):
            fold = p[:, 0, 1] / p[:, j, 1]
            print(f"  {lab:14s} abs {p[:, j, 0].mean():.3f}  mse {p[:, j, 1].mean():.3f}  "
                  f"mse fold {fold.mean():.2f}x (partition range {fold.min():.2f}-{fold.max():.2f})")


if __name__ == "__main__":
    main()
