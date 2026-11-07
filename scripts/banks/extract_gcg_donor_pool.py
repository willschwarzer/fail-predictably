"""Extract a warm-start GCG donor pool from an existing password bank.

Scores every stored GCG suffix (RE-ENCODED text) across all slots under the
bank's policy, and writes the strongest N as a donor pool for warm-start bank
rebuilds (LQUANT_GCG_DONOR_POOL_JSON). Storing donor TEXT (re-encoded to init
ids at build time) keeps it tokenizer-robust. One GPU; run it with
scripts/banks/run_extract_gcg_donor_pool.sbatch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from forecast_experiment.password_game import build_password_system_prompt
from forecast_experiment.password_gcg import _apply_chat_template_ids


def _load(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    return tok, model


def _leak(model, tok, sys_prompt, user_text, event_id) -> float:
    ids = _apply_chat_template_ids(
        tok,
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}],
        enable_thinking=False,
    )
    t = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        lp = F.log_softmax(model(input_ids=t).logits[:, -1, :], dim=-1)
    return float(lp[0, event_id].item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--bank", required=True, help="bank_seed_NN dir")
    ap.add_argument("--out", required=True, help="output donor_pool.json path")
    ap.add_argument("--pool-size", type=int, default=12)
    args = ap.parse_args()

    tok, model = _load(args.model)
    b = Path(args.bank)
    m = json.loads((b / "metadata.json").read_text())
    policy = m["password_policy"]
    gidx = m["families"].index("gcg_steps80")
    s2p = m["slot_to_password"]
    rows = [json.loads(l) for l in (b / "prompt_rows.jsonl").read_text().splitlines()]

    scored = []
    for r in rows:
        if r.get("f") != gidx:
            continue
        slot = f"password_slot_{int(r['t']):02d}"
        pw = s2p[slot]
        eid = tok(pw, add_special_tokens=False).input_ids[0]
        sysp = build_password_system_prompt(pw, policy=policy)
        leak = math.exp(_leak(model, tok, sysp, r["p"], eid))
        scored.append({"slot": int(r["t"]), "password": pw, "text": r["p"], "reencoded_leak": leak})

    scored.sort(key=lambda d: d["reencoded_leak"], reverse=True)
    donors = scored[: args.pool_size]
    out = {
        "model_id": m["model_id"],
        "password_policy": policy,
        "source_bank": str(b),
        "pool_size": len(donors),
        "donors": donors,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[donor-pool] wrote {len(donors)} donors -> {args.out}", flush=True)
    for d in donors:
        print(f"  slot {d['slot']:2d} pw={d['password']!r:12s} leak={d['reencoded_leak']:.4g}", flush=True)


if __name__ == "__main__":
    main()
