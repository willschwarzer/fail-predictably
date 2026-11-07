"""Classify every free generation in the think-then-answer probe outputs into one outcome per prompt.

  normal      opened <think>, closed it, and gave a non-empty reply that is not degenerate
  in_think    opened <think>, never closed it, and ended the turn (EOS) inside it: the answer, if any, is in the think block
  empty       closed the think block and then gave no reply
  cap         still inside the think block at the token cap, not looping (long or rambling reasoning)
  loop        the probe's loop flag: the last 64 tokens contain at most 4 distinct ids (at most 2 in our probe of the
              natural-prompt sizes, whose 0.6B outputs carry no loop flag)
  star        the think block is mostly "*" (over half of its first 400 characters)
  degenerate  closed and replied, but the reply is degenerate (few distinct tokens, or mostly "*")
  no_think    never opened <think> (the reply is classified separately: no_think_ok / no_think_degenerate)
Also: forced_ok = replies after the base model's think block that are not degenerate.
The paper's "answered normally" is the normal outcome.
  python scripts/audits/think_then_answer_classify.py DIR [DIR ...]   (writes a table to stdout)
"""
import collections, glob, json, os, sys

MAX_NEW = 3072
rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        recs = [json.loads(x) for x in open(f) if x.strip()]
        if not recs:
            continue
        c = collections.Counter()
        for r in recs:
            star = (r.get("think_star_frac") or 0) > 0.5 or (r.get("think_head", "").count("*") / max(1, len(r.get("think_head", ""))) > 0.5 if r.get("think_head") else False)
            if not r["opened"]:
                c["no_think_degenerate" if r.get("degenerate") else "no_think_ok"] += 1
            elif star:
                c["star"] += 1
            elif r["closed"]:
                if r.get("reply_tokens", 0) == 0:
                    c["empty"] += 1
                elif r.get("degenerate"):
                    c["degenerate"] += 1
                else:
                    c["normal"] += 1
            elif r.get("loop"):
                c["loop"] += 1
            elif r["gen_tokens"] >= MAX_NEW:
                c["cap"] += 1
            else:
                c["in_think"] += 1
        fz = [r for r in recs if r.get("forced_reply_degenerate") is not None]
        c["forced_ok"] = sum(not r["forced_reply_degenerate"] for r in fz)
        rows.append((os.path.basename(d.rstrip("/")), recs[0]["size"], recs[0]["label"], len(recs), len(fz), c))
keys = ["normal", "in_think", "empty", "cap", "loop", "star", "degenerate", "no_think_ok", "no_think_degenerate"]
print(f"{'dir':<30} {'group':<6} {'label':<15} n  " + " ".join(f"{k[:11]:>11}" for k in keys) + "  forced_ok")
for d, g, lab, n, nf, c in rows:
    print(f"{d[-30:]:<30} {g:<6} {lab:<15} {n:<2} " + " ".join(f"{c.get(k, 0):>11}" for k in keys) + f"  {c['forced_ok']}/{nf}")
