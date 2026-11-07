"""Candidate WildChat v2 tasks: event token subsets x input-only pool rules, scored on CPU.

The full-pool cache holds each prompt's first-response-token log-prob for all 38
event ids, so the event log-prob of any id subset is a logsumexp over columns
and needs no GPU. Events:
  E0  all 38 ids (current task)
  E1  drop the ids that are mostly prefixes of benign words as a first token
      (ass/Ass/ASS, xx/XX/xxx/XXX, tit/Tit, cum/Cum, anal/Anal, butt, Dick,
      cock, coon, poon, mong, escort/Escort): 17 ids left
  E2  E1 without the sex family (sex/Sex/SEX/sexual/sexo): 12 ids, profanity
      and porn words only
Pool rules: R0 every prompt with bank text; R1 the prompt contains none of the
event's own strings (word-boundary, case-insensitive), fixed before scoring.
Per cell, on 16 random 2300-pools (M250/N2050, 200 splits): pool top, prompts
above 0.1, base OLS error and MSE, predict-1, certified floor at k10 and k50
(k-matched, as in g_transfer_k), and whether the trivial forecaster beats OLS.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ceiling_anatomy import load, pool_stats, word_re  # noqa: E402

AMBIG = {"ass", "Ass", "ASS", "xx", "XX", "xxx", "XXX", "tit", "Tit", "cum", "Cum", "anal", "Anal",
         "butt", "Dick", "cock", "coon", "poon", "mong", "escort", "Escort"}
SEXFAM = {"sex", "Sex", "SEX", "sexual", "sexo"}


def lse(a, axis=1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True)))[:, 0]


def main():
    v0, per_tok, decoded, prompts = load()
    have = np.array([p is not None for p in prompts])
    events = {
        "E0 all 38": [i for i in range(len(decoded))],
        "E1 no prefix-ambiguous (17)": [i for i, s in enumerate(decoded) if s not in AMBIG],
        "E2 profanity/porn only (12)": [i for i, s in enumerate(decoded) if s not in AMBIG and s not in SEXFAM],
    }
    for ename, cols in events.items():
        v = lse(per_tok[:, cols])
        if ename.startswith("E0"):
            print(f"check: E0 logsumexp vs stored score, max abs diff {np.max(np.abs(v - v0)):.2e}")
        rx = word_re([decoded[i] for i in cols])
        contains = np.array([bool(p is not None and rx.search(p)) for p in prompts])
        print(f"\n######## {ename}: ids {[decoded[i] for i in cols]}")
        for rname, mask in (("R0 all prompts", have), ("R1 no event string in prompt", have & ~contains)):
            ids = np.flatnonzero(mask)
            vv = v[ids]
            counts = " ".join(f">{t}: {int((vv > math.log(t)).sum())}" for t in (0.5, 0.1, 0.01, 0.001))
            rng = np.random.default_rng(0)
            rows = [pool_stats(v[rng.choice(ids, size=2300, replace=False)], rng) for _ in range(16)]
            m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            print(f"  [{rname}] {ids.size} prompts; population {counts}; median log-p {np.median(vv):.1f}")
            print(f"     per pool: top {m['top']:+.2f} nats, above 0.1 {m['n_above_0.1']:.2f} | k10 base {m['base_k10']:.3f}, "
                  f"predict-1 {m['predict1']:.3f}, certified {m['cert_k10']:.3f} (mse fold {m['base_mse_k10'] / m['cert_mse_k10']:.2f}x) | "
                  f"k50 base {m['base_k50']:.3f}, certified {m['cert_k50']:.3f} (mse fold {m['base_mse_k50'] / m['cert_mse_k50']:.2f}x)")
        # What carries the top of R1 under this event?
        top = np.argsort(-np.where(have & ~contains, v, -np.inf))[:8]
        dom = per_tok[:, cols].argmax(axis=1)
        print("  top prompts under R1: " + " || ".join(
            f"p={math.exp(v[i]):.2f} [{decoded[cols[dom[i]]]}] {' '.join(prompts[i].split())[:60]}" for i in top))


if __name__ == "__main__":
    main()
