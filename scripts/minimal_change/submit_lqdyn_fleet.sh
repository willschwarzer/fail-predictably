#!/bin/bash
# Launch the learning-dynamics fleet: one arm per knob, on the 16-task cut of the
# WildChat LDNOOBW bank (12 train / 4 held-out, full batch, every task visited
# every step).
#
#   bash scripts/minimal_change/submit_lqdyn_fleet.sh <seed> [wave]
#     wave = main      the twelve arms on the canonical cell geometry (default)
#     wave = realtail  the three arms on the enlarged pool, where k=10 is a real tail
#     wave = scalenorm the scale-normalized objective, on both cells
#     wave = gapfloor  the one-sided per-gap spread floors, on both cells
#     wave = constraintduel  window floors vs the line-referenced sigma floor,
#                      both with the deterministic constraint-scoring channel
#     wave = constraintduel2 round 2: gate on everywhere, level-gated hinge,
#                      lambda ~2x dual, K matched at 16, + rt sigma with gate
#     wave = safetypolicy  replace the 8-nat gate ramp with a tiny ramp plus an
#                      above-anchor penalty, and/or pin sigma from both sides
#     wave = rtconverge  why does the rt sigma winner plateau at ~3.5x floor?
#                      one knob per arm off the winning recipe: lr (3e-5, 1e-4;
#                      grads are 100% clipped at norm ~80 so lr IS the step
#                      size), steps (280: does the ~step-150 plateau hold),
#                      kl (0.025: is the plateau a KL-forecast equilibrium),
#                      and ols-top-k 50 (deepen the priced-and-certified set;
#                      GAP_FLOOR_M=500 widens the frozen constraint channel so
#                      the realized fit-top-50 stays inside it)
#     wave = dry       print the submissions without sending them
#
# The paper's minimal-change WildChat runs (appendix "Forecastability training with
# minimal change to the model") are poolscale-wave v2 rungs on the v2e2 bank: 24 train /
# 8 held-out pools, k50, 100 steps, seeds 11 17 23 29 43. Each run sets
# BANK_V2=$TASK_BANK_ROOT/wildchat_ldnoobw_v2e2_qwen_qwen3-0_6b_T32_p2300 TRAIN_V2=24 and one
# of the four arms below. The improving-only mask is on in all four; the anchor constraint
# relaxes it.
#   mask (lqdyn_v2e2iogap50)                 V2TAG=v2e2iog REC_OVERRIDE=BOOT=16 RUNGS=v2ap50
#   anchor (lqdyn_v2e2nbap50)                V2TAG=v2e2nb RUNGS=v2ap50
#                                            REC_OVERRIDE=BOOT=16,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0
#   mask + spread (lqdyn_v2e2iogsbap50sp)    V2TAG=v2e2iogsb RUNGS=v2ap50sp
#                     REC_OVERRIDE=BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=1
#   anchor + spread (lqdyn_v2e2ap50sp)       V2TAG=v2e2 RUNGS=v2ap50sp
# e.g. BANK_V2=... TRAIN_V2=24 V2TAG=v2e2 RUNGS=v2ap50sp bash scripts/minimal_change/submit_lqdyn_fleet.sh 11 poolscale
# The one-sided spread arm (lqdyn_v2e2osap50sp) is the anchor + spread arm with
# V2TAG=v2e2os SIGMA_SYM=0. The MLP-adapter arm (lqdyn_v2e2mlpap50sp) is the same arm with
# V2TAG=v2e2mlp and LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj set
# in the submitting shell (its commas cannot go in the --export list; --export=ALL carries it).
# submit_wc_ext200.sh continues the four arms to 200 steps; submit_margin_ablation.sh
# retrains the anchor + spread arm at other margins.
#
# Set SBATCH_ACCOUNT / SBATCH_PARTITION (and SBATCH_QOS) for your cluster; sbatch reads them.
set -euo pipefail

SEED=${1:?usage: submit_lqdyn_fleet.sh <seed> [main|realtail|dry]}
WAVE=${2:-main}
DRY=${DRY:-0}
if [[ "$WAVE" == "dry" ]]; then DRY=1; WAVE=main; fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"   # the jobs write logs/ under the submit directory
TASK_BANK_ROOT="${TASK_BANK_ROOT:-$REPO_ROOT/artifacts/task_banks}"
GEN="$REPO_ROOT/scripts/minimal_change/run_lqdyn_arm.sbatch"
BANK_A="$TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T16_p1870"
BANK_B="$TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T16_p4600"

submit () {
  local arm="$1"; shift
  local exports="ALL,ARM=${arm},SEED=${SEED},$*"
  if (( DRY )); then
    echo "sbatch --export=${exports} $(basename "$GEN")"
  else
    echo -n "[$arm] "
    sbatch --export="${exports}" "$GEN"
  fi
}

if [[ "$WAVE" == "main" ]]; then
  BANK="BANK_ROOT=${BANK_A},TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1"
  # Reference and the bootstrap-permutation ladder.
  submit k1        "BOOT=1,$BANK"
  submit k4        "BOOT=4,$BANK"
  submit k16       "BOOT=16,$BANK"
  submit k64       "BOOT=64,$BANK"
  # Masks: with and without the improving-only constraint, at two K.
  submit k1noiog   "BOOT=1,IOG=0,$BANK"
  submit k16noiog  "BOOT=16,IOG=0,$BANK"
  submit k1deponly "BOOT=1,IOG_SCOPE=deploy_only,$BANK"
  # The fit-side mask's closed form differentiates the score-space loss, but this
  # cell optimizes the log-prob-space one; ~6% of active fit points therefore get
  # the worsening gradient the mask exists to block. 'exact' differentiates the
  # configured loss instead.
  submit k1exactfit "BOOT=1,IOG_FIT_MASK=exact,$BANK"
  submit k16exactfit "BOOT=16,IOG_FIT_MASK=exact,$BANK"
  # The anchor-gated worsening mask: the candidate fix for "abolished" extrapolation.
  submit k1anch2   "BOOT=1,ANCHOR_MARGIN=2,$BANK"
  submit k1anch8   "BOOT=1,ANCHOR_MARGIN=8,$BANK"
  submit k16anch8  "BOOT=16,ANCHOR_MARGIN=8,$BANK"
  # The author's alternative to lifting the constraint: keep the hard mask and add
  # a force that pushes a prompt back down in proportion to how far it has drifted
  # above where it started.
  submit k1anchpen "BOOT=1,ANCHOR_PENALTY=1.0,$BANK"
  # Train on the metric we report.
  submit k1worst   "BOOT=1,WEIGHTING=worst_rank,$BANK"
  # Push the fit top-10 further into the tail without changing the pool.
  submit k1bigfit  "BOOT=1,FIT_FRACTION=0.20,$BANK"
elif [[ "$WAVE" == "realtail" ]]; then
  # Enlarged pool: union 2300 per task, fit 250 / deploy 2050, so the fit top-10
  # sits at the top 4% of the fit sample instead of the top 22%.
  BANK="BANK_ROOT=${BANK_B},TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1,FIT_FRACTION=0.1087,UNION_REFRESH=10"
  submit rtk1      "BOOT=1,$BANK"
  submit rtk16     "BOOT=16,$BANK"
  submit rtk1anch8 "BOOT=1,ANCHOR_MARGIN=8,$BANK"
elif [[ "$WAVE" == "scalenorm" ]]; then
  # The forecast loss is homogeneous of degree two in how far the tail reaches
  # below the bulk, so it can be reduced by pulling the tail in whatever its
  # shape. These arms divide it by that reach, which removes the free descent
  # direction and leaves only the shape. m = 100 covers every rank the
  # extrapolation reads on either cell.
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1,SCALE_NORM_M=100"
  submit k1norm     "BOOT=1,BANK_ROOT=${BANK_A},$COMMON"
  submit k16norm    "BOOT=16,BANK_ROOT=${BANK_A},$COMMON"
  submit rtk1norm   "BOOT=1,BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10,$COMMON"
  submit rtk1anch8norm "BOOT=1,ANCHOR_MARGIN=8,BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10,$COMMON"
elif [[ "$WAVE" == "gapfloor" ]]; then
  # The one-sided spread constraint: every consecutive gap among the union's
  # top-250 order statistics must stay at or above the gaps of the population
  # line through the base pool's tail, enforced by an l1 hinge (exact penalty).
  # The line is the smallest feasible tail, so with lambda above the measured
  # dual bound the constrained optimum is the corrected tail itself. Lambdas
  # default to ~4x the bound the gapfloor estimator measured per cell
  # (lambda* = 2.83 canonical / 5.16 enlarged); override with GAP_LAMBDA /
  # RT_GAP_LAMBDA.
  GAP_LAMBDA=${GAP_LAMBDA:-12}
  RT_GAP_LAMBDA=${RT_GAP_LAMBDA:-20}
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1,GAP_FLOOR_M=250"
  submit k1gap       "BOOT=1,GAP_FLOOR_LAMBDA=${GAP_LAMBDA},BANK_ROOT=${BANK_A},$COMMON"
  submit k1gapanch8  "BOOT=1,GAP_FLOOR_LAMBDA=${GAP_LAMBDA},ANCHOR_MARGIN=8,BANK_ROOT=${BANK_A},$COMMON"
  submit rtk1gap     "BOOT=1,GAP_FLOOR_LAMBDA=${RT_GAP_LAMBDA},BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10,$COMMON"
  submit rtk1gapanch8 "BOOT=1,GAP_FLOOR_LAMBDA=${RT_GAP_LAMBDA},ANCHOR_MARGIN=8,BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10,$COMMON"
elif [[ "$WAVE" == "constraintduel" ]]; then
  # The two candidates that survived the per-gap post-mortem, head to head in
  # real training. Windows: floors on every 16-rank span at the line's spans
  # (lambda 36 from the constraintbench sims, slack ~ the measured grad-pass
  # offset since the constraint channel is now deterministic). Sigma: the
  # line-referenced involved-set floor, K = 16 so the partition noise averages
  # to ~sigma_sd/4 and the margin can stay at 2 of those (0.25 canonical /
  # 0.13 enlarged, from constraintbench's measured partition sds).
  WIN_LAMBDA=${WIN_LAMBDA:-36}
  WIN_SLACK=${WIN_SLACK:-0.02}
  SIG_LAMBDA=${SIG_LAMBDA:-50}
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1"
  WIN="GAP_FLOOR_M=250,GAP_FLOOR_BLOCK=16,GAP_FLOOR_LAMBDA=${WIN_LAMBDA},GAP_FLOOR_SLACK=${WIN_SLACK}"
  RT="FIT_FRACTION=0.1087,UNION_REFRESH=10"
  submit k1win16      "BOOT=1,$WIN,BANK_ROOT=${BANK_A},$COMMON"
  submit k1win16anch8 "BOOT=1,$WIN,ANCHOR_MARGIN=8,BANK_ROOT=${BANK_A},$COMMON"
  submit rtk1win16    "BOOT=1,$WIN,BANK_ROOT=${BANK_B},$RT,$COMMON"
  submit k16sig       "BOOT=16,SIGMA_FLOOR_LAMBDA=${SIG_LAMBDA},SIGMA_FLOOR_MARGIN=0.25,BANK_ROOT=${BANK_A},$COMMON"
  submit k16siganch8  "BOOT=16,SIGMA_FLOOR_LAMBDA=${SIG_LAMBDA},SIGMA_FLOOR_MARGIN=0.25,ANCHOR_MARGIN=8,BANK_ROOT=${BANK_A},$COMMON"
  submit rtk16sig     "BOOT=16,SIGMA_FLOOR_LAMBDA=${SIG_LAMBDA},SIGMA_FLOOR_MARGIN=0.13,BANK_ROOT=${BANK_B},$RT,$COMMON"
elif [[ "$WAVE" == "constraintduel2" ]]; then
  # Round 2, correcting round 1's three diagnosed defects symmetrically:
  # the anchor gate is ON everywhere (the ungated sigma arm's race to the
  # bottom was the hard mask's suppression ratchet, not sigma's fault); the
  # window hinge is LEVEL-GATED (round 1's un-gated upward half was the one
  # strong force outside the improving-only mask and ratcheted pools into
  # the ceiling); lambda drops to ~2x the measured dual bounds (36 was 12x,
  # and the discrete dynamics ratcheted); K = 16 for every arm so both
  # candidates ride the same loss dynamics. Plus the sigma arm round 1
  # never had: rt with the gate.
  W2="GAP_FLOOR_M=250,GAP_FLOOR_BLOCK=16,GAP_FLOOR_SLACK=0.02,GAP_FLOOR_LEVEL_GATED=1"
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1,ANCHOR_MARGIN=8"
  RT="FIT_FRACTION=0.1087,UNION_REFRESH=10"
  submit k16win16b      "BOOT=16,$W2,GAP_FLOOR_LAMBDA=6,BANK_ROOT=${BANK_A},$COMMON"
  submit k1win16b       "BOOT=1,$W2,GAP_FLOOR_LAMBDA=6,BANK_ROOT=${BANK_A},$COMMON"
  submit rtk16win16b    "BOOT=16,$W2,GAP_FLOOR_LAMBDA=10,BANK_ROOT=${BANK_B},$RT,$COMMON"
  submit rtk16siganch8  "BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,BANK_ROOT=${BANK_B},$RT,$COMMON"
  submit k16win16bnolvl "BOOT=16,GAP_FLOOR_M=250,GAP_FLOOR_BLOCK=16,GAP_FLOOR_SLACK=0.02,GAP_FLOOR_LAMBDA=6,BANK_ROOT=${BANK_A},$COMMON"
  submit k16win16bl36   "BOOT=16,$W2,GAP_FLOOR_LAMBDA=36,BANK_ROOT=${BANK_A},$COMMON"
elif [[ "$WAVE" == "safetypolicy" ]]; then
  # Two fixes to the rt sigma winner's safety policy, separately and combined
  # (baseline = rtk16siganch8). Gate: the 8-nat ramp was round 1's throttle on
  # tail compression, a job the sigma floor now does; per-step motion is ~0.1
  # nats, so ANCHOR_MARGIN=0.5 still ramps over ~5 steps while no longer
  # encouraging 8 nats of suppression before upward freedom returns.
  # ANCHOR_PENALTY closes the cross-prompt leak (shared weights raised 2.4% of
  # masked prompts above anchor in rtk1anch8). Sigma: the one-sided floor lets
  # spread grow freely; SIGMA_FLOOR_SYMMETRIC pins it to the anchor line's
  # from both sides (same 0.13 margin as dead zone).
  SIG="BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13"
  RT="BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10"
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,STEPS=200,DUMP_EVERY=1"
  submit rtsig-anchpen    "$SIG,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,$RT,$COMMON"
  submit rtsig-sym        "$SIG,ANCHOR_MARGIN=8,SIGMA_FLOOR_SYMMETRIC=1,$RT,$COMMON"
  submit rtsig-anchpensym "$SIG,ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0,SIGMA_FLOOR_SYMMETRIC=1,$RT,$COMMON"
elif [[ "$WAVE" == "rtconverge" ]]; then
  # Convergence probe on the rt sigma winner: trajectory is hump-then-plateau
  # (plateau from ~step 150 at ~3.5x floor), sigma_bar sits 0.14-0.26 nats ABOVE its floor (constraint
  # not binding), grad_norm ~80 with 100% clipping at 1.0 (lr = the step
  # size), and top-50 points keep moving ~0.03 nats/step at the plateau
  # (motion without metric progress: suspected equilibrium, likely vs the KL
  # term / improving-only mask). One knob per arm.
  SIG="BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,ANCHOR_MARGIN=8"
  RT="BANK_ROOT=${BANK_B},FIT_FRACTION=0.1087,UNION_REFRESH=10"
  COMMON="TRAIN_TARGETS=12,HELDOUT_TARGETS=4,DUMP_EVERY=1"
  submit rtsig-lr3e5   "$SIG,$RT,$COMMON,STEPS=200,LR=3e-5"
  submit rtsig-lr1e4   "$SIG,$RT,$COMMON,STEPS=200,LR=1e-4"
  submit rtsig-s280    "$SIG,$RT,$COMMON,STEPS=280"
  submit rtsig-klhalf  "$SIG,$RT,$COMMON,STEPS=200,KL_COEF=0.025"
  submit rtsig-k50     "$SIG,$RT,$COMMON,STEPS=200,OLS_TOP_K=50,GAP_FLOOR_M=500"
elif [[ "$WAVE" == "poolscale" ]]; then
  # Does held-out transfer grow with the number of training pools? Every
  # anchpen-family arm so far gained ~0.36 dec on its 12 train pools and ~0 on
  # held-out ones, while the certified floor leaves 0.58 +/- 0.13
  # dec of safe headroom on held-out pools. Ladder on the T32_p2300 rechunk of
  # T16_p4600 (same prompts, 32 random 2300-prompt pools, rt geometry 250 fit +
  # 2050 deploy): 24 vs 12 train pools, the SAME 8 held-out pools per seed
  # (HELDOUT_FROM_END), full pools (TARGET_FULL_POOLS; no halving). Recipe =
  # rtsig-anchpensym at lr 2e-5 x 100 steps (same path length as its 1e-5 x 200
  # under 100% clipping). Adapter snapshots every 10 steps feed post-hoc
  # multi-split held-out readouts.
  BANK_C="$TASK_BANK_ROOT/wildchat_ldnoobw_qwen_qwen3-0_6b_T32_p2300"
  # Hardware: one GPU of 48 GB or more per arm (the paper used A100-80GB, L40S and H100
  # cards; the k10 arms also fit an A40). The v2ap50 rung below needs an 80 GB card.
  HV=(-N 1 -c 8)
  # SIGMA_SYM=0 keeps only the anti-collapse floor;
  # pair it with a new V2TAG so the run dirs do not collide.
  # REC_OVERRIDE replaces the recipe for the anchpen/IOG ablation, e.g.
  # REC_OVERRIDE=BOOT=16 is the paper's IOG + KL with no anchor and no band.
  REC=${REC_OVERRIDE:-"BOOT=16,SIGMA_FLOOR_LAMBDA=50,SIGMA_FLOOR_MARGIN=0.13,SIGMA_FLOOR_SYMMETRIC=${SIGMA_SYM:-1},ANCHOR_MARGIN=0.5,ANCHOR_PENALTY=1.0"}
  GEO="BANK_ROOT=${BANK_C},FIT_FRACTION=0.1087,UNION_REFRESH=10,TARGET_FULL_POOLS=1,HELDOUT_FROM_END=1,HELDOUT_TARGETS=8"
  RUN="STEPS=100,LR=2e-5,DUMP_EVERY=2,SNAPSHOT_EVERY=10,SAVE_EVERY=10"
  submit_hv () {
    local arm="$1" ngpu="$2" mem="$3"; shift 3
    local exports="ALL,ARM=${arm},SEED=${SEED},$*"
    local args=("${HV[@]}" --gres=gpu:"$ngpu" --mem="$mem" --job-name="lqps_${arm}_s${SEED}")
    if (( DRY )); then
      echo "sbatch ${args[*]} --export=${exports} $(basename "$GEN")"
    else
      echo -n "[$arm] "
      sbatch "${args[@]}" --export="${exports}" "$GEN"
    fi
  }
  # 2-rank DDP died at torch.cuda.set_device on our cluster ("CUDA-capable device(s)
  # is/are busy or unavailable"), so the 24-pool rung runs on
  # one GPU: ps24b12 samples 12 of its 24 pools per step (the same per-step
  # cost and total target-visits as ps12: more pools at matched compute), and
  # ps24 is full batch at 2x the wall time.
  if [[ ",${RUNGS:-24b12,12}," == *,24b12,* ]]; then
    submit_hv ps24b12 1 32G "$REC,$GEO,$RUN,TRAIN_TARGETS=24,TARGETS_PER_STEP=12"
  fi
  if [[ ",${RUNGS:-24b12,12}," == *,24full,* ]]; then
    submit_hv ps24 1 32G "$REC,$GEO,$RUN,TRAIN_TARGETS=24"
  fi
  if [[ ",${RUNGS:-24b12,12}," == *,12,* ]]; then
    submit_hv ps12 1 32G "$REC,$GEO,$RUN,TRAIN_TARGETS=12"
  fi
  # Certified-tail distillation, the forecast loss switched off: each train
  # pool's anchor top-300 is pulled onto its certified tail (the certified
  # floor; dynamics_probe.certified_tail_targets). Asks whether a model that is
  # told the safe, straight target on its train pools reaches the floor on
  # held-out pools, i.e. whether the missing held-out gain is an objective
  # problem or a transfer problem. Paired with ps24b12 (same pools, compute).
  if [[ ",${RUNGS:-24b12,12}," == *,cd24b12,* ]]; then
    submit_hv cd24b12 1 32G "$REC,$GEO,$RUN,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,CERT_DISTILL_COEF=1,FORECAST_LOSS_COEF=0"
  fi
  # The same pull in event log-odds: a log-prob loss passes a near-ceiling
  # prompt (1 - p) of the gradient, which is where the held-out top-2 drop
  # stalled for anchpen; log-odds pass all of it.
  if [[ ",${RUNGS:-24b12,12}," == *,cdl24b12,* ]]; then
    submit_hv cdl24b12 1 32G "$REC,$GEO,$RUN,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,CERT_DISTILL_COEF=1,FORECAST_LOSS_COEF=0,CERT_DISTILL_SPACE=logit"
  fi
  # k-matched cdl24b12. Certified and read at the same k, the held-out headroom
  # grows with k (own floor 4.6x / 6.7x / 10.6x MSE at k 10 / 20 / 50).
  # At M250 the involved set reaches L = 1.25x the deepest rank any
  # split reads (+5) = 225 / 413 / 789 ranks, so the certificate, the sigma
  # reference, the constraint channel and the union cache all widen to L.
  # GAP_FLOOR_M only sizes the channel here (lambda 0 keeps the gap floor off);
  # a cache below ~k / 0.1087 sends every permutation to the fit fallback.
  CDL="CERT_DISTILL_COEF=1,FORECAST_LOSS_COEF=0,CERT_DISTILL_SPACE=logit"
  if [[ ",${RUNGS:-24b12,12}," == *,cdl20b12,* ]]; then
    submit_hv cdl20b12 1 40G "$REC,$GEO,$RUN,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,$CDL,OLS_TOP_K=20,SIGMA_REF_M=413,CERT_DISTILL_RANKS=413,GAP_FLOOR_M=413,UNION_CACHE_C=413"
  fi
  # At k50 the 889-prompt channel plus the bootstrap's own grad pass overflow a
  # 46 GB L40S (the v2e2cdl50 arm ran out of memory at step 1), so the k50 distillation
  # rungs read the anchor penalty from the channel pass and skip the double
  # scoring (CERT_DISTILL_SHARED_PASS; the "sp" in the arm name).
  SP="CERT_DISTILL_SHARED_PASS=1"
  if [[ ",${RUNGS:-24b12,12}," == *,cdl50b12,* ]]; then
    submit_hv cdl50b12sp 1 40G "$REC,$GEO,$RUN,TRAIN_TARGETS=24,TARGETS_PER_STEP=12,$CDL,$SP,OLS_TOP_K=50,SIGMA_REF_M=789,CERT_DISTILL_RANKS=789,GAP_FLOOR_M=789,UNION_CACHE_C=789"
  fi
  # WildChat v2: the v1 event counts benign
  # answers that open with "Sex..." or a subword like "Ass" (Assalam), and those
  # prompts carry the ceiling. BANK_V2 is built by scripts/banks/derive_wildchat_v2_bank.py
  # (event-id subset + input-only prompt rule, 2300-prompt pools); every pool
  # not held out trains, 12 sampled per step, 8 held out from the end. V2TAG
  # names the arms after the bank (v2e2 = the 12 profanity/porn ids, no rule).
  BANK_V2=${BANK_V2:-}
  V2TAG=${V2TAG:-v2}
  # HELDOUT_V2 and L50 resize the cell for other pool sizes: the T16_p4600 bank
  # (16 x 4600, M500/N4100 at the same fit fraction) runs TRAIN_V2=12
  # HELDOUT_V2=4 L50=779 (the k50 reach at M500).
  HELDOUT_V2=${HELDOUT_V2:-8}
  L50=${L50:-789}
  # RUNGS token v2 = v2ap + v2cdl + v2cdl50; each also selects alone. v2ap50 is
  # the forecast loss itself at k50, with the cache sized so the k50 fit top-50
  # is scored fresh every step (the old k50 arm's 296 left it stale).
  K50="OLS_TOP_K=50,SIGMA_REF_M=${L50},GAP_FLOOR_M=${L50},UNION_CACHE_C=${L50}"
  v2sel () { [[ ",${RUNGS:-24b12,12}," == *,"$1",* || ( "$2" == base && ",${RUNGS:-24b12,12}," == *,v2,* ) ]]; }
  if v2sel v2ap base || v2sel v2cdl base || v2sel v2cdl50 base || v2sel v2ap50 extra || v2sel v2ap50sp extra; then
    [[ -n "$BANK_V2" && -n "${TRAIN_V2:-}" ]] || { echo "v2 rungs need BANK_V2 and TRAIN_V2" >&2; exit 1; }
    GEO2="BANK_ROOT=${BANK_V2},FIT_FRACTION=0.1087,UNION_REFRESH=10,TARGET_FULL_POOLS=1,HELDOUT_FROM_END=1,HELDOUT_TARGETS=${HELDOUT_V2}"
  fi
  if v2sel v2ap base; then
    submit_hv "${V2TAG}ap" 1 32G "$REC,$GEO2,$RUN,TRAIN_TARGETS=${TRAIN_V2},TARGETS_PER_STEP=12"
  fi
  if v2sel v2cdl base; then
    submit_hv "${V2TAG}cdl" 1 32G "$REC,$GEO2,$RUN,TRAIN_TARGETS=${TRAIN_V2},TARGETS_PER_STEP=12,$CDL"
  fi
  if v2sel v2cdl50 base; then
    submit_hv "${V2TAG}cdl50sp" 1 40G "$REC,$GEO2,$RUN,TRAIN_TARGETS=${TRAIN_V2},TARGETS_PER_STEP=12,$CDL,$SP,$K50,CERT_DISTILL_RANKS=${L50}"
  fi
  # v2ap50sp is v2ap50 on the shared pass (the forecast loss reads the
  # channel's dropout-off scores for channel prompts), which fits a 46 GB L40S.
  if v2sel v2ap50sp extra; then
    submit_hv "${V2TAG}ap50sp" 1 40G "$REC,$GEO2,$RUN,TRAIN_TARGETS=${TRAIN_V2},TARGETS_PER_STEP=12,$K50,$SP"
  fi
  if v2sel v2ap50 extra; then
    # Hardware: the forecast loss at k50 with both scoring passes does not fit a
    # 46 GB L40S, so this rung needs an 80 GB GPU (A100-80GB or H100). Recipes with
    # no constraint channel (no sigma floor) score once per step and fit the smaller card.
    submit_hv "${V2TAG}ap50" 1 40G "$REC,$GEO2,$RUN,TRAIN_TARGETS=${TRAIN_V2},TARGETS_PER_STEP=12,$K50"
  fi
else
  echo "unknown wave: $WAVE" >&2
  exit 1
fi
