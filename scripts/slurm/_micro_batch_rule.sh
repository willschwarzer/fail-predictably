# Model-aware micro-batch rule (sourced; sets MICRO_BATCH).
#
# WHY THIS EXISTS. The 0.6B launchers pick micro-batch from a VRAM-only ladder
# (<30 GB -> 16, <50 -> 48, <90 -> 96, else 128). That ladder is calibrated for a
# 0.6B policy and is model-blind, so it OOMs larger models: the policy plus the
# frozen KL base occupy 2 x params x 2 bytes of the card before a single
# activation is allocated, and per-sample activation cost grows with
# hidden_size x n_layers. Replacing the ladder with a hardcoded constant
# (MICRO_BATCH=16 for 4B and 8B) failed the same way for the opposite reason --
# it ignored VRAM. Three 4B seeds and three 8B seeds all OOM'd at the first
# training step: 4B used 43.7 of 44.4 GiB, 8B used 78.4 of 79.2 GiB.
#
# THE RULE. Budget the card explicitly:
#
#   weights_GB  = 2 * params_B * 2 bytes          (policy + frozen KL base, fp16)
#   budget_GB   = VRAM_GB - weights_GB - RESERVE_GB
#   per_sample  = CAL_GB_PER_SAMPLE * (hidden*layers) / (1024*28)
#   MICRO_BATCH = floor(budget / per_sample / SAFETY), clamped to [1, 128]
#
# CALIBRATION. CAL_GB_PER_SAMPLE = 0.78 comes from the single measured working
# point we have: Qwen3-0.6B trains at micro-batch 48 on a 40 GB card, i.e.
# (40 - 2.4) / 48 = 0.78 GB per sample. SAFETY = 1.4 exists because that is a
# ONE-POINT calibration with an assumed linearity in both batch size and
# hidden*layers; it is not measured at 4B or 8B. Smoke-test any new size before
# committing a fleet to it.
#
# WHAT THIS RULE DOES NOT FIX. Micro-batch bounds the TRANSIENT per-chunk
# activation, not the retained autograd graph: a require_grad scoring call keeps
# every micro-batch's graph alive until the caller's backward, so peak memory on
# the grad path is nearly independent of chunk size. Measured: 4B OOM'd using
# 43.7 GiB at micro-batch 16 and 43.6 GiB at micro-batch 7 -- cutting the chunk
# by more than half moved peak memory by 0.1 GiB.
# --gradient-checkpointing is the lever that actually cuts the retained graph,
# and the 4B/8B launchers now pass it by default (GRAD_CKPT=0 to disable). Use
# this rule to size chunks sensibly; do not expect it to prevent an OOM on its
# own.
#
# NUMERICS. micro_batch_size is gradient-accumulation chunking only: the
# optimizer step aggregates the same total gradient regardless of chunk size, and
# it is excluded from _critical_cfg_signature (train_lquant.py) so it does not
# invalidate resume or the baseline cache. It does change floating-point
# summation order, so runs at different micro-batch are NOT bitwise identical.
#
# Required inputs: MODEL_HIDDEN, MODEL_LAYERS, MODEL_PARAMS_B.
# Optional: MICRO_BATCH (explicit override wins), RESERVE_GB, SAFETY, CAL.

_mb_reserve_gb=${RESERVE_GB:-3}
_mb_safety=${SAFETY:-1.4}
_mb_cal=${CAL:-0.78}

if [[ -n "${MICRO_BATCH:-}" ]]; then
  echo "[mb] explicit override MICRO_BATCH=${MICRO_BATCH} (model-aware rule skipped)"
else
  _mb_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -z "${_mb_vram}" ]]; then
    echo "[mb] WARNING: no nvidia-smi reading; falling back to MICRO_BATCH=4" >&2
    MICRO_BATCH=4
  else
    MICRO_BATCH=$(awk -v vram="${_mb_vram}" -v hid="${MODEL_HIDDEN:?MODEL_HIDDEN required}" \
                      -v lay="${MODEL_LAYERS:?MODEL_LAYERS required}" \
                      -v par="${MODEL_PARAMS_B:?MODEL_PARAMS_B required}" \
                      -v res="${_mb_reserve_gb}" -v saf="${_mb_safety}" -v cal="${_mb_cal}" '
      BEGIN {
        vram_gb    = vram / 1024.0;
        weights_gb = 2.0 * par * 2.0;             # two fp16 copies
        budget     = vram_gb - weights_gb - res;
        per_sample = cal * (hid * lay) / (1024.0 * 28.0);
        mb         = int(budget / per_sample / saf);
        if (mb < 1)   mb = 1;
        if (mb > 128) mb = 128;
        printf "%d", mb;
      }')
    echo "[mb] vram=${_mb_vram} MB  hidden=${MODEL_HIDDEN} layers=${MODEL_LAYERS}" \
         "params=${MODEL_PARAMS_B}B  cal=${_mb_cal} safety=${_mb_safety} reserve=${_mb_reserve_gb}GB" \
         "-> MICRO_BATCH=${MICRO_BATCH}"
  fi
fi
