"""Shared helpers for the Gemma tail-shape scoring scripts.

Two jobs:

1. Load an HF checkpoint as a text causal LM regardless of whether the config
   declares a text-only ``*ForCausalLM`` (gemma-3-270m/1b) or the multimodal
   ``*ForConditionalGeneration`` (gemma-3-4b, which bundles a vision tower but is
   perfectly happy with text-only inputs).
2. Compute next-token log-probs for only the last ``k`` positions via the
   ``logits_to_keep`` forward argument, so we never materialize a full
   ``[batch, seq_len, vocab]`` logit tensor. Gemma 3's vocab is 262,144, so a
   full-sequence head on a 512-token batch would be tens of GB; restricting to
   the last ``k`` positions keeps it small and also respects any final-logit
   transform the model applies inside ``forward`` (none for gemma-3, but this
   keeps the code correct for other families too).
"""
from __future__ import annotations

import os
import re

# Reduce allocator fragmentation from OOM-halving retries on long-sequence
# batches. Must be set before the CUDA caching allocator initializes; this
# module is imported before any model touches the GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_WS_RE = re.compile(r"\s+")


def normalize_prompt_text(text: str) -> str:
    """Model-independent dedup key: collapse all whitespace runs (including the
    non-breaking spaces that jailbreak templates use as word separators) to a
    single space and strip. Keeps case/content, so genuinely different prompts
    stay distinct, while whitespace-variant copies of the same spammed template
    collapse to one key. Being model-independent, the deduped set is identical
    across tokenizers/model sizes.
    """
    return _WS_RE.sub(" ", text).strip()

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    return DTYPE_MAP[name]


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_causal_lm(model_path: str, *, dtype: torch.dtype, device: str):
    """Load ``model_path`` as a causal LM, handling multimodal Gemma configs.

    Returns the model in eval mode on ``device``.
    """
    cfg = AutoConfig.from_pretrained(model_path)
    arch = ""
    if getattr(cfg, "architectures", None):
        arch = cfg.architectures[0] or ""
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        # Multimodal checkpoint (e.g. gemma-3-4b-it). Load the concrete class and
        # drive it with text-only inputs; it returns ``.logits`` over the text
        # vocabulary just like a plain causal LM.
        import transformers

        model_cls = getattr(transformers, arch)
        model = model_cls.from_pretrained(model_path, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    return model.to(device).eval()


def text_backbone(model):
    """Return the text decoder module that yields ``last_hidden_state``.

    For text-only ``*ForCausalLM`` this is ``model.model``; for the multimodal
    ``*ForConditionalGeneration`` the text decoder is nested at
    ``model.model.language_model`` (alongside the vision tower). Calling it
    directly with text-only ``input_ids`` returns only the final-layer hidden
    states, avoiding both the vision path and the all-layers memory blow-up of
    ``output_hidden_states=True``.
    """
    m = getattr(model, "model", None)
    if m is None:
        return model.base_model
    lm = getattr(m, "language_model", None)
    return lm if lm is not None else m


def model_hidden_size(model) -> int:
    cfg = model.config
    for attr in ("hidden_size", "n_embd"):
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    tcfg = getattr(cfg, "text_config", None)
    if tcfg is not None and getattr(tcfg, "hidden_size", None):
        return int(tcfg.hidden_size)
    raise ValueError("Could not determine hidden size from model config")


def lm_head_module(model):
    """Return the LM head that maps final hidden states to vocab logits.

    Both ``*ForCausalLM`` and Gemma's ``*ForConditionalGeneration`` expose it as
    ``lm_head``. Assumes no post-head logit transform (true for Gemma 3, which
    has ``final_logit_softcapping=None``).
    """
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError(f"{type(model).__name__} has no lm_head")
    return head


@torch.no_grad()
def last_k_target_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
    *,
    head_chunk: int = 256,
) -> torch.Tensor:
    """Teacher-forced log-probs of the last ``k`` tokens, without truncation.

    Returns ``[batch, k]`` where entry ``[i, j]`` is
    ``log P(input_ids[i, L-k+j] | input_ids[i, :L-k+j])``.

    Unlike ``last_k_log_probs`` this never materializes a ``[batch, k, vocab]``
    logit tensor: the backbone runs once (hidden states are cheap even at 8k
    tokens), then the LM head is applied in ``head_chunk``-position slices and
    only the target token's log-prob is kept per position. Memory is bounded by
    ``batch * head_chunk * vocab`` regardless of ``k``, so arbitrarily long
    response spans can be scored exactly.
    """
    hidden = text_backbone(model)(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state  # [n, L, H]
    head = lm_head_module(model)
    n, L, _ = hidden.shape
    k = min(k, L - 1)
    # Logits at position p predict the token at p+1: to score targets at
    # positions [L-k, L-1] we need hidden at positions [L-k-1, L-2].
    h_lo, h_hi = L - k - 1, L - 1
    targets = input_ids[:, L - k:]  # [n, k]
    out = torch.empty((n, k), dtype=torch.float32, device=input_ids.device)
    for a in range(h_lo, h_hi, head_chunk):
        b = min(a + head_chunk, h_hi)
        logits = head(hidden[:, a:b, :]).float()  # [n, b-a, V]
        lse = torch.logsumexp(logits, dim=-1)  # [n, b-a]
        t = targets[:, a - h_lo : b - h_lo]  # [n, b-a]
        tgt = logits.gather(-1, t.unsqueeze(-1)).squeeze(-1)  # [n, b-a]
        out[:, a - h_lo : b - h_lo] = tgt - lse
        del logits
    return out


@torch.no_grad()
def last_k_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Return log-softmax over the vocab for the last ``k`` input positions.

    Output shape ``[batch, k, vocab]``. Uses ``logits_to_keep`` so the head is
    only applied to the last ``k`` positions.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=k,
    )
    logits = out.logits[:, -k:, :].float()
    return F.log_softmax(logits, dim=-1)
