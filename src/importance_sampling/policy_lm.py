from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None


@dataclass
class LoRASetup:
    use_lora: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"


class PolicyLM:
    """Behavior policy π_θ used to sample prompts and compute their log-probability.

    This wrapper uses HuggingFace causal LMs and optionally applies LoRA adapters.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[object] = None,
        lora: Optional[LoRASetup] = None,
    ):
        assert AutoModelForCausalLM is not None and AutoTokenizer is not None, "transformers not installed"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            # Ensure pad exists to allow batching
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if torch_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        # If device_map is None, move to device directly
        if device_map is None:
            self.model.to(device)
        self.model.eval()

        self._use_lora = bool(lora and lora.use_lora and get_peft_model is not None and LoraConfig is not None)
        if self._use_lora:
            peft_cfg = LoraConfig(
                r=lora.r,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                bias=lora.bias,
                target_modules=list(lora.target_modules),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg)
            # Explicitly zero-initialize LoRA up-projection so delta == 0.
            # PEFT stores adapter stacks in ModuleDict/ParameterDict-like
            # containers, not plain dicts, so handle any mapping with .items().
            try:
                for _name, module in self.model.named_modules():
                    lora_B = getattr(module, "lora_B", None)
                    if lora_B is None or not hasattr(lora_B, "items"):
                        continue
                    for _adapter_name, b in lora_B.items():
                        if hasattr(b, "weight") and b.weight is not None:
                            b.weight.data.zero_()
                            if getattr(b, "bias", None) is not None:
                                b.bias.data.zero_()
                        elif isinstance(b, torch.nn.Parameter):
                            b.data.zero_()
            except Exception:
                pass
            # Freeze base weights; train only LoRA params
            for n, p in self.model.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
            # Keep model in eval() to avoid dropout; gradients still flow
            self.model.eval()
        else:
            # Full finetune if no LoRA requested/available
            for p in self.model.parameters():
                p.requires_grad_(True)
            # Keep model in eval during likelihood computations for stability
            self.model.eval()

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.shape[0])

    def _initial_context(self) -> List[int]:
        """Return a single start token id to avoid empty inputs.

        Prefer BOS if available; otherwise fall back to EOS or PAD. The
        tokenizer PAD token is ensured to exist in __init__, so this should
        always return a non-empty context, which some models (e.g., Qwen3)
        require to build attention masks.
        """
        # Try BOS, then EOS, then PAD
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            tok_id = getattr(self.tokenizer, attr, None)
            if tok_id is not None:
                return [int(tok_id)]
        # Final fallback: index 0 (should not be reached in practice)
        return [0]

    @torch.no_grad()
    def sample_exact(self, batch_size: int, L: int) -> Tuple[torch.LongTensor, List[str], torch.Tensor]:
        """Sample from full softmax at each step.

        Returns tokens [B, L] (policy tokenizer space), decoded texts, and per-sample log π_θ(x).
        """
        device = self.device()
        ctx = torch.tensor([self._initial_context()], dtype=torch.long, device=device)  # [1, C]
        tokens = torch.full((batch_size, 0), 0, dtype=torch.long, device=device)
        logp_sum = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # Temporarily switch to eval mode for stable sampling
        prev_mode = self.model.training
        self.model.eval()
        for t in range(L):
            # Build batch contexts with their own generated prefixes
            # Efficient way: generate step by step, one forward for the whole batch
            if t == 0:
                cur = ctx.expand(batch_size, ctx.shape[1])
            else:
                cur = torch.cat([ctx.expand(batch_size, ctx.shape[1]), tokens], dim=1)
            out = self.model(input_ids=cur, use_cache=False)
            logits = out.logits[:, -1, :].float()
            # Replace non-finite logits to avoid NaNs in softmax
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            logprobs = F.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            # accumulate log prob under full π
            lp = logprobs.gather(-1, next_ids.view(-1, 1)).squeeze(1)
            logp_sum += lp
            tokens = torch.cat([tokens, next_ids.view(-1, 1)], dim=1)
        # Restore prior mode
        if prev_mode:
            self.model.train()
        # decode texts
        texts = [self.tokenizer.decode(tokens[b].tolist(), skip_special_tokens=False) for b in range(batch_size)]
        return tokens.detach(), texts, logp_sum.detach()

    @torch.no_grad()
    def sample_top_p_temp(
        self, batch_size: int, L: int, top_p: float = 0.95, temperature: float = 1.0
    ) -> Tuple[torch.LongTensor, List[str], torch.Tensor, torch.Tensor]:
        """Sample using nucleus at temperature.

        Returns tokens [B, L], decoded texts, log μ(x) (sampler density), and log π_θ(x).
        """
        device = self.device()
        ctx = torch.tensor([self._initial_context()], dtype=torch.long, device=device)
        tokens = torch.full((batch_size, 0), 0, dtype=torch.long, device=device)
        logmu_sum = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # We also gather stepwise tokens to later compute logπ via teacher-forcing
        prev_mode = self.model.training
        self.model.eval()
        for t in range(L):
            if t == 0:
                cur = ctx.expand(batch_size, ctx.shape[1])
            else:
                cur = torch.cat([ctx.expand(batch_size, ctx.shape[1]), tokens], dim=1)
            out = self.model(input_ids=cur, use_cache=False)
            logits = out.logits[:, -1, :].float()
            logits = logits / float(max(1e-8, temperature))
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            probs = F.softmax(logits, dim=-1)
            # Build nucleus mask per batch element
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # mask tokens beyond top_p
            mask = cumsum <= top_p
            # ensure at least one token
            mask[:, 0] = True
            # renormalize probabilities inside nucleus
            renorm = (sorted_probs * mask).sum(dim=-1, keepdim=True)
            renormed = (sorted_probs * mask) / (renorm + 1e-20)
            # sample from renormalized nucleus distribution
            next_samples = torch.multinomial(renormed, num_samples=1).squeeze(1)
            next_ids = sorted_idx.gather(-1, next_samples.view(-1, 1)).squeeze(1)
            # log μ: probability under truncated sampler
            chosen_probs = renormed.gather(-1, next_samples.view(-1, 1)).squeeze(1)
            logmu_sum += torch.log(chosen_probs + 1e-20)
            tokens = torch.cat([tokens, next_ids.view(-1, 1)], dim=1)
        # Restore prior mode
        if prev_mode:
            self.model.train()
        # Compute logπ via teacher forcing
        logpi = self.logprob(tokens)
        texts = [self.tokenizer.decode(tokens[b].tolist(), skip_special_tokens=False) for b in range(batch_size)]
        return tokens.detach(), texts, logmu_sum.detach(), logpi.detach()

    def logprob(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Sum log π_θ(x) per example by teacher forcing on sampled tokens.

        tokens: [B, L]
        Returns: [B] log-prob per sequence.
        """
        device = self.device()
        B, L = tokens.shape
        bos = self._initial_context()
        # Build inputs and labels to sum only over prompt tokens
        inp = torch.full((B, len(bos) + L), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        labels = torch.full_like(inp, fill_value=-100)
        if len(bos) > 0:
            inp[:, : len(bos)] = torch.tensor(bos, dtype=torch.long, device=device).view(1, -1).expand(B, -1)
        inp[:, len(bos):] = tokens
        labels[:, len(bos):] = tokens
        # Temporarily ensure eval mode (dropout off) for density computation
        prev_mode = self.model.training
        self.model.eval()
        out = self.model(input_ids=inp, labels=labels, use_cache=False)
        logits = out.logits  # [B, L+bos, V]
        logprobs = F.log_softmax(logits, dim=-1)
        lp = torch.zeros(B, dtype=torch.float32, device=device)
        for b in range(B):
            if L == 0:
                continue
            lp_b = logprobs[b, len(bos) : len(bos) + L, :]
            tgt = tokens[b]
            lp[b] = lp_b.gather(-1, tgt.view(-1, 1)).squeeze(1).sum()
        if prev_mode:
            self.model.train()
        return lp

    def nll(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Per-sample NLL of tokens under current policy with teacher forcing."""
        return -self.logprob(tokens)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_pretrained(self, out_dir: str):
        # Delegate to HF save; works with PEFT or base
        self.model.save_pretrained(out_dir)
