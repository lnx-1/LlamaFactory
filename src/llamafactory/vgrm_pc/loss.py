# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F

from .model import VerdictOutput, _validate_verdict_positions


@dataclass(frozen=True)
class SupervisedLoss:
    total: torch.Tensor
    principle: torch.Tensor
    critique: torch.Tensor
    preference: torch.Tensor
    kl: torch.Tensor


def _masked_text_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rows, positions = mask.nonzero(as_tuple=True)
    losses = F.cross_entropy(logits[rows, positions - 1].float(), targets[rows, positions], reduction="none")
    sums = losses.new_zeros(targets.shape[0]).index_add(0, rows, losses)
    counts = mask.sum(dim=1)
    # Rows without supervision do not contribute to this term's denominator.
    return (sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)


def compute_supervised_loss(
    text_logits: torch.Tensor,
    input_ids: torch.Tensor,
    principle_mask: torch.Tensor,
    critique_format_mask: torch.Tensor,
    verdict_positions: torch.Tensor,
    verdict: VerdictOutput,
    preference_target: torch.Tensor,
    *,
    beta: float,
    lambda_principle: float = 1.0,
    lambda_critique: float = 1.0,
    lambda_preference: float = 1.0,
) -> SupervisedLoss:
    """Compute the v0.3 joint objective, with masks aligned to target token positions."""
    if text_logits.ndim != 3 or input_ids.shape != text_logits.shape[:2] or input_ids.dtype != torch.long:
        raise ValueError("Expected text_logits [batch, sequence, vocabulary] and long input_ids [batch, sequence].")
    batch, length = input_ids.shape
    device = text_logits.device
    _validate_verdict_positions(verdict_positions, batch, length, device)
    if batch == 0 or input_ids.device != device:
        raise ValueError("input_ids must be non-empty and on the logits device.")
    for mask in (principle_mask, critique_format_mask):
        if mask.shape != input_ids.shape or mask.dtype != torch.bool or mask.device != device:
            raise ValueError("Text masks must be boolean [batch, sequence] tensors on the logits device.")
        if mask[:, 0].any() or mask.gather(1, verdict_positions[:, None]).any():
            raise ValueError("Do not supervise token 0 or final A/B with the text loss.")
    if (principle_mask & critique_format_mask).any():
        raise ValueError("Principle and critique text masks must not overlap.")
    if (
        preference_target.shape != (batch,)
        or preference_target.dtype != torch.long
        or preference_target.device != device
    ):
        raise ValueError("preference_target must be long [batch] on the logits device.")
    if ((preference_target < 0) | (preference_target > 1)).any():
        raise ValueError("Preference labels must be 0 (A) or 1 (B).")
    if (
        verdict.logits.ndim != 3
        or verdict.logits.shape[0] != batch
        or verdict.logits.shape[1] < 1
        or verdict.logits.shape[2] != 2
        or verdict.mu.ndim != 2
        or verdict.mu.shape[0] != batch
        or verdict.logvar.shape != verdict.mu.shape
        or any(t.device != device for t in (verdict.logits, verdict.mu, verdict.logvar))
    ):
        raise ValueError("Expected verdict logits [batch, samples, 2] and mu/logvar [batch, latent_size].")
    if any(not isfinite(w) or w < 0 for w in (beta, lambda_principle, lambda_critique, lambda_preference)):
        raise ValueError("Loss weights must be finite and non-negative.")

    principle = _masked_text_ce(text_logits, input_ids, principle_mask)
    critique = _masked_text_ce(text_logits, input_ids, critique_format_mask)
    targets = preference_target[:, None].expand(-1, verdict.logits.shape[1])
    preference = F.cross_entropy(verdict.logits.float().reshape(-1, 2), targets.reshape(-1))
    mu, logvar = verdict.mu.float(), verdict.logvar.float()
    kl = 0.5 * (mu.square() + torch.expm1(logvar) - logvar).sum(dim=-1).mean()
    total = lambda_principle * principle + lambda_critique * critique + lambda_preference * preference + beta * kl
    return SupervisedLoss(total=total, principle=principle, critique=critique, preference=preference, kl=kl)
