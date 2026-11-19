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
from torch import nn


@dataclass(frozen=True)
class VerdictOutput:
    logits: torch.Tensor  # [batch, samples, 2], ordered A, B
    mu: torch.Tensor  # [batch, latent_size]
    logvar: torch.Tensor  # [batch, latent_size], clipped


def gather_verdict_hidden(
    hidden_states: torch.Tensor, verdict_positions: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Read the LM-head input at t-1 for each explicit verdict target at t."""
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Expected hidden_states [batch, sequence, hidden] and attention_mask [batch, sequence].")
    batch, length, _ = hidden_states.shape
    _validate_verdict_positions(verdict_positions, batch, length, hidden_states.device)
    if attention_mask.device != hidden_states.device:
        raise ValueError("attention_mask must be on the same device as hidden_states.")
    rows = torch.arange(batch, device=hidden_states.device)
    if (
        not (attention_mask[rows, verdict_positions] == 1).all()
        or not (attention_mask[rows, verdict_positions - 1] == 1).all()
    ):
        raise ValueError("Neither the verdict target nor its predecessor may be padding.")
    return hidden_states[rows, verdict_positions - 1]


def _validate_verdict_positions(positions: torch.Tensor, batch: int, length: int, device: torch.device) -> None:
    if positions.shape != (batch,) or positions.dtype != torch.long or positions.device != device:
        raise ValueError("verdict_positions must be a long [batch] tensor on the input device.")
    if ((positions < 1) | (positions >= length)).any():
        raise ValueError("Each verdict target position must be in [1, sequence_length).")


class ResidualVariationalVerdict(nn.Module):
    """Use latent samples to correct the final hidden state, not to replace it."""

    def __init__(
        self,
        hidden_size: int,
        *,
        latent_size: int = 64,
        alpha: float = 0.1,
        logvar_bounds: tuple[float, float] = (-8.0, 4.0),
    ) -> None:
        super().__init__()
        if hidden_size < 1 or latent_size < 1:
            raise ValueError("hidden_size and latent_size must be positive.")
        if not isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative.")
        lower, upper = logvar_bounds
        if not -80.0 <= lower < upper <= 80.0:
            raise ValueError("logvar bounds must be ordered and inside the safe FP32 range [-80, 80].")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.logvar_bounds = logvar_bounds
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU()
        )
        self.mu_projection = nn.Linear(64, latent_size)
        self.logvar_projection = nn.Linear(64, latent_size)
        self.decoder = nn.Sequential(nn.Linear(latent_size, 128), nn.GELU(), nn.Linear(128, hidden_size))

        with torch.no_grad():
            for layer in (self.mu_projection, self.logvar_projection, self.decoder[-1]):
                layer.weight.mul_(0.1)
                layer.bias.zero_()

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head: nn.Linear,
        candidate_token_ids: tuple[int, int],
        *,
        samples: int = 4,
        epsilon: torch.Tensor | None = None,
    ) -> VerdictOutput:
        """Score A/B using the frozen original linear LM head at the verdict only."""
        if hidden.ndim != 2 or hidden.shape[0] < 1 or hidden.shape[1] != self.hidden_size:
            raise ValueError("hidden must be a non-empty [batch, hidden_size] tensor.")
        if type(lm_head) is not nn.Linear or lm_head.in_features != self.hidden_size:
            raise ValueError("Expected the original dense nn.Linear LM head with matching hidden_size.")
        if any(parameter.requires_grad for parameter in lm_head.parameters()):
            raise ValueError("Freeze the original LM head (and any tied embeddings) before computing a verdict.")
        if lm_head.weight.device != hidden.device or self.encoder[0].weight.device != hidden.device:
            raise ValueError("The verdict state, branch, and LM head must be on the same device.")
        if samples < 1:
            raise ValueError("samples must be positive.")
        if (
            len(candidate_token_ids) != 2
            or candidate_token_ids[0] == candidate_token_ids[1]
            or any(
                type(token_id) is not int or not 0 <= token_id < lm_head.out_features
                for token_id in candidate_token_ids
            )
        ):
            raise ValueError("Supply two distinct vocabulary token IDs in (A, B) order.")

        features = self.encoder(hidden.to(dtype=self.encoder[0].weight.dtype))
        mu = self.mu_projection(features).float()
        logvar = self.logvar_projection(features).float().clamp(*self.logvar_bounds)
        sample_shape = (hidden.shape[0], samples, self.latent_size)
        if epsilon is None:
            epsilon = torch.randn(sample_shape, device=hidden.device, dtype=torch.float32)
        elif epsilon.shape != sample_shape or epsilon.device != hidden.device or not epsilon.is_floating_point():
            raise ValueError("epsilon must be floating point [batch, samples, latent_size] on the input device.")

        latent = mu[:, None] + (0.5 * logvar[:, None]).exp() * epsilon.float()
        correction = self.decoder(latent.to(dtype=self.decoder[0].weight.dtype))
        fused = hidden.float()[:, None, :] + self.alpha.float() * correction.float()

        # Select before projecting: no [batch, samples, vocabulary] allocation.
        token_ids = torch.tensor(candidate_token_ids, dtype=torch.long, device=hidden.device)
        weight = lm_head.weight.index_select(0, token_ids).float()
        bias = lm_head.bias.index_select(0, token_ids).float() if lm_head.bias is not None else None
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = F.linear(fused, weight, bias)
        return VerdictOutput(logits=logits, mu=mu, logvar=logvar)
