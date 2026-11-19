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


"""Check the isolated VGRM-PC components on a local LFM3 checkpoint (no optimizer or dataset)."""

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.vgrm_pc.loss import compute_supervised_loss
from llamafactory.vgrm_pc.model import ResidualVariationalVerdict, gather_verdict_hidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    torch.manual_seed(17)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": args.device},
        )
        .requires_grad_(False)
        .eval()
    )
    head = model.get_output_embeddings()
    if model.config.tie_word_embeddings:
        assert head.weight is model.get_input_embeddings().weight

    prefix = "Principles: Prefer factual answers.\nCritique: A is correct.\nVerdict:"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    candidate_suffixes = [tokenizer.encode(f"{prefix} {label}", add_special_tokens=False) for label in ("A", "B")]
    assert all(ids[:-1] == prefix_ids for ids in candidate_suffixes), "Test requires single-token A/B suffixes."
    a_id, b_id = (ids[-1] for ids in candidate_suffixes)
    ids = torch.tensor([candidate_suffixes[0]], device=args.device)
    positions = torch.tensor([len(prefix_ids)], device=args.device)
    attention = torch.ones_like(ids)
    principle_mask = torch.zeros_like(ids, dtype=torch.bool)
    critique_mask = torch.zeros_like(ids, dtype=torch.bool)
    principle_mask[0, 1] = True
    critique_mask[0, -2] = True

    with torch.no_grad():
        hidden = model.base_model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
        text_logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        prefix_logits = model(input_ids=ids[:, :-1], use_cache=False, logits_to_keep=1).logits
        torch.testing.assert_close(text_logits[0, -2], prefix_logits[0, -1], atol=0.125, rtol=0.02)
        h = gather_verdict_hidden(hidden, positions, attention)
        original = F.linear(h.float(), head.weight[[a_id, b_id]].float())
        no_branch = ResidualVariationalVerdict(model.config.hidden_size, alpha=0).to(args.device)
        plain = no_branch(h, head, (a_id, b_id), samples=2)
        torch.testing.assert_close(plain.logits, original[:, None].expand_as(plain.logits))

    branch = ResidualVariationalVerdict(model.config.hidden_size).to(args.device)
    verdict = branch(h, head, (a_id, b_id))
    assert verdict.logits.var(dim=1).max().item() > 0
    losses = compute_supervised_loss(
        text_logits,
        ids,
        principle_mask,
        critique_mask,
        positions,
        verdict,
        torch.tensor([0], device=args.device),
        beta=0.001,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in branch.parameters()
    )
    assert branch.decoder[-1].weight.grad.abs().sum().item() > 0
    assert head.weight.grad is None
    print(f"{type(model).__name__}: h={tuple(h.shape)}, A/B logits={tuple(verdict.logits.shape)}")
    print("losses:", {name: round(value.item(), 6) for name, value in vars(losses).items()})
    print("PASS: target masking, causal state, residual A/B, finite loss and frozen-head backward.")


if __name__ == "__main__":
    main()
