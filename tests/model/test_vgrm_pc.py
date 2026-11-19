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


import io
import unittest
from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch import nn

from llamafactory.vgrm_pc.loss import compute_supervised_loss
from llamafactory.vgrm_pc.model import ResidualVariationalVerdict, VerdictOutput, gather_verdict_hidden


class TestResidualVerdict(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = nn.Linear(12, 9).requires_grad_(False)
        self.hidden = torch.randn(2, 12)
        self.epsilon = torch.randn(2, 4, 64)

    def test_zero_alpha_recovers_original_candidate_logits(self):
        branch = ResidualVariationalVerdict(12, alpha=0)
        result = branch(self.hidden, self.head, (5, 2), epsilon=self.epsilon)
        original = self.head(self.hidden)[:, None, [5, 2]].expand_as(result.logits)
        torch.testing.assert_close(result.logits, original)
        torch.testing.assert_close(result.logits.var(1), torch.zeros(2, 2))

    def test_matches_explicit_fusion_and_full_head(self):
        branch = ResidualVariationalVerdict(12)
        with torch.no_grad():
            branch.logvar_projection.bias[0] = 100
            branch.logvar_projection.bias[1] = -100
        result = branch(self.hidden, self.head, (5, 2), epsilon=self.epsilon)
        latent = result.mu[:, None] + (result.logvar[:, None] / 2).exp() * self.epsilon
        fused = self.hidden[:, None] + branch.alpha * branch.decoder(latent)
        torch.testing.assert_close(result.logits, self.head(fused)[..., [5, 2]])
        torch.testing.assert_close(result.logvar[:, :2], torch.tensor([[4.0, -8.0]]).expand(2, 2))
        assert result.logits.var(1).max().item() > 0

    def test_preference_gradients_reach_branch_and_adapter_not_frozen_head(self):
        frozen_embedding = nn.Embedding(9, 12)
        self.head.weight = frozen_embedding.weight
        self.head.requires_grad_(False)
        down = nn.Linear(12, 3, bias=False)
        up = nn.Linear(3, 12, bias=False)
        hidden = self.hidden + up(down(self.hidden))
        branch = ResidualVariationalVerdict(12)
        result = branch(hidden, self.head, (5, 2), epsilon=self.epsilon)
        preference = -result.logits.log_softmax(-1)[..., 0].mean()
        preference.backward()
        for parameter in (down.weight, up.weight, *branch.parameters()):
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0
        assert self.head.weight.grad is None
        assert frozen_embedding.weight.grad is None
        assert not frozen_embedding.weight.requires_grad
        assert branch.alpha.grad is None

    def test_zero_alpha_blocks_preference_gradient_only(self):
        branch = ResidualVariationalVerdict(12, alpha=0)
        result = branch(self.hidden, self.head, (5, 2), epsilon=self.epsilon)
        preference = -result.logits.log_softmax(-1)[..., 0].mean()
        preference.backward()
        for parameter in branch.parameters():
            assert parameter.grad.count_nonzero().item() == 0

    def test_bfloat16_autocast_returns_finite_fp32_verdict(self):
        branch = ResidualVariationalVerdict(12)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = branch(self.hidden.bfloat16(), self.head.bfloat16(), (5, 2), epsilon=self.epsilon)
        for tensor in (result.logits, result.mu, result.logvar):
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()

    def test_state_dict_restores_alpha_and_branch(self):
        branch = ResidualVariationalVerdict(12, alpha=0.3)
        expected = branch(self.hidden, self.head, (5, 2), epsilon=self.epsilon)
        checkpoint = io.BytesIO()
        torch.save(branch.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = ResidualVariationalVerdict(12)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        actual = restored(self.hidden, self.head, (5, 2), epsilon=self.epsilon)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.mu, expected.mu)
        torch.testing.assert_close(actual.logvar, expected.logvar)
        torch.testing.assert_close(restored.alpha, branch.alpha)

    def test_explicit_target_position_selects_predecessor_and_rejects_padding(self):
        hidden = torch.randn(2, 5, 12, requires_grad=True)
        attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        selected = gather_verdict_hidden(hidden, torch.tensor([4, 2]), attention)
        torch.testing.assert_close(selected, hidden[[0, 1], [3, 1]])
        selected.sum().backward()
        assert hidden.grad.count_nonzero().item() == 24
        with self.assertRaisesRegex(ValueError, "padding"):
            gather_verdict_hidden(hidden, torch.tensor([1, 3]), attention)

    def test_rejects_trainable_head_and_invalid_candidate_ids(self):
        branch = ResidualVariationalVerdict(12)
        with self.assertRaisesRegex(ValueError, "Freeze"):
            branch(self.hidden, nn.Linear(12, 9), (5, 2))
        with self.assertRaisesRegex(ValueError, "distinct"):
            branch(self.hidden, self.head, (5, 5))


class TestSupervisedLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.ids = torch.tensor([[0, 2, 1, 3, 4], [0, 1, 2, 4, 0]])
        self.text = torch.randn(2, 5, 6, requires_grad=True)
        self.principle = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.bool)
        self.critique = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
        self.positions = torch.tensor([4, 3])
        self.target = torch.tensor([0, 1])
        self.verdict = VerdictOutput(
            logits=torch.tensor([[[3.0, 0.0], [-1.0, 0.0]], [[2.0, 0.0], [0.0, 1.0]]]),
            mu=torch.tensor([[0.0, 1.0], [2.0, 0.0]]),
            logvar=torch.tensor([[0.0, 0.4], [-0.5, 0.0]]),
        )

    def compute_loss(self):
        return compute_supervised_loss(
            self.text,
            self.ids,
            self.principle,
            self.critique,
            self.positions,
            self.verdict,
            self.target,
            beta=0.2,
            lambda_principle=2.0,
            lambda_critique=3.0,
            lambda_preference=4.0,
        )

    def test_matches_per_sequence_text_ce_expected_sample_ce_and_kl(self):
        loss = self.compute_loss()
        lp = (
            F.cross_entropy(self.text[0, :2], self.ids[0, 1:3]) + F.cross_entropy(self.text[1, :1], self.ids[1, 1:2])
        ) / 2
        lc = F.cross_entropy(self.text[0, 2:3], self.ids[0, 3:4])
        lv = F.cross_entropy(self.verdict.logits.flatten(0, 1), torch.tensor([0, 0, 1, 1]))
        mu, ell = self.verdict.mu, self.verdict.logvar
        kl = 0.5 * (mu.square() + ell.exp() - ell - 1).sum(1).mean()
        for actual, expected in zip(
            (loss.principle, loss.critique, loss.preference, loss.kl, loss.total),
            (lp, lc, lv, kl, 2 * lp + 3 * lc + 4 * lv + 0.2 * kl),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        loss.total.backward()
        assert self.text.grad[0, 3:].count_nonzero().item() == 0
        assert self.text.grad[1, 1:].count_nonzero().item() == 0

    def test_all_empty_text_masks_are_finite_and_zero(self):
        self.principle.zero_()
        self.critique.zero_()
        loss = self.compute_loss()
        assert loss.principle.item() == 0
        assert loss.critique.item() == 0
        loss.total.backward()
        assert self.text.grad.count_nonzero().item() == 0

    def test_rejects_verdict_in_text_loss(self):
        self.critique[0, 4] = True
        with self.assertRaisesRegex(ValueError, "final A/B"):
            self.compute_loss()

    def test_rejects_non_binary_preference_target(self):
        self.target[1] = -100
        with self.assertRaisesRegex(ValueError, "0 \\(A\\) or 1 \\(B\\)"):
            self.compute_loss()


@unittest.skipUnless(find_spec("peft") and find_spec("transformers"), "PEFT and Transformers are not installed")
class TestLoRAGradient(unittest.TestCase):
    def test_joint_objective_updates_real_lora_and_branch(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(11)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            tie_word_embeddings=True,
        )
        base = LlamaForCausalLM(config)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
        head = base.get_output_embeddings()
        assert head.weight is base.get_input_embeddings().weight
        assert not head.weight.requires_grad

        ids = torch.tensor([[1, 6, 8, 9, 3], [1, 7, 9, 8, 4]])
        positions = torch.tensor([4, 4])
        attention = torch.ones_like(ids)
        outputs = model(input_ids=ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
        hidden = gather_verdict_hidden(outputs.hidden_states[-1], positions, attention)
        branch = ResidualVariationalVerdict(config.hidden_size)
        verdict = branch(hidden, head, (3, 4))
        principle = torch.zeros_like(ids, dtype=torch.bool)
        critique = torch.zeros_like(ids, dtype=torch.bool)
        principle[:, 1] = True
        critique[:, 3] = True
        loss = compute_supervised_loss(
            outputs.logits, ids, principle, critique, positions, verdict, torch.tensor([0, 1]), beta=0.001
        )
        trainable_backbone = {
            name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert trainable_backbone
        assert all("lora_" in name for name in trainable_backbone)
        preference_grads = torch.autograd.grad(loss.preference, tuple(trainable_backbone.values()), retain_graph=True)
        assert any(grad.abs().sum().item() > 0 for grad in preference_grads)
        loss.total.backward()
        assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in trainable_backbone.values())
        assert branch.decoder[-1].weight.grad.abs().sum().item() > 0
        assert head.weight.grad is None


if __name__ == "__main__":
    unittest.main()
