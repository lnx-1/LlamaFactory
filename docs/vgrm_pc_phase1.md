# VGRM-PC phase 1: verdict branch and objective

This is the v0.3 architecture and loss, isolated from the existing trainer.
It does not register a new training stage or change how the base model generates text.

## Review points

- `src/llamafactory/vgrm_pc/model.py`: gather the original LM-head input at
  `verdict_position - 1`. Only this state goes through the three-layer GELU
  encoder, separate 64-dimensional mean and log-variance projections, latent
  sampling, and the two-layer decoder. Read A/B from `h + alpha * D(z)`;
  `alpha` defaults to a fixed 0.1 and is stored in the branch state dict.
- The caller supplies the original frozen dense LM head and the two token IDs
  in A/B order. The branch reads only those rows, computing fused logits in FP32.
  A frozen head still propagates gradients to the input; quantized, sharded or
  adapter-wrapped heads are outside this implementation.
- `src/llamafactory/vgrm_pc/loss.py`: principle CE and critique/format CE average
  valid target positions per sequence, then average supervised sequences.
  Final A/B is excluded from both text masks. Preference CE averages across
  latent draws and batch; Gaussian KL sums over latent dimensions and averages
  across batch. The caller supplies the current KL weight `beta`.

## Verification

From the LlamaFactory repository root, with PyTorch available:

```sh
PYTHONPATH=src OMP_NUM_THREADS=2 python tests/model/test_vgrm_pc.py -v
```

The tiny PEFT integration test also needs a compatible Transformers/PEFT pair.
The real-model smoke check loads only local weights and runs no optimizer:

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/vgrm_pc_smoke.py ../../model/LFM3-1.7B
```

The local LFM3 model code may require a newer Transformers version than the
repository pins for its training workflow. Use separate smoke-test dependencies;
do not change the training environment to run the smoke script.

Verified on WSL2 CUDA with the local BF16 LFM3 weights, Transformers 5.21.0 and
tokenizers 0.23.1 in a temporary dependency directory. The check produced
`h: [1, 2048]` and `A/B logits: [1, 4, 2]`. It freezes the full LFM3 backbone;
the real PEFT LoRA gradient check is separate and uses a tiny random model.
This is not verification of LFM3 through LlamaFactory's trainer.

Data schemas, preprocessing, trainer/optimizer wiring, checkpoint integration,
evaluation and deployment are deferred until approval.
