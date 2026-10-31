# Captum attribution on TorchTPU

`captum_attribution.py` runs [Captum](https://captum.ai) Integrated Gradients
on a TPU, checks the result against the same computation on CPU, reports the
warm wall-clock time, and optionally writes an XProf trace. It runs unmodified
Captum: attribution methods are plain PyTorch autograd, so they work on
TorchTPU the same way they work on CUDA.

The example has two models:

-   `--model tiny` (default): a small MLP with Captum `IntegratedGradients`. No
    downloads.
-   `--model Qwen/Qwen3-0.6B` (or another Hugging Face causal LM): Captum
    `LayerIntegratedGradients` of the next-token logit with respect to the
    input embeddings, with `logits_to_keep=1` so the full `[steps, T, V]`
    logits tensor is never materialized.

## Install

```bash
pip install captum transformers
```

## Run

TorchTPU reads its execution mode when it is imported, so one process runs one
mode:

```bash
# Strict eager mode (default).
python3 captum_attribution.py --model Qwen/Qwen3-0.6B

# Optimized eager mode (DeferAndFuse).
TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py --model Qwen/Qwen3-0.6B

# torch.compile(backend="tpu") around the model forward.
python3 captum_attribution.py --model Qwen/Qwen3-0.6B --compile
```

Other flags:

-   `--steps N`: number of integration steps (default 64).
-   `--internal-batch-size N`: chunk the integration steps to bound peak HBM.
    On a 32 GiB TPU v6e, 128 steps of Qwen3-0.6B fit in one batch; 256 do not.
-   `--precision highest|high|medium`: the `float32` matmul precision. The
    default here is `highest` so TPU attributions match CPU. The TorchTPU
    default (`high`) rounds `float32` matmul inputs to `bfloat16`, which shows
    up as about `1e-3` relative error in attributions.
-   `--profile-dir DIR`: write an XProf trace of the warm runs to `DIR` and
    print the `torch.profiler` operator table. Open the trace with
    `xprof --logdir DIR`.

## Expected output

Strict eager mode with the tiny model, TPU v6e:

```text
model=tiny mode=strict precision=highest steps=64
CPU reference (one run):           0.03 s
first call (includes compilation): 1.4 s
warm median of 3 runs:             0.065 s
max relative error vs CPU:         2.00e-07
```

Qwen3-0.6B, 64 steps, one TPU v6e chip, warm median seconds:

| Mode                           | Warm time | First call | Max relative error vs CPU |
| :----------------------------- | --------: | ---------: | ------------------------: |
| Strict eager                   |    0.80 s |        5 s |                   9.2e-05 |
| DeferAndFuse                   |    0.36 s |       43 s |                   9.3e-05 |
| `torch.compile(backend="tpu")` |    0.16 s |      130 s |                   7.0e-05 |

The compiled row needs Captum with
[meta-pytorch/captum#1952](https://github.com/meta-pytorch/captum/pull/1952)
and TorchTPU with
[google-pytorch/torch_tpu#6106](https://github.com/google-pytorch/torch_tpu/pull/6106).
Without them, the compiled Captum forward fails with `expected tensor to be on
tpu, got cpu` or returns zero attributions for part of the input.

## Reading the profile

With `--profile-dir`, strict eager mode shows most of the host time in
`aten::_to_copy`: each Captum step moves scalars and small tensors between the
host and the device, and in strict mode every such copy blocks. DeferAndFuse
batches the deferred ops into fused executables, which removes most of those
stalls. See
[Explainability as a performance lens](../../docs/case_studies_explainability.md)
for the full walkthrough.
