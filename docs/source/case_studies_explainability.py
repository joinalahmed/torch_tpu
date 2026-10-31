# Copyright 2026 Google LLC
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

# pylint: skip-file
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Explainability as a performance lens

    Attribution methods such as Integrated Gradients run many forward and
    backward passes of a model with host-side scalars in the loop, which makes
    them a compact stress test for the TorchTPU execution modes. This guide runs
    [Captum](https://captum.ai) unmodified on a TPU, checks the numbers against
    CPU, compares strict eager, DeferAndFuse, and `torch.compile`, and shows
    what each mode looks like in XProf.

    The runnable script is
    [`examples/explainability/captum_attribution.py`](../examples/explainability).

    In this guide, you will learn:

    1.  **Run Captum on a TPU** with no changes to Captum.
    2.  **Set the matmul precision** so attributions match CPU.
    3.  **Compare the execution modes** on a Hugging Face model.
    4.  **Read the XProf trace** to see where strict eager mode loses time.
    5.  **Bound peak memory** for many integration steps.
    6.  **Check convergence** before trusting an attribution.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. Run Captum on a TPU

    Captum's attribution methods are plain PyTorch autograd: forward hooks,
    `torch.autograd.grad`, and tensor arithmetic. They work on TorchTPU the same
    way they work on CUDA. The following code attributes the next-token logit of
    a causal language model to its input embeddings with
    `LayerIntegratedGradients`:

    ```python
    import torch
    from captum.attr import LayerIntegratedGradients
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    device = torch.device("tpu")

    name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, device_map="tpu"
    ).eval()

    input_ids = tokenizer("The capital of France is", return_tensors="pt")
    input_ids = input_ids.input_ids.to(device)
    target = tokenizer(" Paris").input_ids[0]
    baselines = torch.full_like(input_ids, tokenizer.pad_token_id)


    def logit(ids):
      # Keep only the last position so the [steps, T, V] logits tensor is never
      # materialized.
      return model(input_ids=ids, logits_to_keep=1).logits[:, -1, target]


    lig = LayerIntegratedGradients(logit, model.get_input_embeddings())
    attributions, delta = lig.attribute(
        input_ids, baselines=baselines, n_steps=64, return_convergence_delta=True
    )
    print(attributions.sum(-1).cpu(), delta.cpu())
    ```

    Captum expands the input to `n_steps` interpolated copies and runs one
    batched forward and backward pass, so the TPU sees a fixed-shape program
    regardless of the number of steps. `device_map="tpu"` loads the weights
    directly onto the device. Every public `captum.attr` method, including the
    `LLMAttribution` and `LLMGradientAttribution` wrappers, runs on TorchTPU
    0.2.0; the deterministic ones match CPU to `1e-7` relative error on small
    models once the matmul precision is set as described next.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. Set the matmul precision

    By default, TorchTPU runs `float32` matmuls with `bfloat16` inputs on the
    MXU (see
    [Manage numerical precision in TorchTPU](numerical_precision.py)). Model
    outputs tolerate this well; attributions, which are differences of gradients,
    do not. Maximum relative error of the TPU attribution against the CPU
    attribution, same weights and inputs, small MLP and CNN:

    | Method | Default precision | `torch.set_float32_matmul_precision("highest")` |
    | :----- | ----------------: | ----------------------------------------------: |
    | Saliency | 2.2e-3 | 1.7e-7 |
    | InputXGradient | 1.7e-3 | 7.0e-8 |
    | IntegratedGradients (50 steps) | 2.1e-3 | 4.4e-8 |
    | LayerIntegratedGradients | 3.1e-3 | 1.0e-7 |
    | LayerGradCam | 1.0e-2 | 2.2e-7 |

    Call `torch.set_float32_matmul_precision("highest")` before attributing. On
    Qwen3-0.6B the `highest` setting costs nothing measurable for attribution:
    the warm times in the next section are the same at `high` and `highest`.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Compare the execution modes

    TorchTPU picks its execution mode when it is imported (see
    [The TorchTPU execution model](torchtpu_execution_model.py)), so the example
    script runs one mode per process:

    ```bash
    python3 captum_attribution.py --model Qwen/Qwen3-0.6B
    TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py --model Qwen/Qwen3-0.6B
    python3 captum_attribution.py --model Qwen/Qwen3-0.6B --compile
    ```

    `LayerIntegratedGradients` on Qwen3-0.6B, 64 steps, `float32`, one TPU v6e
    chip, median of three warm runs:

    | Mode | Warm time | First call | Max relative error vs CPU |
    | :--- | --------: | ---------: | ------------------------: |
    | Strict eager | 0.80 s | 5 s | 9.2e-05 |
    | DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`) | 0.36 s | 43 s | 9.3e-05 |
    | `torch.compile(backend="tpu")` | 0.16 s | 130 s | 7.0e-05 |

    The CPU reference for the same attribution takes several seconds, so even
    strict eager mode is a large speedup, and the gap widens with more steps:
    at 128 steps and a 53-token prompt, Captum on CPU takes 17 s, strict eager
    1.1 s, and DeferAndFuse 0.45 s.

    What the table shows:

    *   **Strict eager mode** runs each PyTorch op as its own executable. Captum
        issues hundreds of small ops per attribution, so the TPU waits on the
        host between them.
    *   **DeferAndFuse** keeps the Python code unchanged but fuses the deferred
        ops into larger executables. The one-time cost is the fusion compile on
        the first call.
    *   **`torch.compile`** compiles the model forward into one program. Captum
        still runs its own glue eagerly around it, so the attribution does not
        become a single program, but the model is where the time goes.

    > **Note:** The compiled row needs two fixes that landed after TorchTPU
    > 0.2.0 and Captum 0.9.0: Captum passed a CPU index tensor into the compiled
    > forward
    > ([meta-pytorch/captum#1952](https://github.com/meta-pytorch/captum/pull/1952)),
    > and TorchTPU compiled a CPU view argument against the wrong layout
    > ([google-pytorch/torch_tpu#6106](https://github.com/google-pytorch/torch_tpu/pull/6106)).
    > Without them, the compiled Captum forward fails with `expected tensor to
    > be on tpu, got cpu` or returns zero attributions for part of the input.
    > Compiled mode needs every tensor argument on the device; see the
    > pitfalls in [The TorchTPU execution model](torchtpu_execution_model.py).
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 4. Read the XProf trace

    Pass `--profile-dir` to the example script to capture an XProf trace of the
    warm runs with `torch.profiler`, as described in
    [Profile performance with XProf](performance_profiling_with_xprof.py):

    ```bash
    python3 captum_attribution.py --model Qwen/Qwen3-0.6B --profile-dir ~/prof/strict
    TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py --model Qwen/Qwen3-0.6B --profile-dir ~/prof/daf
    xprof --logdir ~/prof --port 6006
    ```

    The operator table that the script prints is enough to see the difference.
    For 32-step `LayerIntegratedGradients` on Qwen3-0.6B:

    | Mode | Self CPU total | `aten::_to_copy` | Mean per copy |
    | :--- | -------------: | ---------------: | ------------: |
    | Strict eager | 1.93 s | 1.35 s (70%) | 2.6 ms |
    | DeferAndFuse | 0.78 s | 0.29 s | 0.57 ms |

    Captum moves step sizes, targets, and small index tensors between host and
    device on every call. In strict eager mode each `aten::_to_copy` waits for
    the device to drain, which is the 2.6 ms per copy. DeferAndFuse issues the
    same copies but overlaps them with the fused executables. In the trace
    viewer, the strict eager timeline shows the TPU idle between short
    executables; the DeferAndFuse timeline shows fewer, longer executables.

    This pattern is not specific to Captum. Any loop that mixes small
    host-side tensors into device work shows the same `_to_copy` wall in strict
    eager mode, and DeferAndFuse is the first thing to try.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 5. Bound peak memory

    Integrated Gradients multiplies the batch by `n_steps`. Two settings keep
    that within HBM:

    *   **`logits_to_keep=1`** in the model forward (Transformers v5) returns
        only the last position's logits. Without it, a 128-step attribution of a
        53-token prompt materializes a `[128, 53, 151936]` `float32` logits
        tensor (3.9 GiB) alongside the interpolated embeddings with gradients,
        and fails with `Attempting to allocate 3.84G` on a 32 GiB TPU v6e.
    *   **`internal_batch_size`** in `attribute()` chunks the steps. With
        `logits_to_keep=1`, 128 steps of Qwen3-0.6B fit in one batch on a v6e
        chip (peak 23.9 GiB); 256 steps do not, so pass
        `internal_batch_size=128`.

    `torch.accelerator.max_memory_allocated()` reports the peak after the call.
    Ops run asynchronously in every execution mode, so call
    `torch.accelerator.synchronize()` first.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 6. Check convergence

    A fast attribution is not a correct one. Integrated Gradients approximates
    a path integral, and `return_convergence_delta=True` reports how far the sum
    of the attributions is from the difference of the model outputs at the input
    and the baseline. On Qwen3-0.6B with a raw logit target and a pad-token
    baseline, the delta stays at roughly half the total attribution even at 256
    steps, on CPU and TPU alike. That is a property of the setup, not of the
    device: prefer a log-softmax target, consider a zero-embedding baseline, and
    print the delta.

    The TPU and CPU attributions agree to `1e-4` or better in every mode, so
    the device is not a variable in this check. Run the comparison once on CPU
    at a small step count, then move to the TPU for the step counts you need.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 7. Run the example

    [`examples/explainability`](../examples/explainability) contains the script
    and its README:

    | Command | What it checks |
    | :------ | :------------- |
    | `python3 captum_attribution.py` | `IntegratedGradients` on a small MLP matches CPU to `1e-7`. No downloads. |
    | `python3 captum_attribution.py --model Qwen/Qwen3-0.6B` | `LayerIntegratedGradients` on a Hugging Face model matches CPU, with warm time and peak HBM. |
    | `TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py ...` | The same attribution in optimized eager mode. |
    | `python3 captum_attribution.py ... --compile` | The same attribution with the model under `torch.compile(backend="tpu")`. |
    | `python3 captum_attribution.py ... --profile-dir DIR` | Writes an XProf trace and prints the operator table. |
    """)
  return


if __name__ == "__main__":
  app.run()
