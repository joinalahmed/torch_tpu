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

"""Captum attribution on TPU, checked against CPU and timed.

Runs Captum's Integrated Gradients on a TPU in the current execution mode,
compares the attributions against the same computation on CPU, reports the
warm wall-clock time, and optionally writes an XProf trace.

The execution mode is chosen before `torch_tpu` is imported, so one process
runs one mode:

  python3 captum_attribution.py                          # strict eager
  TORCH_TPU_DEFER_AND_FUSE=1 python3 captum_attribution.py   # DeferAndFuse
  python3 captum_attribution.py --compile                # torch.compile

`--model tiny` (default) attributes a small MLP and needs no downloads.
`--model Qwen/Qwen3-0.6B` (or any Hugging Face causal LM) attributes the
next-token logit to the input embeddings with LayerIntegratedGradients.
"""

import argparse
import os
import pathlib
import statistics
import time

import torch
from torch import nn

_TPU = torch.device("tpu")


def _parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", default="tiny")
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--runs", type=int, default=3)
  parser.add_argument(
      "--precision",
      default="highest",
      choices=["highest", "high", "medium"],
      help="torch.set_float32_matmul_precision() value. 'highest' matches CPU "
      "numerics; the TPU default rounds float32 matmul inputs to bfloat16.",
  )
  parser.add_argument(
      "--compile",
      action="store_true",
      help="Wrap the model forward in torch.compile(backend='tpu').",
  )
  parser.add_argument(
      "--internal-batch-size",
      type=int,
      default=None,
      help="Chunk the integration steps to bound peak HBM.",
  )
  parser.add_argument(
      "--profile-dir",
      default=None,
      help="Write an XProf trace of the warm runs to this directory.",
  )
  return parser.parse_args()


class _TinyMlp(nn.Module):

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3)
    )

  def forward(self, x):
    return self.net(x)


def _tiny_attribution(device, steps, internal_batch_size, compile_model):
  """Integrated Gradients on a small MLP, class 1 of a 4-sample batch."""
  from captum.attr import IntegratedGradients  # pylint: disable=g-import-not-at-top

  torch.manual_seed(0)
  model = _TinyMlp().eval()
  inputs = torch.randn(4, 16)
  model, inputs = model.to(device), inputs.to(device)
  forward = torch.compile(model, backend="tpu") if compile_model else model
  ig = IntegratedGradients(forward)

  def run():
    return ig.attribute(
        inputs,
        target=1,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    ).cpu()

  return run


def _llm_attribution(name, device, steps, internal_batch_size, compile_model):
  """LayerIntegratedGradients of the next-token logit wrt input embeddings."""
  # pylint: disable=g-import-not-at-top
  from captum.attr import LayerIntegratedGradients
  from transformers import AutoModelForCausalLM, AutoTokenizer
  # pylint: enable=g-import-not-at-top

  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModelForCausalLM.from_pretrained(
      name, dtype=torch.float32, device_map=str(device)
  ).eval()
  forward = torch.compile(model, backend="tpu") if compile_model else model
  prompt = (
      "Integrated Gradients attributes a prediction to its inputs by "
      "integrating gradients along a path from a baseline to the input. On a "
      "TPU the question is whether one fixed-shape program beats a loop. The "
      "answer is"
  )
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  print(f"prompt tokens: {input_ids.shape[1]}")
  target = tokenizer(" yes").input_ids[0]
  pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  baselines = torch.full_like(input_ids, pad)

  def logit(ids):
    # `logits_to_keep=1` keeps only the last position, so the [steps, T, V]
    # logits tensor is never materialized.
    return forward(input_ids=ids, logits_to_keep=1).logits[:, -1, target]

  lig = LayerIntegratedGradients(logit, model.get_input_embeddings())

  def run():
    return lig.attribute(
        input_ids,
        baselines=baselines,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    ).cpu()

  return run


def _build(args, device, compile_model):
  if args.model == "tiny":
    return _tiny_attribution(
        device, args.steps, args.internal_batch_size, compile_model
    )
  return _llm_attribution(
      args.model, device, args.steps, args.internal_batch_size, compile_model
  )


def _time(run, runs):
  """Returns (first-call seconds, median warm seconds, last result)."""
  start = time.perf_counter()
  result = run()
  torch.accelerator.synchronize()
  first = time.perf_counter() - start
  warm = []
  for _ in range(runs):
    start = time.perf_counter()
    result = run()
    torch.accelerator.synchronize()
    warm.append(time.perf_counter() - start)
  return first, statistics.median(warm), result


def _profile(run, runs, profile_dir):
  """Writes an XProf trace of `runs` warm calls to `profile_dir`."""
  from torch.tpu.profiler import TpuProfilerConfig  # pylint: disable=g-import-not-at-top

  log_dir = pathlib.Path(profile_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  config = TpuProfilerConfig(
      host_tracer_level=2, device_tracer_level=1, run_dir=log_dir
  )
  with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.TPU,
      ],
      experimental_config=config,
      on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
  ) as profiler:
    for _ in range(runs):
      run()
      torch.accelerator.synchronize()
      profiler.step()
  print(f"XProf trace written to {log_dir}; open with: xprof --logdir {log_dir}")
  print(
      profiler.key_averages().table(
          sort_by="self_cpu_time_total", row_limit=10
      )
  )


def main():
  args = _parse_args()
  torch.set_float32_matmul_precision(args.precision)
  mode = "compile" if args.compile else (
      "defer_and_fuse"
      if os.environ.get("TORCH_TPU_DEFER_AND_FUSE", "0") == "1"
      else "strict"
  )
  print(
      f"model={args.model} mode={mode} precision={args.precision} "
      f"steps={args.steps}"
  )

  start = time.perf_counter()
  reference = _build(args, torch.device("cpu"), compile_model=False)()
  cpu_seconds = time.perf_counter() - start
  run = _build(args, _TPU, compile_model=args.compile)
  first, warm, result = _time(run, args.runs)

  denominator = reference.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
  rel_err = ((result - reference).abs().max() / denominator).item()
  print(f"CPU reference (one run):           {cpu_seconds:.3f} s")
  print(f"first call (includes compilation): {first:.3f} s")
  print(f"warm median of {args.runs} runs:    {warm:.3f} s")
  print(f"max relative error vs CPU:         {rel_err:.2e}")
  if args.model != "tiny":
    print(
        f"peak HBM allocated: {torch.accelerator.max_memory_allocated() / 2**30:.1f} GiB"
    )

  if args.profile_dir:
    _profile(run, args.runs, args.profile_dir)


if __name__ == "__main__":
  main()
