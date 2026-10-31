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
<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Discussion Forum</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch on TPU</b></a>
</p>

## About

TorchTPU is an open-source, native PyTorch backend for Google Cloud TPUs, developed in collaboration between Google and Meta. With TorchTPU, you can run existing PyTorch models on TPU hardware with minimal code changes and scale training and inference workloads from a single chip to multi-slice TPU pods.

## What are you trying to do today?

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Get started**

    Provision environments, understand fundamental architectures, and run your first workloads.

    - [<!--mkdocs::octicons-arrow-right-24:--> Introduction to TorchTPU](./intro_to_torchtpu.py): Why TorchTPU was created, what it provides, and how it can help you.
    - [<!--mkdocs::octicons-arrow-right-24:--> Get started with TorchTPU on a VM](get_started_on_a_vm.py): Setup TorchTPU on a TPU VM.
    - [<!--mkdocs::octicons-arrow-right-24:--> Get started with TorchTPU on GKE](get_started_on_gke.py): Use TorchTPU through GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Supported feature set and roadmap](feature_set.py): Currently supported features and plans for future releases.
    - [<!--mkdocs::octicons-arrow-right-24:--> API reference](api_reference.py): The TorchTPU API and environment variables for the current release.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU FAQ](faq.py): Frequently asked questions about TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> In-depth architecture & core concepts**

    Deep-dive into the mechanics of memory layouts, execution engines, and distributed execution.

    - [<!--mkdocs::octicons-arrow-right-24:--> How TorchTPU works](how_torchtpu_works.py): How TorchTPU integrates with PyTorch and the TPU hardware from top to bottom.
    - [<!--mkdocs::octicons-arrow-right-24:--> The TorchTPU execution model](torchtpu_execution_model.py): In depth detail of how TorchTPU executes PyTorch code and the different execution modes it supports.
    - [<!--mkdocs::octicons-arrow-right-24:--> The TorchTPU memory model](torchtpu_memory_model.py): How TorchTPU handles the way the TPU and XLA compiler deal with memory.
    - [<!--mkdocs::octicons-arrow-right-24:--> Distributed support](distributed_support.py): How TorchTPU supports distributed workloads.
    - [<!--mkdocs::octicons-arrow-right-24:--> Custom kernel support](custom_kernels.py): How to write custom kernels for TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Dynamic shapes](dynamic_shapes.py): Support for dynamic tensor shapes.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Case studies**

    Explore practical examples of how to use TorchTPU to solve real-world problems.

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX Optimization](case_studies_flux.py): Example demonstrating how to identify and apply multiple optimizations on FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B inference](case_studies_muse_glimmer_30b_inference.py): Example of iterative decoding using real weights with static/dynamic KV cache.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B training](case_studies_muse_glimmer_30b_training.py): A training loop on a standalone implementation of Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Using FSDP](case_studies_fsdp.py): The recommended approach to using FSDP for TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Explainability as a performance lens](case_studies_explainability.py): Captum attribution across the execution modes, checked against CPU and profiled with XProf.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analyze your workload**

    - [<!--mkdocs::octicons-arrow-right-24:--> Numerical precision and parity](numerical_precision.py): Controlling precision in TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Profile performance with XProf](performance_profiling_with_xprof.py): How to use XProf as a TorchTPU profiler.

<!--mkdocs:</div>-->
<!-- mdformat on -->
""")
  return


if __name__ == "__main__":
  app.run()
