# Copyright 2025 Google LLC
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

"""Macro phase bzl code for pip_parse customizations."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

_HEADER_EXTENSIONS = [
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".cuh",
    ".inc",
    ".inl",
    ".tcc",
]

def define_extra_torch_targets():
    """Defines additional targets for the PyTorch distribution inside site-packages."""

    # Each include root of the wheel is exposed through strip_include_prefix,
    # so the compiler searches a virtual include tree that holds only the
    # declared headers. The wheel bundles an old copy of the protobuf headers
    # under include/google/protobuf; leaving them out of hdrs then keeps them
    # out of the include path in every execution strategy, sandboxed or not.
    # With `includes` the compiler would search the real directory, and an
    # unsandboxed build would pick up torch's protobuf headers ahead of
    # @com_google_protobuf.
    _TORCH_INCLUDE_ROOTS = {
        "torch_headers_include": "site-packages/torch/include",
        "torch_headers_api_include": (
            "site-packages/torch/include/torch/csrc/api/include"
        ),
        "torch_headers_kineto_include": "site-packages/torch/include/kineto",
    }
    for target_name, include_root in _TORCH_INCLUDE_ROOTS.items():
        cc_library(
            # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
            name = target_name,
            hdrs = native.glob(
                include = [
                    include_root + "/**/*" + extension
                    for extension in _HEADER_EXTENSIONS
                ],
                exclude = [
                    "site-packages/torch/include/google/protobuf/**",
                ],
                allow_empty = True,
            ),
            strip_include_prefix = include_root,
        )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "torch_headers",
        deps = [":" + target_name for target_name in _TORCH_INCLUDE_ROOTS],
        visibility = ["//visibility:public"],
    )

    # This is a subset of all the `.so` files bundled into the PyTorch wheel.
    # More libraries can be added here as necessary.
    standard_libs = [
        "libc10",
        "libtorch_cpu",
        "libtorch_python",
        "libshm",
    ]

    for lib_name in standard_libs:
        cc_library(
            # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
            name = lib_name,
            srcs = native.glob([
                "site-packages/torch/lib/{}.so".format(lib_name),
            ]),
            visibility = ["//visibility:public"],
        )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "libtorch",
        srcs = native.glob([
            "site-packages/torch/lib/libtorch.so",
        ]),
        deps = [
            ":libc10",
            ":libtorch_cpu",
            ":libshm",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "libtorch_global_deps",
        deps = [
            ":libc10",
            ":libtorch_cpu",
        ],
        visibility = ["//visibility:public"],
    )

    cc_library(
        # ALLOW_CC_TARGETS=Defines external PyTorch wheel targets in OSS Bazel build
        name = "torch_libs",
        deps = [
            ":libtorch",
            ":libtorch_python",
            ":libtorch_global_deps",
        ],
        visibility = ["//visibility:public"],
    )
