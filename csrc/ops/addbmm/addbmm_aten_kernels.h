/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_OPS_ADDBMM_ADDBMM_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_ADDBMM_ADDBMM_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// aten::addbmm: beta * self + alpha * sum_b(batch1[b] @ batch2[b]), where
// self broadcasts to the (n, p) shape of the batched matrix product.
at::Tensor AtenAddbmm(const at::Tensor& self, const at::Tensor& batch1,
                      const at::Tensor& batch2, const at::Scalar& beta,
                      const at::Scalar& alpha);

// aten::addbmm.out
at::Tensor& AtenAddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                          const at::Tensor& batch2, const at::Scalar& beta,
                          const at::Scalar& alpha, at::Tensor& out);

// aten::addbmm_
at::Tensor& AtenAddbmm_(at::Tensor& self, const at::Tensor& batch1,
                        const at::Tensor& batch2, const at::Scalar& beta,
                        const at::Scalar& alpha);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_ADDBMM_ADDBMM_ATEN_KERNELS_H_
