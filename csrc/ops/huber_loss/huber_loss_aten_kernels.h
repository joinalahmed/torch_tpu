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

#ifndef TORCH_TPU_CSRC_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_

#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"

namespace torch_tpu {

// aten::huber_loss.
at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta);

// aten::huber_loss.out.
at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out);

// aten::huber_loss_backward.out. The functional aten::huber_loss_backward is
// a CompositeExplicitAutograd op that calls this.
at::Tensor& AtenHuberLossBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction, double delta,
                                     at::Tensor& grad_input);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_HUBER_LOSS_HUBER_LOSS_ATEN_KERNELS_H_
