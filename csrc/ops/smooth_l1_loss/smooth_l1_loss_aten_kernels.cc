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

#include "csrc/ops/smooth_l1_loss/smooth_l1_loss_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Smooth L1 with threshold `beta` is 0.5 * d^2 / beta inside the threshold and
// |d| - 0.5 * beta outside; Huber with threshold `delta` is 0.5 * d^2 inside
// and delta * (|d| - 0.5 * delta) outside, i.e. delta * smooth_l1(delta).
enum class LossKind { kSmoothL1, kHuber };

absl::StatusOr<mlir::MlirOp> ReduceLoss(mlir::MlirOp loss, int64_t reduction) {
  if (reduction == at::Reduction::None) {
    return loss;
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp reduced,
                      BuildSumShlo(loss, GetAllDimensions(loss)));
  if (reduction == at::Reduction::Sum) {
    return reduced;
  }
  const mlir::RankedTensorType loss_type = GetTensorTypeOrDie(loss);
  mlir::MlirOp numel = GetNumElements(loss, loss_type.getElementType());
  TT_ASSIGN_OR_RETURN((auto [reduced_b, numel_b]),
                      ApplyBroadcastIfNeeded(reduced, numel));
  return mlir::stablehlo::Div(reduced_b, numel_b);
}

absl::StatusOr<mlir::MlirOp> BuildPiecewiseLossShlo(mlir::MlirOp self_op,
                                                    mlir::MlirOp target_op,
                                                    int64_t reduction,
                                                    double threshold,
                                                    LossKind kind) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
  if (kind == LossKind::kSmoothL1 && threshold == 0.0) {
    // ATen treats beta == 0 as L1 loss.
    return ReduceLoss(abs_diff, reduction);
  }

  const mlir::Type type = GetTensorTypeOrDie(diff).getElementType();
  mlir::MlirOp thr = MakeConstantLike(diff, threshold, type);
  mlir::MlirOp half = MakeConstantLike(diff, 0.5, type);
  mlir::MlirOp sq = mlir::stablehlo::Mul(diff, diff);
  mlir::MlirOp half_sq = mlir::stablehlo::Mul(half, sq);
  mlir::MlirOp half_thr = mlir::stablehlo::Mul(half, thr);
  mlir::MlirOp beyond = mlir::stablehlo::Subtract(abs_diff, half_thr);
  mlir::MlirOp inside;
  mlir::MlirOp outside;
  if (kind == LossKind::kSmoothL1) {
    inside = mlir::stablehlo::Div(half_sq, thr);
    outside = beyond;
  } else {
    inside = half_sq;
    outside = mlir::stablehlo::Mul(thr, beyond);
  }
  mlir::MlirOp is_inside = mlir::stablehlo::Compare(
      abs_diff, thr, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp loss = mlir::stablehlo::Select(is_inside, inside, outside);
  return ReduceLoss(loss, reduction);
}

// d(loss)/d(self) * grad_output. For smooth L1 the derivative is d / beta
// inside the threshold and sign(d) outside; for Huber it is d inside and
// delta * sign(d) outside. The mean reduction scales by 1 / numel.
absl::StatusOr<mlir::MlirOp> BuildPiecewiseLossBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    int64_t reduction, double threshold, LossKind kind) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  mlir::MlirBuilder& builder = self_b.getBuilder();
  const mlir::Type type = GetTensorTypeOrDie(self_b).getElementType();

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp sign = mlir::stablehlo::Sign(diff);
  mlir::MlirOp grad;
  if (kind == LossKind::kSmoothL1 && threshold == 0.0) {
    grad = sign;
  } else {
    mlir::MlirOp thr = MakeConstantLike(diff, threshold, type);
    mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
    mlir::MlirOp is_inside = mlir::stablehlo::Compare(
        abs_diff, thr, mlir::stablehlo::ComparisonDirection::LT);
    mlir::MlirOp inside;
    mlir::MlirOp outside;
    if (kind == LossKind::kSmoothL1) {
      inside = mlir::stablehlo::Div(diff, thr);
      outside = sign;
    } else {
      inside = diff;
      outside = mlir::stablehlo::Mul(thr, sign);
    }
    grad = mlir::stablehlo::Select(is_inside, inside, outside);
  }

  if (reduction == at::Reduction::Mean) {
    // 1 / numel in at least f32, as in mse_loss_backward.
    const mlir::Type norm_type =
        type.isF64() ? type : builder.getOpBuilder().getF32Type();
    mlir::MlirOp one = MakeScalarConstant(builder, 1.0, norm_type);
    mlir::MlirOp numel = GetNumElements(self_b, norm_type);
    mlir::MlirOp norm = mlir::stablehlo::Div(one, numel);
    if (norm_type != type) {
      norm = mlir::stablehlo::ConvertElementType(norm, type);
    }
    TT_ASSIGN_OR_RETURN((auto [norm_b, grad_b]),
                        ApplyBroadcastIfNeeded(norm, grad));
    grad = mlir::stablehlo::Mul(norm_b, grad_b);
  }
  TT_ASSIGN_OR_RETURN((auto [grad_b, grad_out_b]),
                      ApplyBroadcastIfNeeded(grad, grad_output_op));
  return mlir::stablehlo::Mul(grad_b, grad_out_b);
}

absl::Status ValidateLossInputs(const at::Tensor& self,
                                const at::Tensor& target, int64_t reduction,
                                double threshold, LossKind kind) {
  TT_RET_CHECK(self.is_floating_point(), error::kInvalidArgument)
      << "expected the input dtype to be a floating-point type, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(self.scalar_type() == target.scalar_type(),
               error::kInvalidArgument)
      << "expected input and target to have the same dtype, got "
      << ToString(self.scalar_type()) << " and "
      << ToString(target.scalar_type());
  TT_RET_CHECK(reduction == at::Reduction::None ||
                   reduction == at::Reduction::Mean ||
                   reduction == at::Reduction::Sum,
               error::kInvalidArgument)
      << "unrecognized reduction mode " << reduction;
  if (kind == LossKind::kSmoothL1) {
    TT_RET_CHECK(threshold >= 0, error::kInvalidArgument)
        << "smooth_l1_loss does not support negative values for beta.";
  } else {
    TT_RET_CHECK(threshold > 0, error::kInvalidArgument)
        << "huber_loss does not support non-positive values for delta.";
  }
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> PiecewiseLoss(
    const at::Tensor& self, const at::Tensor& target, int64_t reduction,
    double threshold, LossKind kind, OpParamCacheKeys param_keys,
    std::optional<std::reference_wrapper<at::Tensor>> out = std::nullopt) {
  TT_RETURN_IF_ERROR(
      ValidateLossInputs(self, target, reduction, threshold, kind));
  TT_ASSIGN_OR_RETURN(Dimensions bcast_shape, InferSize(self, target));
  Dimensions out_dims = (reduction == at::Reduction::None)
                            ? std::move(bcast_shape)
                            : Dimensions();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(
                          out.has_value() ? out->get().scalar_type()
                                          : self.scalar_type()));
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType computation_dtype,
      ConvertTo<mlir::ElementType>(ToAccumulateType(self.scalar_type())));

  auto op_builder = [reduction, threshold,
                     kind](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, target_op] = inputs;
    return BuildPiecewiseLossShlo(self_op, target_op, reduction, threshold,
                                  kind);
  };
  DispatchOpOptions<1> options = {
      .out_dtype = out_dtype,
      .out_dims = out_dims,
      .computation_dtype = computation_dtype,
      .op_param_cache_keys = std::move(param_keys),
  };
  if (out.has_value()) {
    TT_RETURN_IF_ERROR(DispatchOpOut<2>(std::move(op_builder), {self, target},
                                        out->get(), std::move(options)));
    return out->get();
  }
  TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                      DispatchOp<2>(std::move(op_builder), {self, target},
                                    std::move(options)));
  return MakeTensor(buffer);
}

absl::Status PiecewiseLossBackwardOut(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Tensor& target,
                                      int64_t reduction, double threshold,
                                      LossKind kind, at::Tensor& grad_input,
                                      OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(
      ValidateLossInputs(self, target, reduction, threshold, kind));
  TT_ASSIGN_OR_RETURN(Dimensions out_dims,
                      InferSize(self, target, grad_output));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));

  auto op_builder = [reduction, threshold,
                     kind](FixedSizeSpan<mlir::MlirOp, 3> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [grad_output_op, self_op, target_op] = inputs;
    return BuildPiecewiseLossBackwardShlo(grad_output_op, self_op, target_op,
                                          reduction, threshold, kind);
  };
  // Pointwise, so it runs in the input dtype like mse_loss_backward.
  return DispatchOpOut<3>(std::move(op_builder), {grad_output, self, target},
                          grad_input,
                          {.out_dtype = out_dtype,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenSmoothL1LossOut(const at::Tensor& self,
                                const at::Tensor& target, int64_t reduction,
                                double beta, at::Tensor& out) {
  TT_KERNEL(OpName::kSmoothL1LossOut, param_keys,
            (self, target, reduction, beta, out), {
              TT_THROW_IF_ERROR(PiecewiseLoss(self, target, reduction, beta,
                                              LossKind::kSmoothL1,
                                              std::move(param_keys), out)
                                    .status());
              return out;
            });
}

at::Tensor& AtenSmoothL1LossBackwardGradInput(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction, double beta,
                                              at::Tensor& grad_input) {
  TT_KERNEL(OpName::kSmoothL1LossBackwardGradInput, param_keys,
            (grad_output, self, target, reduction, beta, grad_input), {
              TT_THROW_IF_ERROR(PiecewiseLossBackwardOut(
                  grad_output, self, target, reduction, beta,
                  LossKind::kSmoothL1, grad_input, std::move(param_keys)));
              return grad_input;
            });
}

at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta) {
  TT_KERNEL(OpName::kHuberLoss, param_keys, (self, target, reduction, delta), {
    TT_ASSIGN_OR_THROW(at::Tensor result,
                       PiecewiseLoss(self, target, reduction, delta,
                                     LossKind::kHuber, std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out) {
  TT_KERNEL(OpName::kHuberLossOut, param_keys,
            (self, target, reduction, delta, out), {
              TT_THROW_IF_ERROR(PiecewiseLoss(self, target, reduction, delta,
                                              LossKind::kHuber,
                                              std::move(param_keys), out)
                                    .status());
              return out;
            });
}

at::Tensor& AtenHuberLossBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction, double delta,
                                     at::Tensor& grad_input) {
  TT_KERNEL(OpName::kHuberLossBackwardOut, param_keys,
            (grad_output, self, target, reduction, delta, grad_input), {
              TT_THROW_IF_ERROR(PiecewiseLossBackwardOut(
                  grad_output, self, target, reduction, delta,
                  LossKind::kHuber, grad_input, std::move(param_keys)));
              return grad_input;
            });
}

}  // namespace torch_tpu
