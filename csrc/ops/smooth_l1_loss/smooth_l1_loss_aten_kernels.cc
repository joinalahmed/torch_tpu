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
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Reduction.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/result_type.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
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

// Applies the loss reduction: the elementwise loss for None, its sum for Sum
// and its sum divided by the element count for Mean.
absl::StatusOr<mlir::MlirOp> ReduceLoss(mlir::MlirOp loss, int64_t reduction) {
  if (reduction == at::Reduction::None) {
    return loss;
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp sum,
                      BuildSumShlo(loss, GetAllDimensions(loss)));
  if (reduction == at::Reduction::Sum) {
    return sum;
  }
  const mlir::Type type = GetTensorTypeOrDie(loss).getElementType();
  mlir::MlirOp numel = GetNumElements(loss, type);
  TT_ASSIGN_OR_RETURN((auto [sum_b, numel_b]),
                      ApplyBroadcastIfNeeded(sum, numel));
  return mlir::stablehlo::Div(sum_b, numel_b);
}

// With d = self - target, the loss of each element is
//   0.5 * d^2 / beta  if |d| < beta,
//   |d| - 0.5 * beta  otherwise.
// For beta == 0 the first branch is never selected and the loss is |d|, the
// L1 loss, as in ATen.
absl::StatusOr<mlir::MlirOp> BuildSmoothL1LossShlo(mlir::MlirOp self_op,
                                                   mlir::MlirOp target_op,
                                                   int64_t reduction,
                                                   double beta) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  const mlir::Type type = GetTensorTypeOrDie(self_b).getElementType();

  // The stablehlo builders take MlirOp& and reject temporaries.
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
  mlir::MlirOp beta_op = MakeConstantLike(diff, beta, type);
  mlir::MlirOp half = MakeConstantLike(diff, 0.5, type);
  mlir::MlirOp half_beta = MakeConstantLike(diff, 0.5 * beta, type);
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(diff, diff);
  mlir::MlirOp half_sq_diff = mlir::stablehlo::Mul(half, sq_diff);
  mlir::MlirOp quadratic = mlir::stablehlo::Div(half_sq_diff, beta_op);
  mlir::MlirOp linear = mlir::stablehlo::Subtract(abs_diff, half_beta);
  mlir::MlirOp is_quadratic = mlir::stablehlo::Compare(
      abs_diff, beta_op, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp loss = mlir::stablehlo::Select(is_quadratic, quadratic, linear);
  return ReduceLoss(loss, reduction);
}

// With d = self - target, the derivative of the loss with respect to self is
//   -1        if d <= -beta,
//    1        if d >= beta,
//    d / beta otherwise,
// times grad_output, and divided by self.numel() for the Mean reduction. The
// closed comparisons follow the ATen CPU and CUDA kernels; with beta == 0 they
// give the derivative of |d| with -1 at d == 0. ATen divides by self.numel()
// even when target broadcasts self to more elements.
absl::StatusOr<mlir::MlirOp> BuildSmoothL1LossBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    int64_t reduction, double beta) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  const mlir::Type type = GetTensorTypeOrDie(self_b).getElementType();

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp beta_op = MakeConstantLike(diff, beta, type);
  mlir::MlirOp neg_beta = MakeConstantLike(diff, -beta, type);
  mlir::MlirOp one = MakeConstantLike(diff, 1.0, type);
  mlir::MlirOp neg_one = MakeConstantLike(diff, -1.0, type);
  mlir::MlirOp scaled_diff = mlir::stablehlo::Div(diff, beta_op);
  mlir::MlirOp is_upper = mlir::stablehlo::Compare(
      diff, beta_op, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp is_lower = mlir::stablehlo::Compare(
      diff, neg_beta, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp upper = mlir::stablehlo::Select(is_upper, one, scaled_diff);
  mlir::MlirOp dloss = mlir::stablehlo::Select(is_lower, neg_one, upper);

  TT_ASSIGN_OR_RETURN((auto [dloss_b, grad_output_b]),
                      ApplyBroadcastIfNeeded(dloss, grad_output_op));
  mlir::MlirOp grad = mlir::stablehlo::Mul(dloss_b, grad_output_b);
  if (reduction != at::Reduction::Mean) {
    return grad;
  }
  mlir::MlirOp numel = GetNumElements(self_op, type);
  TT_ASSIGN_OR_RETURN((auto [grad_b, numel_b]),
                      ApplyBroadcastIfNeeded(grad, numel));
  return mlir::stablehlo::Div(grad_b, numel_b);
}

}  // namespace

at::Tensor& AtenSmoothL1LossOut(const at::Tensor& self,
                                const at::Tensor& target, int64_t reduction,
                                double beta, at::Tensor& out) {
  TT_KERNEL(
      OpName::kSmoothL1LossOut, param_keys,
      (self, target, reduction, beta, out), {
        // ATen only implements smooth_l1_loss for floating-point dtypes.
        const at::ScalarType result_type = at::result_type(self, target);
        TT_CHECK_THROW(c10::isFloatingType(result_type),
                       error::kPythonNotImplementedError)
            << "\"smooth_l1\" not implemented for '" << ToString(result_type)
            << "'";
        TT_CHECK_THROW(beta >= 0, error::kInvalidArgument)
            << "smooth_l1_loss does not support negative values for beta.";
        TT_THROW_IF_ERROR(ValidateOutDtype(out, result_type));

        TT_ASSIGN_OR_THROW(Dimensions bcast_dims, InferSize(self, target));
        const Dimensions out_dims = (reduction == at::Reduction::None)
                                        ? std::move(bcast_dims)
                                        : Dimensions();
        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType computation_dtype,
            ConvertTo<mlir::ElementType>(ToAccumulateType(result_type)));

        auto op_builder =
            [reduction, beta](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [self_op, target_op] = inputs;
          return BuildSmoothL1LossShlo(self_op, target_op, reduction, beta);
        };
        DispatchOpOptions<1> options = {
            .out_dtype = out_dtype,
            .out_dims = out_dims,
            .computation_dtype = computation_dtype,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_THROW_IF_ERROR(DispatchOpOut<2>(std::move(op_builder),
                                           {self, target}, out,
                                           std::move(options)));
        return out;
      });
}

at::Tensor& AtenSmoothL1LossBackwardGradInput(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction, double beta,
                                              at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kSmoothL1LossBackwardGradInput, param_keys,
      (grad_output, self, target, reduction, beta, grad_input), {
        // ATen promotes the inputs to a common dtype and computes in it, for
        // integral dtypes too. TPU only supports the floating-point ones.
        const at::ScalarType result_type = c10::promoteTypes(
            at::result_type(grad_output, self), target.scalar_type());
        TT_CHECK_THROW(c10::isFloatingType(result_type),
                       error::kPythonNotImplementedError)
            << "\"smooth_l1_backward\" not implemented for '"
            << ToString(result_type) << "'";
        TT_THROW_IF_ERROR(ValidateOutDtype(grad_input, result_type));

        TT_ASSIGN_OR_THROW(const Dimensions out_dims,
                           InferSize(grad_output, self, target));
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType out_dtype,
            ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType computation_dtype,
            ConvertTo<mlir::ElementType>(ToAccumulateType(result_type)));

        auto op_builder =
            [reduction, beta](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, self_op, target_op] = inputs;
          return BuildSmoothL1LossBackwardShlo(grad_output_op, self_op,
                                               target_op, reduction, beta);
        };
        DispatchOpOptions<1> options = {
            .out_dtype = out_dtype,
            .out_dims = out_dims,
            .computation_dtype = computation_dtype,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_THROW_IF_ERROR(DispatchOpOut<3>(std::move(op_builder),
                                           {grad_output, self, target},
                                           grad_input, std::move(options)));
        return grad_input;
      });
}

}  // namespace torch_tpu
