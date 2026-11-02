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

#include "csrc/ops/huber_loss/huber_loss_aten_kernels.h"

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
//   0.5 * d^2                    if |d| < delta,
//   delta * (|d| - 0.5 * delta)  otherwise.
absl::StatusOr<mlir::MlirOp> BuildHuberLossShlo(mlir::MlirOp self_op,
                                                mlir::MlirOp target_op,
                                                int64_t reduction,
                                                double delta) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  const mlir::Type type = GetTensorTypeOrDie(self_b).getElementType();

  // The stablehlo builders take MlirOp& and reject temporaries.
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
  mlir::MlirOp delta_op = MakeConstantLike(diff, delta, type);
  mlir::MlirOp half = MakeConstantLike(diff, 0.5, type);
  mlir::MlirOp half_delta = MakeConstantLike(diff, 0.5 * delta, type);
  mlir::MlirOp sq_diff = mlir::stablehlo::Mul(diff, diff);
  mlir::MlirOp quadratic = mlir::stablehlo::Mul(half, sq_diff);
  mlir::MlirOp excess = mlir::stablehlo::Subtract(abs_diff, half_delta);
  mlir::MlirOp linear = mlir::stablehlo::Mul(delta_op, excess);
  mlir::MlirOp is_quadratic = mlir::stablehlo::Compare(
      abs_diff, delta_op, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp loss = mlir::stablehlo::Select(is_quadratic, quadratic, linear);
  return ReduceLoss(loss, reduction);
}

// With d = self - target, the derivative of the loss with respect to self is
//   -delta  if d <= -delta,
//    delta  if d >= delta,
//    d      otherwise,
// times grad_output, and divided by self.numel() for the Mean reduction. The
// closed comparisons follow the ATen CPU and CUDA kernels. ATen divides by
// self.numel() even when target broadcasts self to more elements.
absl::StatusOr<mlir::MlirOp> BuildHuberLossBackwardShlo(
    mlir::MlirOp grad_output_op, mlir::MlirOp self_op, mlir::MlirOp target_op,
    int64_t reduction, double delta) {
  TT_ASSIGN_OR_RETURN((auto [self_b, target_b]),
                      ApplyBroadcastIfNeeded(self_op, target_op));
  const mlir::Type type = GetTensorTypeOrDie(self_b).getElementType();

  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_b, target_b);
  mlir::MlirOp delta_op = MakeConstantLike(diff, delta, type);
  mlir::MlirOp neg_delta = MakeConstantLike(diff, -delta, type);
  mlir::MlirOp is_upper = mlir::stablehlo::Compare(
      diff, delta_op, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp is_lower = mlir::stablehlo::Compare(
      diff, neg_delta, mlir::stablehlo::ComparisonDirection::LE);
  mlir::MlirOp upper = mlir::stablehlo::Select(is_upper, delta_op, diff);
  mlir::MlirOp dloss = mlir::stablehlo::Select(is_lower, neg_delta, upper);

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

// Validates the arguments of huber_loss and huber_loss.out in the order ATen
// checks them, and returns the promoted dtype of self and target. `out_type`
// is the dtype of the tensor the loss is written to: that of `out`, or of
// `self` for the functional op.
absl::StatusOr<at::ScalarType> ValidateHuberLossArgs(const at::Tensor& self,
                                                     const at::Tensor& target,
                                                     double delta,
                                                     at::ScalarType out_type) {
  TT_RET_CHECK(delta > 0, error::kInvalidArgument)
      << "huber_loss does not support non-positive values for delta.";
  const at::ScalarType result_type = at::result_type(self, target);
  TT_RET_CHECK(c10::canCast(result_type, out_type), error::kInvalidArgument)
      << "result type " << ToString(result_type)
      << " can't be cast to the desired output type " << ToString(out_type);
  // ATen only implements huber_loss for floating-point dtypes.
  TT_RET_CHECK(c10::isFloatingType(result_type),
               error::kPythonNotImplementedError)
      << "\"huber\" not implemented for '" << ToString(result_type) << "'";
  return result_type;
}

NAryMlirOpBuilder<2> MakeHuberLossOpBuilder(int64_t reduction, double delta) {
  return [reduction, delta](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, target_op] = inputs;
    return BuildHuberLossShlo(self_op, target_op, reduction, delta);
  };
}

}  // namespace

at::Tensor AtenHuberLoss(const at::Tensor& self, const at::Tensor& target,
                         int64_t reduction, double delta) {
  TT_KERNEL(OpName::kHuberLoss, param_keys, (self, target, reduction, delta), {
    TT_ASSIGN_OR_THROW(
        const at::ScalarType result_type,
        ValidateHuberLossArgs(self, target, delta, self.scalar_type()));

    TT_ASSIGN_OR_THROW(Dimensions bcast_dims, InferSize(self, target));
    const Dimensions out_dims = (reduction == at::Reduction::None)
                                    ? std::move(bcast_dims)
                                    : Dimensions();
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(
        const mlir::ElementType computation_dtype,
        ConvertTo<mlir::ElementType>(ToAccumulateType(result_type)));

    DispatchOpOptions<1> options = {
        .out_dtype = out_dtype,
        .out_dims = out_dims,
        .computation_dtype = computation_dtype,
        .op_param_cache_keys = std::move(param_keys),
    };
    TT_ASSIGN_OR_THROW(
        auto result_buf,
        (DispatchOp<2>(MakeHuberLossOpBuilder(reduction, delta),
                       {self, target}, std::move(options))));
    return MakeTensor(result_buf);
  });
}

at::Tensor& AtenHuberLossOut(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction, double delta, at::Tensor& out) {
  TT_KERNEL(
      OpName::kHuberLossOut, param_keys, (self, target, reduction, delta, out),
      {
        TT_ASSIGN_OR_THROW(
            const at::ScalarType result_type,
            ValidateHuberLossArgs(self, target, delta, out.scalar_type()));

        TT_ASSIGN_OR_THROW(Dimensions bcast_dims, InferSize(self, target));
        const Dimensions out_dims = (reduction == at::Reduction::None)
                                        ? std::move(bcast_dims)
                                        : Dimensions();
        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out.scalar_type()));
        TT_ASSIGN_OR_THROW(
            const mlir::ElementType computation_dtype,
            ConvertTo<mlir::ElementType>(ToAccumulateType(result_type)));

        DispatchOpOptions<1> options = {
            .out_dtype = out_dtype,
            .out_dims = out_dims,
            .computation_dtype = computation_dtype,
            .op_param_cache_keys = std::move(param_keys),
        };
        TT_THROW_IF_ERROR(
            DispatchOpOut<2>(MakeHuberLossOpBuilder(reduction, delta),
                             {self, target}, out, std::move(options)));
        return out;
      });
}

at::Tensor& AtenHuberLossBackwardOut(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction, double delta,
                                     at::Tensor& grad_input) {
  TT_KERNEL(
      OpName::kHuberLossBackwardOut, param_keys,
      (grad_output, self, target, reduction, delta, grad_input), {
        // ATen computes huber_loss_backward in the dtype of self without
        // promotion, and only implements it for floating-point dtypes.
        TT_CHECK_THROW(target.scalar_type() == self.scalar_type() &&
                           grad_output.scalar_type() == self.scalar_type() &&
                           grad_input.scalar_type() == self.scalar_type(),
                       error::kInvalidArgument)
            << "expected self, target, grad_output and grad_input to have the "
               "same dtype, got "
            << ToString(self.scalar_type()) << ", "
            << ToString(target.scalar_type()) << ", "
            << ToString(grad_output.scalar_type()) << " and "
            << ToString(grad_input.scalar_type());
        TT_CHECK_THROW(self.is_floating_point(),
                       error::kPythonNotImplementedError)
            << "\"huber_backward\" not implemented for '"
            << ToString(self.scalar_type()) << "'";

        TT_ASSIGN_OR_THROW(const Dimensions out_dims,
                           InferSize(grad_output, self, target));
        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        TT_ASSIGN_OR_THROW(const mlir::ElementType computation_dtype,
                           ConvertTo<mlir::ElementType>(
                               ToAccumulateType(self.scalar_type())));

        auto op_builder =
            [reduction, delta](FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [grad_output_op, self_op, target_op] = inputs;
          return BuildHuberLossBackwardShlo(grad_output_op, self_op, target_op,
                                            reduction, delta);
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
