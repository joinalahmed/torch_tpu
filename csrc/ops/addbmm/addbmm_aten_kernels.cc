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

#include "csrc/ops/addbmm/addbmm_aten_kernels.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/bmm/bmm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::Status ValidateAddbmmOut(const at::Tensor& out,
                               at::ScalarType expected_dtype) {
  TT_RET_CHECK(out.scalar_type() == expected_dtype, error::kInvalidArgument)
      << "expected out tensor to have dtype " << ToString(expected_dtype)
      << ", got " << ToString(out.scalar_type());
  return absl::OkStatus();
}

absl::Status ValidateAddbmmInputs(const at::Tensor& self,
                                  const at::Tensor& batch1,
                                  const at::Tensor& batch2) {
  TT_RET_CHECK(batch1.numel() == 0 || batch2.numel() == 0 ||
                   c10::isFloatingType(batch1.scalar_type()) ||
                   c10::isComplexType(batch1.scalar_type()),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(batch1.scalar_type());

  TT_RET_CHECK(self.scalar_type() == batch1.scalar_type() &&
                   batch1.scalar_type() == batch2.scalar_type(),
               error::kInvalidArgument)
      << "expected input dtypes to be the same, got: self="
      << ToString(self.scalar_type())
      << ", batch1=" << ToString(batch1.scalar_type())
      << ", batch2=" << ToString(batch2.scalar_type());

  TT_RET_CHECK(batch1.dim() == 3, error::kInvalidArgument)
      << "expected batch1 to be a 3D tensor (batch of matrices), got "
      << batch1.dim() << "D";
  TT_RET_CHECK(batch2.dim() == 3, error::kInvalidArgument)
      << "expected batch2 to be a 3D tensor (batch of matrices), got "
      << batch2.dim() << "D";

  TT_RET_CHECK(batch1.size(0) == batch2.size(0), error::kInvalidArgument)
      << "expected the batch dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the batch dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(0) << " vs "
      << batch2.size(0);
  TT_RET_CHECK(batch1.size(2) == batch2.size(1), error::kInvalidArgument)
      << "expected the last dimension of the first argument (of shape "
      << ToString(batch1.sizes())
      << ") to match the second dimension of the second argument (of shape "
      << ToString(batch2.sizes()) << "), got " << batch1.size(2) << " vs "
      << batch2.size(1);

  // Unlike baddbmm, self is not batched: it broadcasts to the (n, p) result
  // of the batch-reduced matrix product.
  const Dimensions out_dims = {batch1.size(1), batch2.size(2)};
  TT_RET_CHECK(self.dim() <= 2, error::kInvalidArgument)
      << "expected input to have at most 2 dimensions, got " << self.dim()
      << "D";
  absl::StatusOr<Dimensions> broadcast_dims = InferSize(self.sizes(), out_dims);
  TT_RET_CHECK(broadcast_dims.ok() && *broadcast_dims == out_dims,
               error::kInvalidArgument)
      << "expected input (of shape " << ToString(self.sizes())
      << ") to be broadcastable to the matrix product shape "
      << ToString(out_dims);

  TT_RETURN_IF_ERROR(ValidateNotLowPrecisionFloat(batch1));

  return absl::OkStatus();
}

// Builds beta * self + alpha * sum_b(batch1[b] @ batch2[b]).
//
// The product term is left out of the graph when alpha is zero or there are
// no batches, and the self term when beta is zero, so that NaN or Inf in the
// skipped operand does not reach the result. This matches ATen, which reduces
// an empty batch to beta * self and treats beta == 0 as 0 regardless of self.
absl::StatusOr<mlir::MlirOp> BuildAddbmmShlo(
    std::optional<mlir::MlirOp> self_op, mlir::MlirOp batch1_op,
    mlir::MlirOp batch2_op, std::optional<mlir::MlirOp> beta_op,
    std::optional<mlir::MlirOp> alpha_op, const MaybePromotedScalar& beta,
    const MaybePromotedScalar& alpha, mlir::stablehlo::Precision precision,
    mlir::ElementType out_dtype) {
  auto batch1_shape = GetTensorTypeOrDie(batch1_op).getShape();
  auto batch2_shape = GetTensorTypeOrDie(batch2_op).getShape();
  const Dimensions out_shape = {batch1_shape[1], batch2_shape[2]};
  auto& builder = batch1_op.getBuilder();

  const bool has_product = !alpha.IsZero() && batch1_shape[0] > 0;

  // Case 1: no product term. The result is beta * self.
  if (!has_product) {
    if (beta.IsZero()) {
      return MakeConstant(builder, at::Scalar(0.0), out_dtype, out_shape);
    }
    // beta is not zero, so self_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp self_bcst,
                        BroadcastIfNeeded(*self_op, out_shape));
    if (beta.IsOne()) {
      return self_bcst;
    }
    // beta is neither 0 nor 1, so beta_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_tensor,
                        BroadcastIfNeeded(*beta_op, self_bcst));
    return mlir::stablehlo::Mul(self_bcst, beta_tensor);
  }

  // Case 2: compute product = alpha * sum_b(batch1[b] @ batch2[b]).
  TT_ASSIGN_OR_RETURN(mlir::MlirOp bmm_res,
                      BuildBmmShlo(batch1_op, batch2_op, out_dtype, precision));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp product,
      BuildSumShlo(bmm_res, {0}, ReductionMode::kDropDims, out_dtype));
  if (!alpha.IsOne()) {
    // alpha is neither 0 nor 1, so alpha_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_tensor,
                        BroadcastIfNeeded(*alpha_op, product));
    product = mlir::stablehlo::Mul(product, alpha_tensor);
  }
  if (beta.IsZero()) {
    return product;
  }

  // Case 3: add beta * self. self_op is guaranteed to have a value.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_bcst,
                      BroadcastIfNeeded(*self_op, product));
  if (!beta.IsOne()) {
    // beta is neither 0 nor 1, so beta_op is guaranteed to have a value.
    TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_tensor,
                        BroadcastIfNeeded(*beta_op, self_bcst));
    self_bcst = mlir::stablehlo::Mul(self_bcst, beta_tensor);
  }
  return mlir::stablehlo::Add(self_bcst, product);
}

auto GetAddbmmOpBuilder(MaybePromotedScalar beta, MaybePromotedScalar alpha,
                        bool beta_is_zero, bool has_beta_tensor,
                        bool has_alpha_tensor, mlir::ElementType out_dtype_mlir,
                        mlir::stablehlo::Precision precision) {
  return [beta = std::move(beta), alpha = std::move(alpha), beta_is_zero,
          has_beta_tensor, has_alpha_tensor, out_dtype_mlir, precision](
             absl::Span<mlir::MlirOp> inputs_op,
             mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    // Unpacks inputs from the flat span. Packing order in BuildAddbmmInputs():
    // [self (if beta != 0), batch1, batch2, beta_tensor, alpha_tensor]
    size_t idx = 0;
    std::optional<mlir::MlirOp> self_op;
    if (!beta_is_zero) {
      self_op = inputs_op[idx++];
    }
    mlir::MlirOp batch1_op = inputs_op[idx++];
    mlir::MlirOp batch2_op = inputs_op[idx++];
    std::optional<mlir::MlirOp> beta_op;
    if (has_beta_tensor) {
      beta_op = inputs_op[idx++];
    }
    std::optional<mlir::MlirOp> alpha_op;
    if (has_alpha_tensor) {
      alpha_op = inputs_op[idx++];
    }

    return BuildAddbmmShlo(self_op, batch1_op, batch2_op, beta_op, alpha_op,
                           beta, alpha, precision, out_dtype_mlir);
  };
}

std::vector<at::Tensor> BuildAddbmmInputs(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    bool beta_is_zero, std::optional<at::Tensor>& beta_tensor,
    std::optional<at::Tensor>& alpha_tensor) {
  std::vector<at::Tensor> inputs;
  inputs.reserve(5);
  if (!beta_is_zero) {
    inputs.push_back(self);
  }
  inputs.push_back(batch1);
  inputs.push_back(batch2);
  if (beta_tensor.has_value()) {
    inputs.push_back(std::move(*beta_tensor));
  }
  if (alpha_tensor.has_value()) {
    inputs.push_back(std::move(*alpha_tensor));
  }
  return inputs;
}

template <typename ReturnType, typename DispatchFn>
ReturnType DispatchAddbmm(const at::Tensor& self, const at::Tensor& batch1,
                          const at::Tensor& batch2, MaybePromotedScalar beta,
                          MaybePromotedScalar alpha,
                          OpParamCacheKeys& param_keys,
                          std::optional<OpName> op_name_override,
                          DispatchFn&& dispatch_fn) {
  TT_RETURN_IF_ERROR(ValidateAddbmmInputs(self, batch1, batch2));

  const at::ScalarType out_dtype = batch1.scalar_type();
  TT_ASSIGN_OR_RETURN(mlir::ElementType out_dtype_mlir,
                      ConvertTo<mlir::ElementType>(out_dtype));
  const Dimensions out_dims = {batch1.size(1), batch2.size(2)};

  std::optional<at::Tensor> beta_tensor;
  if (!beta.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(beta_tensor, beta.GetTensor(out_dtype));
  }
  std::optional<at::Tensor> alpha_tensor;
  if (!alpha.ValueMatchesExclude()) {
    TT_ASSIGN_OR_RETURN(alpha_tensor, alpha.GetTensor(out_dtype));
  }

  const bool beta_is_zero = beta.IsZero();
  const bool has_beta_tensor = beta_tensor.has_value();
  const bool has_alpha_tensor = alpha_tensor.has_value();
  const auto precision = GetAndAddPrecisionTo(param_keys);

  std::vector<at::Tensor> inputs = BuildAddbmmInputs(
      self, batch1, batch2, beta_is_zero, beta_tensor, alpha_tensor);
  auto op_builder = GetAddbmmOpBuilder(
      std::move(beta), std::move(alpha), beta_is_zero, has_beta_tensor,
      has_alpha_tensor, out_dtype_mlir, precision);

  return dispatch_fn(
      std::move(op_builder), std::move(inputs),
      DispatchOpOptions<1>{.op_name = op_name_override,
                           .out_dtype = out_dtype_mlir,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<DeviceBufferRef> ResolveAndRunAddbmm(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    MaybePromotedScalar beta, MaybePromotedScalar alpha,
    OpParamCacheKeys& param_keys, std::optional<OpName> op_name_override) {
  return DispatchAddbmm<absl::StatusOr<DeviceBufferRef>>(
      self, batch1, batch2, std::move(beta), std::move(alpha), param_keys,
      op_name_override,
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto inputs, auto options) {
        return DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                                        std::move(options));
      });
}

absl::Status ResolveAndRunAddbmmOut(
    const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
    MaybePromotedScalar beta, MaybePromotedScalar alpha,
    OpParamCacheKeys& param_keys, at::Tensor& out,
    std::optional<OpName> op_name_override) {
  return DispatchAddbmm<absl::Status>(
      self, batch1, batch2, std::move(beta), std::move(alpha), param_keys,
      op_name_override,
      [&]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto inputs, auto options) {
        return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, out,
                                           std::move(options));
      });
}

}  // namespace

at::Tensor AtenAddbmm(const at::Tensor& self, const at::Tensor& batch1,
                      const at::Tensor& batch2, const at::Scalar& beta,
                      const at::Scalar& alpha) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(OpName::kAddbmm, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha), {
              TT_ASSIGN_OR_THROW(
                  auto result_buffer,
                  ResolveAndRunAddbmm(self, batch1, batch2,
                                      std::move(promoted_beta),
                                      std::move(promoted_alpha), param_keys,
                                      OpName::kAddbmmOut));
              return MakeTensor(result_buffer);
            });
}

at::Tensor& AtenAddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                          const at::Tensor& batch2, const at::Scalar& beta,
                          const at::Scalar& alpha, at::Tensor& out) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(OpName::kAddbmmOut, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha, out), {
              TT_THROW_IF_ERROR(ValidateAddbmmInputs(self, batch1, batch2));
              TT_THROW_IF_ERROR(ValidateAddbmmOut(out, batch1.scalar_type()));
              TT_THROW_IF_ERROR(ResolveAndRunAddbmmOut(
                  self, batch1, batch2, std::move(promoted_beta),
                  std::move(promoted_alpha), param_keys, out, std::nullopt));
              return out;
            });
}

at::Tensor& AtenAddbmm_(at::Tensor& self, const at::Tensor& batch1,
                        const at::Tensor& batch2, const at::Scalar& beta,
                        const at::Scalar& alpha) {
  auto promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kZero,
                                                            ScalarValue::kOne);
  TT_KERNEL(OpName::kAddbmm_, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha), {
              TT_THROW_IF_ERROR(ResolveAndRunAddbmmOut(
                  self, batch1, batch2, std::move(promoted_beta),
                  std::move(promoted_alpha), param_keys, self,
                  OpName::kAddbmmOut));
              return self;
            });
}

}  // namespace torch_tpu
