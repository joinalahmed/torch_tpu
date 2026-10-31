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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
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

absl::Status ValidateAddbmmInputs(const at::Tensor& self,
                                  const at::Tensor& batch1,
                                  const at::Tensor& batch2) {
  TT_RET_CHECK(batch1.dim() == 3, error::kInvalidArgument)
      << "batch1 must be a 3D tensor";
  TT_RET_CHECK(batch2.dim() == 3, error::kInvalidArgument)
      << "batch2 must be a 3D tensor";
  TT_RET_CHECK(batch1.size(0) == batch2.size(0), error::kInvalidArgument)
      << "batch1 and batch2 must have same number of batches, got "
      << batch1.size(0) << " and " << batch2.size(0);
  TT_RET_CHECK(batch1.size(2) == batch2.size(1), error::kInvalidArgument)
      << "Incompatible matrix sizes for bmm (" << batch1.size(1) << "x"
      << batch1.size(2) << " and " << batch2.size(1) << "x" << batch2.size(2)
      << ")";
  TT_RET_CHECK(self.scalar_type() == batch1.scalar_type() &&
                   batch1.scalar_type() == batch2.scalar_type(),
               error::kInvalidArgument)
      << "self and mat2 must have the same dtype, but got "
      << ToString(self.scalar_type()) << " and "
      << ToString(batch1.scalar_type());
  TT_RET_CHECK(c10::isFloatingType(batch1.scalar_type()),
               error::kPythonNotImplementedError)
      << "\"addbmm\" not implemented for '" << ToString(batch1.scalar_type())
      << "'";
  return absl::OkStatus();
}

// beta * self + alpha * sum_b(batch1[b] @ batch2[b]). Inputs are
// [self, batch1, batch2, beta, alpha], or [self, beta] when there are no
// batches. `beta_is_zero` leaves self out of the graph so that a NaN in self
// does not reach the result, as in ATen.
absl::StatusOr<mlir::MlirOp> BuildAddbmmShlo(
    absl::Span<mlir::MlirOp> inputs, absl::Span<const int64_t> out_shape,
    bool beta_is_zero, bool has_batches, mlir::ElementType out_type,
    mlir::stablehlo::Precision precision) {
  mlir::MlirOp self_op = inputs[0];
  if (!has_batches) {
    if (beta_is_zero) {
      return MakeConstant(self_op.getBuilder(), 0.0,
                          GetTensorTypeOrDie(self_op).getElementType(),
                          out_shape);
    }
    TT_ASSIGN_OR_RETURN(mlir::MlirOp self_b,
                        BroadcastIfNeeded(self_op, out_shape));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_b,
                        BroadcastIfNeeded(inputs[1], self_b));
    return mlir::stablehlo::Mul(self_b, beta_b);
  }

  mlir::MlirOp batch1_op = inputs[1];
  mlir::MlirOp batch2_op = inputs[2];
  mlir::MlirOp beta_op = inputs[3];
  mlir::MlirOp alpha_op = inputs[4];

  TT_ASSIGN_OR_RETURN(mlir::MlirOp bmm,
                      BuildBmmShlo(batch1_op, batch2_op, out_type, precision));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp summed,
                      BuildSumShlo(bmm, {0}, ReductionMode::kDropDims));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_b, BroadcastIfNeeded(alpha_op, summed));
  mlir::MlirOp scaled = mlir::stablehlo::Mul(summed, alpha_b);
  if (beta_is_zero) {
    return scaled;
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_b, BroadcastIfNeeded(self_op, scaled));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_b, BroadcastIfNeeded(beta_op, self_b));
  mlir::MlirOp self_scaled = mlir::stablehlo::Mul(self_b, beta_b);
  return mlir::stablehlo::Add(self_scaled, scaled);
}

template <typename DispatchFn>
absl::Status DispatchAddbmm(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2,
                            PromotedScalar& promoted_beta,
                            PromotedScalar& promoted_alpha,
                            OpParamCacheKeys& param_keys,
                            DispatchFn&& dispatch_fn) {
  TT_RETURN_IF_ERROR(ValidateAddbmmInputs(self, batch1, batch2));
  const at::ScalarType scalar_type = batch1.scalar_type();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  const Dimensions out_dims = {batch1.size(1), batch2.size(2)};

  // The graph depends on these host-side values.
  const bool beta_is_zero = promoted_beta.scalar().toDouble() == 0.0;
  const bool has_batches = batch1.size(0) > 0;
  TT_RETURN_IF_ERROR(param_keys.SetParam("beta_is_zero", beta_is_zero));
  TT_RETURN_IF_ERROR(param_keys.SetParam("has_batches", has_batches));
  const mlir::stablehlo::Precision precision =
      GetAndAddPrecisionTo(param_keys);

  TT_ASSIGN_OR_RETURN(at::Tensor beta_tensor,
                      promoted_beta.GetTensor(scalar_type));
  TT_ASSIGN_OR_RETURN(at::Tensor alpha_tensor,
                      promoted_alpha.GetTensor(scalar_type));
  std::vector<at::Tensor> inputs;
  if (has_batches) {
    inputs = {self, batch1, batch2, beta_tensor, alpha_tensor};
  } else {
    inputs = {self, beta_tensor};
  }

  auto op_builder = [out_dims, beta_is_zero, has_batches, out_type, precision](
                        absl::Span<mlir::MlirOp> inputs_op,
                        mlir::MlirBuilder& builder)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildAddbmmShlo(inputs_op, out_dims, beta_is_zero, has_batches,
                           out_type, precision);
  };
  return dispatch_fn(
      std::move(op_builder), std::move(inputs),
      DispatchOpOptions<1>{.out_dtype = out_type,
                           .out_dims = out_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::StatusOr<at::Tensor> Addbmm(const at::Tensor& self,
                                  const at::Tensor& batch1,
                                  const at::Tensor& batch2,
                                  PromotedScalar& promoted_beta,
                                  PromotedScalar& promoted_alpha,
                                  OpParamCacheKeys& param_keys) {
  std::optional<DeviceBufferRef> result;
  TT_RETURN_IF_ERROR(DispatchAddbmm(
      self, batch1, batch2, promoted_beta, promoted_alpha, param_keys,
      [&result]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto inputs, auto options) -> absl::Status {
        TT_ASSIGN_OR_RETURN(
            result, DispatchOp<kDynamicSize>(std::move(op_builder), inputs,
                                             std::move(options)));
        return absl::OkStatus();
      }));
  return MakeTensor(*result);
}

absl::Status AddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                       const at::Tensor& batch2, PromotedScalar& promoted_beta,
                       PromotedScalar& promoted_alpha,
                       OpParamCacheKeys& param_keys, at::Tensor& out) {
  return DispatchAddbmm(
      self, batch1, batch2, promoted_beta, promoted_alpha, param_keys,
      [&out]  // REF_CAPTURE_OK=called synchronously
      (auto op_builder, auto inputs, auto options) {
        return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, out,
                                           std::move(options));
      });
}

}  // namespace

at::Tensor AtenAddbmm(const at::Tensor& self, const at::Tensor& batch1,
                      const at::Tensor& batch2, const at::Scalar& beta,
                      const at::Scalar& alpha) {
  PromotedScalar promoted_beta = PromoteScalar(beta);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kAddbmm, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha), {
              TT_ASSIGN_OR_THROW(at::Tensor result,
                                 Addbmm(self, batch1, batch2, promoted_beta,
                                        promoted_alpha, param_keys));
              return result;
            });
}

at::Tensor& AtenAddbmmOut(const at::Tensor& self, const at::Tensor& batch1,
                          const at::Tensor& batch2, const at::Scalar& beta,
                          const at::Scalar& alpha, at::Tensor& out) {
  PromotedScalar promoted_beta = PromoteScalar(beta);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kAddbmmOut, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha, out), {
              TT_THROW_IF_ERROR(AddbmmOut(self, batch1, batch2, promoted_beta,
                                          promoted_alpha, param_keys, out));
              return out;
            });
}

at::Tensor& AtenAddbmm_(at::Tensor& self, const at::Tensor& batch1,
                        const at::Tensor& batch2, const at::Scalar& beta,
                        const at::Scalar& alpha) {
  PromotedScalar promoted_beta = PromoteScalar(beta);
  PromotedScalar promoted_alpha = PromoteScalar(alpha);
  TT_KERNEL(OpName::kAddbmm_, param_keys,
            (self, batch1, batch2, promoted_beta, promoted_alpha), {
              TT_THROW_IF_ERROR(AddbmmOut(self, batch1, batch2, promoted_beta,
                                          promoted_alpha, param_keys, self));
              return self;
            });
}

}  // namespace torch_tpu
