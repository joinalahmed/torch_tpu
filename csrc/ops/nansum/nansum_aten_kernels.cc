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

#include "csrc/ops/nansum/nansum_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/is/is.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

struct NansumPlan {
  Dimensions reduce_dims;
  Dimensions out_dims;
  ReductionMode reduction_mode;
  at::ScalarType out_scalar_type;
  mlir::ElementType out_type;
  // Integral inputs have no NaNs, so they reduce directly.
  bool replace_nans;
};

absl::StatusOr<NansumPlan> PlanNansum(const at::Tensor& self,
                                      c10::OptionalArrayRef<int64_t> dim,
                                      bool keepdim,
                                      at::ScalarType out_scalar_type) {
  TT_RET_CHECK(!self.is_complex(), error::kInvalidArgument)
      << "nansum does not support complex inputs";
  NansumPlan plan;
  plan.reduction_mode =
      keepdim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(plan.reduce_dims, CanonicalizeDims(self, dim));
  plan.out_dims = GetSizesAfterReduction(self.sizes(), plan.reduction_mode,
                                         plan.reduce_dims);
  plan.out_scalar_type = out_scalar_type;
  TT_ASSIGN_OR_RETURN(plan.out_type,
                      ConvertTo<mlir::ElementType>(out_scalar_type));
  plan.replace_nans = self.is_floating_point();
  return plan;
}

MlirUnaryOpBuilder MakeNansumBuilder(const NansumPlan& plan) {
  return [reduce_dims = plan.reduce_dims, reduction_mode = plan.reduction_mode,
          out_type = plan.out_type, replace_nans = plan.replace_nans](
             mlir::MlirOp self_op) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp summand = self_op;
    if (replace_nans) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp is_nan, BuildIsNanShlo(self_op));
      const mlir::Type type = GetTensorTypeOrDie(self_op).getElementType();
      mlir::MlirOp zero = MakeConstantLike(self_op, 0, type);
      summand = mlir::stablehlo::Select(is_nan, zero, self_op);
    }
    return BuildSumShlo(summand, reduce_dims, reduction_mode, out_type);
  };
}

// nansum promotes integral inputs to int64 like sum.
at::ScalarType DefaultNansumDtype(const at::Tensor& self,
                                  std::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return *dtype;
  }
  return c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
             ? at::kLong
             : self.scalar_type();
}

}  // namespace

at::Tensor AtenNansum(const at::Tensor& self,
                      c10::OptionalArrayRef<int64_t> dim, bool keepdim,
                      std::optional<at::ScalarType> dtype) {
  TT_KERNEL(OpName::kNansum, param_keys, (self, dim, keepdim, dtype), {
    TT_ASSIGN_OR_THROW(
        const NansumPlan plan,
        PlanNansum(self, dim, keepdim, DefaultNansumDtype(self, dtype)));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(self, MakeNansumBuilder(plan),
                {.op_param_cache_keys = std::move(param_keys),
                 .out_dtype = plan.out_type,
                 .out_dims = plan.out_dims}));
    return result;
  });
}

at::Tensor& AtenNansumOut(const at::Tensor& self,
                          c10::OptionalArrayRef<int64_t> dim, bool keepdim,
                          std::optional<at::ScalarType> dtype,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kNansumOut, param_keys, (self, dim, keepdim, dtype, out), {
    TT_ASSIGN_OR_THROW(
        const NansumPlan plan,
        PlanNansum(self, dim, keepdim, dtype.value_or(out.scalar_type())));
    TT_THROW_IF_ERROR(UnaryOpOut(self, out, MakeNansumBuilder(plan),
                                 {.op_param_cache_keys = std::move(param_keys),
                                  .out_dtype = plan.out_type,
                                  .out_dims = plan.out_dims}));
    return out;
  });
}

}  // namespace torch_tpu
