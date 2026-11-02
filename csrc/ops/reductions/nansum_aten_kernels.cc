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

#include "csrc/ops/reductions/nansum_aten_kernels.h"

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
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Everything the MLIR builder needs that depends on the ATen arguments.
struct NansumPlan {
  Dimensions reduce_dims;
  Dimensions out_dims;
  ReductionMode reduction_mode;
  mlir::ElementType out_type;
  // Integral inputs have no NaN, so they are summed as is.
  bool replace_nan;
};

absl::StatusOr<NansumPlan> PlanNansum(const at::Tensor& self,
                                      c10::OptionalArrayRef<int64_t> dim,
                                      bool keep_dim,
                                      c10::ScalarType result_type) {
  // ATen implements nansum for floating-point inputs and, through sum, for
  // integral ones. Complex inputs are rejected like on CPU.
  TT_RET_CHECK(!self.is_complex(), error::kPythonNotImplementedError)
      << "complex inputs are not supported";
  NansumPlan plan;
  plan.reduction_mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(plan.reduce_dims, CanonicalizeDims(self, dim));
  plan.out_dims = GetSizesAfterReduction(self.sizes(), plan.reduction_mode,
                                         plan.reduce_dims);
  TT_ASSIGN_OR_RETURN(plan.out_type,
                      ConvertTo<mlir::ElementType>(result_type));
  plan.replace_nan = self.is_floating_point();
  return plan;
}

// Replaces NaN with zero in the input dtype, then sums like aten::sum. The
// cast to the result dtype happens inside the sum, after the replacement, so
// a float input summed into an integer dtype never casts a NaN.
MlirUnaryOpBuilder MakeNansumBuilder(NansumPlan plan) {
  return [plan = std::move(plan)](
             mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    mlir::MlirOp summand = input_op;
    if (plan.replace_nan) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp is_nan, BuildIsNanShlo(input_op));
      mlir::MlirOp zero = MakeConstantLike(input_op, 0);
      summand = mlir::stablehlo::Select(is_nan, zero, input_op);
    }
    return BuildSumShlo(summand, plan.reduce_dims, plan.reduction_mode,
                        plan.out_type);
  };
}

// Like sum, nansum accumulates integral inputs (including bool) in int64
// unless `dtype` is given.
c10::ScalarType NansumResultType(const at::Tensor& self,
                                 std::optional<c10::ScalarType> dtype) {
  if (dtype.has_value()) {
    return *dtype;
  }
  return c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
             ? c10::kLong
             : self.scalar_type();
}

}  // namespace

at::Tensor AtenNansum(const at::Tensor& self,
                      c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                      std::optional<c10::ScalarType> dtype) {
  TT_KERNEL(OpName::kNansum, param_keys, (self, dim, keep_dim, dtype), {
    TT_ASSIGN_OR_THROW(
        NansumPlan plan,
        PlanNansum(self, dim, keep_dim, NansumResultType(self, dtype)));
    const mlir::ElementType out_type = plan.out_type;
    const Dimensions out_dims = plan.out_dims;
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(self, MakeNansumBuilder(std::move(plan)),
                {.op_param_cache_keys = std::move(param_keys),
                 .out_dtype = out_type,
                 .out_dims = out_dims}));
    return result;
  });
}

at::Tensor& AtenNansumOut(const at::Tensor& self,
                          c10::OptionalArrayRef<int64_t> dim, bool keep_dim,
                          std::optional<c10::ScalarType> dtype,
                          at::Tensor& out) {
  TT_KERNEL(
      OpName::kNansumOut, param_keys, (self, dim, keep_dim, dtype, out), {
        // ATen rejects a `dtype` that differs from `out` instead of casting.
        TT_CHECK_THROW(!dtype.has_value() || *dtype == out.scalar_type(),
                       error::kInvalidArgument)
            << "provided dtype must match dtype of result. Got "
            << ToString(out.scalar_type()) << " and " << ToString(*dtype)
            << ".";
        TT_ASSIGN_OR_THROW(NansumPlan plan,
                           PlanNansum(self, dim, keep_dim, out.scalar_type()));
        const mlir::ElementType out_type = plan.out_type;
        const Dimensions out_dims = plan.out_dims;
        TT_THROW_IF_ERROR(
            UnaryOpOut(self, out, MakeNansumBuilder(std::move(plan)),
                       {.op_param_cache_keys = std::move(param_keys),
                        .out_dtype = out_type,
                        .out_dims = out_dims}));
        return out;
      });
}

}  // namespace torch_tpu
