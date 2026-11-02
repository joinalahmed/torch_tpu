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

#include "csrc/ops/logaddexp2/logaddexp2_aten_kernels.h"

#include <cmath>
#include <limits>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/binary_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// log2(2^a + 2^b) = max(a, b) + log1p(2^(-|a - b|)) / ln(2). This matches the
// ATen CPU and CUDA kernels, which also special-case a == b == +-inf: the
// formula would give inf - inf = NaN there, and ATen returns that infinity.
absl::StatusOr<mlir::MlirOp> BuildLogaddexp2Shlo(mlir::MlirOp self,
                                                 mlir::MlirOp other) {
  TT_ASSIGN_OR_RETURN((auto [self_, other_]),
                      ApplyBroadcastIfNeeded(self, other));
  const mlir::Type type = GetTensorTypeOrDie(self_).getElementType();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(type));

  // The stablehlo builders take MlirOp& and reject temporaries.
  mlir::MlirOp max = mlir::stablehlo::Max(self_, other_);
  mlir::MlirOp diff = mlir::stablehlo::Subtract(self_, other_);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
  mlir::MlirOp neg_abs_diff = mlir::stablehlo::Neg(abs_diff);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp exp2_neg_abs_diff,
                      BuildExp2Shlo(neg_abs_diff, dtype));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp log1p,
                      BuildLog1pShlo(exp2_neg_abs_diff, dtype));
  mlir::MlirOp inv_ln2 = MakeConstantLike(self_, 1.0 / std::log(2.0), type);
  mlir::MlirOp log1p_base2 = mlir::stablehlo::Mul(log1p, inv_ln2);
  mlir::MlirOp result = mlir::stablehlo::Add(max, log1p_base2);

  mlir::MlirOp inf =
      MakeConstantLike(self_, std::numeric_limits<double>::infinity(), type);
  mlir::MlirOp abs_self = mlir::stablehlo::Abs(self_);
  mlir::MlirOp self_is_inf = mlir::stablehlo::Compare(
      abs_self, inf, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp self_eq_other = mlir::stablehlo::Compare(
      self_, other_, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp both_inf = mlir::stablehlo::And(self_is_inf, self_eq_other);
  return mlir::stablehlo::Select(both_inf, self_, result);
}

}  // namespace

at::Tensor& AtenLogaddexp2Out(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogaddexp2Out, _, (self, other, out), {
    // ATen only implements logaddexp2 for floating-point dtypes.
    const at::ScalarType result_type = at::result_type(self, other);
    TT_CHECK_THROW(c10::isFloatingType(result_type),
                   error::kPythonNotImplementedError)
        << "\"logaddexp2\" not implemented for '" << ToString(result_type)
        << "'";
    TT_THROW_IF_ERROR(
        BinaryOpOut(self, other, out, BuildLogaddexp2Shlo,
                    {.op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return out;
  });
}

}  // namespace torch_tpu
