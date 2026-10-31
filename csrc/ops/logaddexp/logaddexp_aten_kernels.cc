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

#include "csrc/ops/logaddexp/logaddexp_aten_kernels.h"

#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
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

// log(exp(a) + exp(b)) = max(a, b) + log1p(exp(-|a - b|)), and the base-2
// variant max(a, b) + log1p(exp2(-|a - b|)) / ln(2). When both inputs are the
// same infinity the formula gives inf - inf = NaN; ATen returns that infinity,
// so it is selected explicitly.
absl::StatusOr<mlir::MlirOp> BuildLogaddexpShlo(mlir::MlirOp a, mlir::MlirOp b,
                                                bool base2) {
  TT_ASSIGN_OR_RETURN((auto [a_b, b_b]), ApplyBroadcastIfNeeded(a, b));
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType dtype,
      ConvertTo<mlir::ElementType>(GetTensorTypeOrDie(a_b).getElementType()));

  const mlir::Type type = GetTensorTypeOrDie(a_b).getElementType();
  mlir::MlirOp max = mlir::stablehlo::Max(a_b, b_b);
  mlir::MlirOp diff = mlir::stablehlo::Subtract(a_b, b_b);
  mlir::MlirOp abs_diff = mlir::stablehlo::Abs(diff);
  mlir::MlirOp neg_abs_diff = mlir::stablehlo::Neg(abs_diff);
  mlir::MlirOp tail;
  if (base2) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp pow2, BuildExp2Shlo(neg_abs_diff, dtype));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp log1p, BuildLog1pShlo(pow2, dtype));
    mlir::MlirOp inv_ln2 = MakeConstantLike(a_b, 1.0 / std::log(2.0), type);
    tail = mlir::stablehlo::Mul(log1p, inv_ln2);
  } else {
    mlir::MlirOp exp = mlir::stablehlo::Exp(neg_abs_diff);
    TT_ASSIGN_OR_RETURN(tail, BuildLog1pShlo(exp, dtype));
  }
  mlir::MlirOp result = mlir::stablehlo::Add(max, tail);

  mlir::MlirOp inf =
      MakeConstantLike(a_b, std::numeric_limits<double>::infinity(), type);
  mlir::MlirOp abs_a = mlir::stablehlo::Abs(a_b);
  mlir::MlirOp is_inf_a = mlir::stablehlo::Compare(
      abs_a, inf, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp a_eq_b = mlir::stablehlo::Compare(
      a_b, b_b, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp same_inf = mlir::stablehlo::And(is_inf_a, a_eq_b);
  return mlir::stablehlo::Select(same_inf, a_b, result);
}

at::Tensor& LogaddexpOut(const at::Tensor& self, const at::Tensor& other,
                         at::Tensor& out, bool base2) {
  TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
      << "expected out tensor to be on " << GetPrivateUse1DeviceDebugName()
      << ", got " << out.device();
  TT_CHECK_THROW(!self.is_complex() && !other.is_complex(),
                 error::kInvalidArgument)
      << "expected non-complex dtypes, got " << ToString(self.scalar_type())
      << " and " << ToString(other.scalar_type());

  TT_ASSIGN_OR_THROW(const auto out_dims,
                     InferSize(self.sizes(), other.sizes()));
  if (out.is_alias_of(self) || out.is_alias_of(other)) {
    TT_CHECK_THROW(absl::MakeConstSpan(out.sizes()) == out_dims,
                   error::kInvalidArgument)
        << "output with shape " << ToString(out.sizes())
        << " doesn't match the broadcast shape of the tensor being "
           "operated on in-place, which has shape "
        << ToString(out_dims);
  }

  BinaryOpOptions opts{
      .force_float_inputs = true,
      .op_param_cache_keys = OpParamCacheKeys::Empty(),
  };
  TT_THROW_IF_ERROR(BinaryOpOut(
      self, other, out,
      [base2](mlir::MlirOp a, mlir::MlirOp b) {
        return BuildLogaddexpShlo(a, b, base2);
      },
      std::move(opts)));
  return out;
}

}  // namespace

at::Tensor& AtenLogaddexpOut(const at::Tensor& self, const at::Tensor& other,
                             at::Tensor& out) {
  TT_KERNEL(OpName::kLogaddexpOut, _, (self, other, out), {
    return LogaddexpOut(self, other, out, /*base2=*/false);
  });
}

at::Tensor& AtenLogaddexp2Out(const at::Tensor& self, const at::Tensor& other,
                              at::Tensor& out) {
  TT_KERNEL(OpName::kLogaddexp2Out, _, (self, other, out), {
    return LogaddexpOut(self, other, out, /*base2=*/true);
  });
}

}  // namespace torch_tpu
