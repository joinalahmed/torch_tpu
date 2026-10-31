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

#include "csrc/ops/trace/trace_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Sums the main diagonal: mask the input with iota(0) == iota(1), then reduce
// over both dimensions.
absl::StatusOr<mlir::MlirOp> BuildTraceShlo(mlir::MlirOp self_op,
                                            mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = self_op.getBuilder();
  const mlir::Type index_type = builder.getOpBuilder().getI32Type();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp rows, Iota(self_op, 0, index_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp cols, Iota(self_op, 1, index_type));
  mlir::MlirOp on_diagonal = mlir::stablehlo::Compare(
      rows, cols, mlir::stablehlo::ComparisonDirection::EQ);
  const mlir::Type type = GetTensorTypeOrDie(self_op).getElementType();
  mlir::MlirOp zero = MakeConstantLike(self_op, 0, type);
  mlir::MlirOp diagonal = mlir::stablehlo::Select(on_diagonal, self_op, zero);
  return BuildSumShlo(diagonal, {0, 1}, ReductionMode::kDropDims, out_type);
}

}  // namespace

at::Tensor AtenTrace(const at::Tensor& self) {
  TT_KERNEL(OpName::kTrace, param_keys, (self), {
    TT_CHECK_THROW(self.dim() == 2, error::kInvalidArgument)
        << "trace: expected a matrix, but got tensor with dim " << self.dim();
    const at::ScalarType scalar_type = self.scalar_type();
    TT_CHECK_THROW(scalar_type != at::kBool && scalar_type != at::kHalf &&
                       scalar_type != at::kBFloat16,
                   error::kPythonNotImplementedError)
        << "\"trace\" not implemented for '" << ToString(scalar_type) << "'";
    // Like sum, trace accumulates integral inputs in int64.
    const at::ScalarType out_scalar_type =
        c10::isIntegralType(scalar_type, /*includeBool=*/false) ? at::kLong
                                                                : scalar_type;
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_type,
                       ConvertTo<mlir::ElementType>(out_scalar_type));
    const Dimensions out_dims;
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(
            self,
            [out_type](mlir::MlirOp self_op) {
              return BuildTraceShlo(self_op, out_type);
            },
            {.op_param_cache_keys = std::move(param_keys),
             .out_dtype = out_type,
             .out_dims = out_dims}));
    return result;
  });
}

}  // namespace torch_tpu
