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

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/util/ArrayRef.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

// Zeroes everything off the main diagonal, then sums both dimensions. For a
// non-square matrix the main diagonal has min(rows, cols) elements.
absl::StatusOr<mlir::MlirOp> BuildTraceShlo(mlir::MlirOp input_op,
                                            mlir::ElementType out_type) {
  const mlir::Type index_type =
      input_op.getBuilder().getOpBuilder().getI32Type();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp row, Iota(input_op, 0, index_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp col, Iota(input_op, 1, index_type));
  mlir::MlirOp on_diagonal = mlir::stablehlo::Compare(
      row, col, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp zero = MakeConstantLike(input_op, 0);
  mlir::MlirOp diagonal = mlir::stablehlo::Select(on_diagonal, input_op, zero);
  return BuildSumShlo(diagonal, {0, 1}, ReductionMode::kDropDims, out_type);
}

// Like sum, trace accumulates integral inputs (including bool) in int64. This
// matches the CUDA kernel, which is diagonal().sum().
c10::ScalarType TraceResultType(const at::Tensor& self) {
  return c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)
             ? c10::kLong
             : self.scalar_type();
}

}  // namespace

at::Tensor AtenTrace(const at::Tensor& self) {
  TT_KERNEL(OpName::kTrace, param_keys, (self), {
    TT_CHECK_THROW(self.dim() == 2, error::kInvalidArgument)
        << "expected a matrix, but got tensor with dim " << self.dim();
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_type,
                       ConvertTo<mlir::ElementType>(TraceResultType(self)));
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(
            self,
            [out_type](mlir::MlirOp input_op) {
              return BuildTraceShlo(input_op, out_type);
            },
            {.op_param_cache_keys = std::move(param_keys),
             .out_dtype = out_type,
             .out_dims = at::IntArrayRef()}));
    return result;
  });
}

}  // namespace torch_tpu
