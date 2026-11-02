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

#include "csrc/ops/reductions/std_aten_kernels.h"

#include <array>
#include <complex>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "ATen/OpMathType.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/python_context.h"
#include "csrc/ops/reductions/mean_aten_kernels.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

NAryMlirOpBuilder<2, 1> CreateStdBuilder(Dimensions dim_vec,
                                         const ReductionMode reduction_mode,
                                         const int64_t reduction_size) {
  return [dim_vec = std::move(dim_vec), reduction_mode,
          reduction_size](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [self_op, correction_op] = inputs;
    mlir::MlirBuilder& builder = self_op.getBuilder();
    const mlir::Type type = GetTensorTypeOrDie(self_op).getElementType();
    TT_ASSIGN_OR_RETURN(
        const mlir::ElementType corr_type,
        ConvertTo<mlir::ElementType>(
            GetTensorTypeOrDie(correction_op).getElementType()));
    const mlir::ElementType var_type = RealComponentOf(corr_type);

    // sum_tensor = ReduceSum(self, dim, keep_dims=true)
    TT_ASSIGN_OR_RETURN(
        const auto sum_op,
        BuildSumShlo(self_op, dim_vec, ReductionMode::kKeepDims));

    // mean = sum_tensor / reduction_size
    const mlir::MlirOp reduction_size_const =
        llvm::isa<mlir::ComplexType>(type)
            ? MakeScalarConstant(builder,
                                 std::complex<double>(
                                     static_cast<double>(reduction_size), 0.0),
                                 type)
            : MakeScalarConstant(builder, static_cast<double>(reduction_size),
                                 type);
    TT_ASSIGN_OR_RETURN(const auto mean,
                        BuildDivShlo(sum_op, reduction_size_const));

    // diff = self - mean
    TT_ASSIGN_OR_RETURN(const auto diff, BuildSubShlo(self_op, mean));

    // conj_diff = conj(diff)
    TT_ASSIGN_OR_RETURN(const auto conj_diff, BuildConjPhysicalShlo(diff));

    // squared_diff = diff * conj_diff
    TT_ASSIGN_OR_RETURN(const auto squared_diff, BuildMulShlo(diff, conj_diff));

    // sum_sq_tensor = ReduceSum(squared_diff, dim, keep_dims=keep_dim)
    TT_ASSIGN_OR_RETURN(
        const auto sum_sq_op,
        BuildSumShlo(squared_diff, dim_vec, reduction_mode, var_type));

    // denom = max(reduction_size - correction, 0)
    const auto reduction_size_const_corr = MakeScalarConstant(
        builder, static_cast<double>(reduction_size), var_type);
    TT_ASSIGN_OR_RETURN(const auto denom_diff,
                        BuildSubShlo(reduction_size_const_corr, correction_op));
    const auto zero = MakeScalarConstant(builder, 0.0, var_type);
    TT_ASSIGN_OR_RETURN(const auto denom, BuildMaximumShlo(denom_diff, zero));

    // var = sum_sq_tensor / denom
    TT_ASSIGN_OR_RETURN(const auto var, BuildDivShlo(sum_sq_op, denom));

    // std = sqrt(var)
    return BuildSqrtShlo(var, var_type);
  };
}

void WarnIfInvalidCorrection(OpName op_name, double correction_val,
                             int64_t reduction_size) {
  const double denom = reduction_size - correction_val;
  if (denom <= 0) {
    TT_WARN(WarningCategory::kCorrectness, GetRootOpName(op_name),
            "(): degrees of freedom (i.e., reduction size - correction) should "
            "be positive, got reduction size = ",
            reduction_size, ", correction = ", correction_val,
            ", and degrees of freedom = ", denom);
  }
}

}  // namespace

at::Tensor AtenStd(const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
                   const std::optional<at::Scalar>& correction, bool keep_dim) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kStdCorrection, param_keys,
      (self, dim, promoted_correction, keep_dim), {
        c10::ScalarType scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(scalar_dtype));
        if (c10::isComplexType(scalar_dtype)) {
          scalar_dtype = c10::toRealValueType(scalar_dtype);
        }

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        const Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);

        TT_ASSIGN_OR_THROW(const at::Tensor correction_tensor,
                           promoted_correction.GetTensor(scalar_dtype));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        const double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kStdCorrection, correction_val,
                                reduction_size);

        TT_ASSIGN_OR_THROW(const mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        std::optional<mlir::ElementType> computation_dtype = std::nullopt;
        const bool upcast_std = (scalar_dtype_mlir == mlir::ElementType::BF16 ||
                                 scalar_dtype_mlir == mlir::ElementType::F16);
        if (upcast_std) {
          computation_dtype = mlir::ElementType::F32;
        }

        auto std_builder = CreateStdBuilder(std::move(canonicalized_dims),
                                            reduction_mode, reduction_size);

        TT_ASSIGN_OR_THROW(
            auto result_bufs,
            (DispatchOp<2, 1>(std::move(std_builder), {self, correction_tensor},
                              {.out_dtype = scalar_dtype_mlir,
                               .out_dims = output_dims,
                               .computation_dtype = computation_dtype,
                               .op_param_cache_keys = std::move(param_keys)})));
        return MakeTensor(result_bufs);
      });
}

at::Tensor& AtenStdOut(const at::Tensor& self,
                       c10::OptionalArrayRef<int64_t> dim,
                       const std::optional<at::Scalar>& correction,
                       bool keep_dim, at::Tensor& out) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kStdCorrectionOut, param_keys,
      (self, dim, promoted_correction, keep_dim, out), {
        c10::ScalarType self_scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(self_scalar_dtype));
        c10::ScalarType scalar_dtype = out.scalar_type();

        const c10::ScalarType expected_dtype =
            c10::toRealValueType(self_scalar_dtype);
        const c10::ScalarType real_out_dtype =
            c10::toRealValueType(scalar_dtype);

        TT_THROW_IF_ERROR(ValidateOutDtype(out, expected_dtype,
                                           /*allow_out_dtype_cast=*/true));

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonicalized_dims,
                           CanonicalizeDims(self, dim));
        const Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonicalized_dims);

        TT_ASSIGN_OR_THROW(const at::Tensor correction_tensor,
                           promoted_correction.GetTensor(real_out_dtype));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        const double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kStdCorrectionOut, correction_val,
                                reduction_size);

        TT_ASSIGN_OR_THROW(const mlir::ElementType self_real_dtype_mlir,
                           internal::ToElementType(expected_dtype));
        TT_ASSIGN_OR_THROW(const mlir::ElementType real_out_dtype_mlir,
                           internal::ToElementType(real_out_dtype));
        TT_ASSIGN_OR_THROW(const mlir::ElementType scalar_dtype_mlir,
                           internal::ToElementType(scalar_dtype));

        std::optional<mlir::ElementType> computation_dtype = std::nullopt;
        const bool upcast_std =
            (self_real_dtype_mlir == mlir::ElementType::BF16 ||
             self_real_dtype_mlir == mlir::ElementType::F16 ||
             real_out_dtype_mlir == mlir::ElementType::BF16 ||
             real_out_dtype_mlir == mlir::ElementType::F16);
        if (upcast_std) {
          computation_dtype = mlir::ElementType::F32;
        }

        auto std_builder = CreateStdBuilder(std::move(canonicalized_dims),
                                            reduction_mode, reduction_size);

        TT_THROW_IF_ERROR((DispatchOpOut<2, 1>(
            std::move(std_builder), {self, correction_tensor}, out,
            {.out_dtype = scalar_dtype_mlir,
             .out_dims = output_dims,
             .computation_dtype = computation_dtype,
             .op_param_cache_keys = std::move(param_keys)})));
        return out;
      });
}

std::tuple<at::Tensor, at::Tensor> AtenStdMeanCorrection(
    const at::Tensor& self, c10::OptionalArrayRef<int64_t> dim,
    const std::optional<at::Scalar>& correction, bool keep_dim) {
  PromotedScalar promoted_correction =
      PromoteScalar(correction.value_or(at::Scalar(1.0)));

  TT_KERNEL(
      OpName::kStdMeanCorrection, param_keys,
      (self, dim, promoted_correction, keep_dim), {
        const c10::ScalarType self_scalar_dtype = self.scalar_type();
        TT_THROW_IF_ERROR(ValidateFloatOrComplex(self_scalar_dtype));
        // The mean keeps the input dtype; the std of a complex input is real.
        const c10::ScalarType std_scalar_dtype =
            c10::toRealValueType(self_scalar_dtype);

        const ReductionMode reduction_mode =
            keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
        TT_ASSIGN_OR_THROW(Dimensions canonical_dims,
                           CanonicalizeDims(self, dim));
        const Dimensions output_dims = GetSizesAfterReduction(
            self.sizes(), reduction_mode, canonical_dims);

        TT_ASSIGN_OR_THROW(const mlir::ElementType std_dtype_mlir,
                           internal::ToElementType(std_scalar_dtype));
        TT_ASSIGN_OR_THROW(const mlir::ElementType self_dtype_mlir,
                           internal::ToElementType(self_scalar_dtype));

        // Like var_mean, compute BF16 and F16 inputs in F32.
        std::optional<mlir::ElementType> computation_dtype = std::nullopt;
        const bool upcast_std = (std_dtype_mlir == mlir::ElementType::BF16 ||
                                 std_dtype_mlir == mlir::ElementType::F16);
        if (upcast_std) {
          computation_dtype = mlir::ElementType::F32;
        }

        TT_ASSIGN_OR_THROW(const at::Tensor correction_tensor,
                           promoted_correction.GetTensor(
                               at::toOpMathType(std_scalar_dtype)));
        TT_ASSIGN_OR_THROW(const int64_t reduction_size,
                           GetReductionFactor(self, dim));
        const double correction_val = promoted_correction.scalar().toDouble();
        WarnIfInvalidCorrection(OpName::kStdMeanCorrection, correction_val,
                                reduction_size);

        auto std_mean_builder =
            [canonical_dims = std::move(canonical_dims), reduction_mode,
             reduction_size, keep_dim](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<std::array<mlir::MlirOp, 2>> {
          auto& [self_op, correction_op] = inputs;
          mlir::MlirBuilder& builder = self_op.getBuilder();

          TT_ASSIGN_OR_RETURN(
              const mlir::ElementType mean_type,
              ConvertTo<mlir::ElementType>(
                  GetTensorTypeOrDie(self_op).getElementType()));
          TT_ASSIGN_OR_RETURN(
              const mlir::ElementType var_type,
              ConvertTo<mlir::ElementType>(
                  GetTensorTypeOrDie(correction_op).getElementType()));

          // mean_keepdims = ReduceSum(self, dim, keep_dims=true) / n
          TT_ASSIGN_OR_RETURN(
              const mlir::MlirOp mean_keepdims,
              BuildMeanShlo(self_op, canonical_dims, ReductionMode::kKeepDims,
                            mean_type));

          // The mean output drops the reduced dimensions unless keep_dim.
          mlir::MlirOp mean_out = mean_keepdims;
          if (!keep_dim) {
            TT_ASSIGN_OR_RETURN(mean_out,
                                Squeeze(mean_keepdims, canonical_dims));
          }

          // diff = self - mean_keepdims
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp diff,
                              BuildSubShlo(self_op, mean_keepdims));

          // squared_diff = diff * conj(diff)
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp conj_diff,
                              BuildConjPhysicalShlo(diff));
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp squared_diff,
                              BuildMulShlo(diff, conj_diff));

          // sum_sq = ReduceSum(squared_diff, dim, keep_dims=keep_dim)
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp sum_sq,
                              BuildSumShlo(squared_diff, canonical_dims,
                                           reduction_mode, var_type));

          // denom = max(reduction_size - correction, 0)
          // If reduction_size == 0, degrees of freedom is 0 and the variance
          // is NaN.
          mlir::MlirOp denom;
          if (reduction_size == 0) {
            denom = MakeScalarConstant(builder, 0.0, var_type);
          } else {
            const mlir::MlirOp reduction_size_const = MakeScalarConstant(
                builder, static_cast<double>(reduction_size), var_type);
            TT_ASSIGN_OR_RETURN(
                const mlir::MlirOp denom_diff,
                BuildSubShlo(reduction_size_const, correction_op));
            const mlir::MlirOp zero =
                MakeScalarConstant(builder, 0.0, var_type);
            TT_ASSIGN_OR_RETURN(denom, BuildMaximumShlo(denom_diff, zero));
          }

          // var = sum_sq / denom
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp var,
                              BuildDivShlo(sum_sq, denom));

          // std = sqrt(var)
          TT_ASSIGN_OR_RETURN(const mlir::MlirOp std_op,
                              BuildSqrtShlo(var, var_type));

          return std::array<mlir::MlirOp, 2>{std_op, mean_out};
        };

        TT_ASSIGN_OR_THROW(
            (auto [std_buf, mean_buf]),
            (DispatchOp<2, 2>(
                std::move(std_mean_builder), {self, correction_tensor},
                {.out_dtypes = {std_dtype_mlir, self_dtype_mlir},
                 .out_dims_list = {output_dims, output_dims},
                 .computation_dtype = computation_dtype,
                 .op_param_cache_keys = std::move(param_keys)})));

        return std::make_tuple(MakeTensor(std_buf), MakeTensor(mean_buf));
      });
}

}  // namespace torch_tpu
