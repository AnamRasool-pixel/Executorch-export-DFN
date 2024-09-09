#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor _nested_tensor_from_tensor_list(at::TensorList list, c10::optional<at::ScalarType> dtype=c10::nullopt, c10::optional<at::Layout> layout=c10::nullopt, c10::optional<at::Device> device=c10::nullopt, c10::optional<bool> pin_memory=c10::nullopt);
TORCH_API at::Tensor & _nested_tensor_from_tensor_list_out(at::TensorList list, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, at::Tensor & out);
} // namespace native
} // namespace at
