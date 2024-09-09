#pragma once

// @generated by torchgen/gen.py from Function.h

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/TensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>



#include <ATen/ops/_efficientzerotensor_ops.h>

namespace at {


// aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor _efficientzerotensor(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::_efficientzerotensor::call(c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor _efficientzerotensor(at::IntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::_efficientzerotensor::call(c10::fromIntArrayRefSlow(size), optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor _efficientzerotensor(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::_efficientzerotensor::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor _efficientzerotensor(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::_efficientzerotensor::call(c10::fromIntArrayRefSlow(size), dtype, layout, device, pin_memory);
  }
}

// aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor _efficientzerotensor_symint(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::_efficientzerotensor::call(size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor _efficientzerotensor(c10::SymIntArrayRef size, at::TensorOptions options={}) {
    return at::_ops::_efficientzerotensor::call(size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
  }
}

// aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
inline at::Tensor _efficientzerotensor_symint(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::_efficientzerotensor::call(size, dtype, layout, device, pin_memory);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor _efficientzerotensor(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at::_ops::_efficientzerotensor::call(size, dtype, layout, device, pin_memory);
  }
}

// aten::_efficientzerotensor.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _efficientzerotensor_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::_efficientzerotensor_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & _efficientzerotensor_out(at::Tensor & out, at::IntArrayRef size) {
    return at::_ops::_efficientzerotensor_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::_efficientzerotensor.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _efficientzerotensor_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::_efficientzerotensor_out::call(c10::fromIntArrayRefSlow(size), out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
  at::Tensor & _efficientzerotensor_outf(at::IntArrayRef size, at::Tensor & out) {
    return at::_ops::_efficientzerotensor_out::call(c10::fromIntArrayRefSlow(size), out);
  }
}

// aten::_efficientzerotensor.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _efficientzerotensor_symint_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::_efficientzerotensor_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & _efficientzerotensor_out(at::Tensor & out, c10::SymIntArrayRef size) {
    return at::_ops::_efficientzerotensor_out::call(size, out);
  }
}

// aten::_efficientzerotensor.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
inline at::Tensor & _efficientzerotensor_symint_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::_efficientzerotensor_out::call(size, out);
}
namespace symint {
  template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
  at::Tensor & _efficientzerotensor_outf(c10::SymIntArrayRef size, at::Tensor & out) {
    return at::_ops::_efficientzerotensor_out::call(size, out);
  }
}

}
