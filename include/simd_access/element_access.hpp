// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Elementwise access of simd elements.
 */

#ifndef SIMD_ACCESS_ELEMENT_ACCESS
#define SIMD_ACCESS_ELEMENT_ACCESS

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include <utility>

namespace simd_access
{

template<class T>
concept simd_accessible = simd_index<T> || any_simd<T>;

/// Return a single index of a simd index.
/**
 * @param index Simd index.
 * @param i Index in the simd index.
 * @return The index at the i'th position of `index`.
 */
inline auto element(const simd_index auto& index, size_t i)
{
  return scalar_index(index, i);
}

/// Return a single value of a simd value.
/**
 * @param value Simd value.
 * @param i Index in the simd value.
 * @return The scalar value at the i'th position of `value`.
 */
inline decltype(auto) element(any_simd auto&& value, size_t i)
{
  return value[i];
}

/// Returns a value.
/**
 * @param value Any value.
 * @return `value`.
 */
inline auto&& element(auto&& value)
{
  return value;
}

/**
 * Call a function for each scalar of a simd value.
 * @param fn Function called for each scalar value of `x` and `y`. If the function intends to write to the element,
 *   then it must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`.
 * @param x Simd value.
 * @param y More simd values.
 */
inline void elementwise(auto&& fn, simd_accessible auto&& x, simd_accessible auto&&... y)
{
  for (size_t i = 0; i < x.size(); ++i)
  {
    fn(element(x, i), element(y, i)...);
  }
}

/**
 * Call a function for each scalar of a simd value.
 * @param fn Function called for each scalar value of `x` and `y`. If the function intends to write to the element,
 *   then it must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`. The last argument of the
 *   call is the index (ranges from 0 to simd_size). In overloads for scalar values this argument is ommitted.
 * @param x Simd value.
 * @param y More simd values.
 */
inline void elementwise_with_index(auto&& fn, simd_accessible auto&& x, simd_accessible auto&&... y)
{
  for (size_t i = 0; i < x.size(); ++i)
  {
    fn(element(x, i), element(y, i)..., i);
  }
}

/**
 * Call a function for the scalar value `x`.
 * This overload can be used with a generic functor to uniformly access scalar values of simds and scalars.
 * @tparam T Deduced non-simd type of the scalar value.
 * @param fn Function called for `x`.
 * @param x Scalar value.
 * @param y More values.
 */
template<class T>
  requires (!simd_accessible<T>)
inline void elementwise(auto&& fn, T&& x, auto&&... y)
{
  fn(x, y...);
}

/**
 * Call a function for the scalar value `x`.
 * This overload can be used with a generic functor to uniformly access scalar values of simds and scalars.
 * @tparam T Deduced non-simd type of the scalar value.
 * @param fn Function called for `x`.
 * @param x Scalar value.
 * @param y More values.
 */
template<class T>
  requires (!simd_accessible<T>)
inline void elementwise_with_index(auto&& fn, T&& x, auto&&... y)
{
  fn(x, y...);
}

/**
 * Used in functors for `elementwise`, which write to the elements. The function and its overload is used
 * to distinguish between usual value types and simd::reference types.
 * @param x Element value.
 * @return Reference to `x`.
 */
inline auto& element_write(std::copyable auto& x)
{
  return x;
}

/**
 * Overloaded function intended to be called for simd::reference types, which are not copyable.
 * @tparam T Deduced type of the element value.
 * @param x Element value.
 * @return Rvalue reference to `x`, which can be used in assignment (since only
 *   `simd::reference::operator=(auto&&) &&` is available.
 */
template<class T>
inline std::remove_reference_t<T>&& element_write(T&& x)
{
  return static_cast<typename std::remove_reference<T>::type&&>(x) ;
}

/**
 * Returns a particular element of a simd type. This overload for non-simd types returns the passed value as
 * first element. For other elements the program is ill-formed.
 * @tparam I Number of the element. Must be 0.
 * @param x Value.
 * @return Constant reference to `x`.
 */
template<int I>
inline const auto& get_element(const auto& x)
{
  static_assert(I == 0);
  return x;
}

/**
 * Returns a particular element of a simd type.
 * @tparam I Number of the element. Must be smaller then `x.size()`.
 * @tparam ValueType Deduced simd type.
 * @param x Simd value.
 * @return Value of `x[I]`.
 */
template<int I, any_simd ValueType>
inline decltype(auto) get_element(const ValueType& x)
{
  static_assert(I < ValueType::size());
  return x[I];
}

} //namespace simd_access

#endif //SIMD_ACCESS_ELEMENT_ACCESS
