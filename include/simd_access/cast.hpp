// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief A helper type and a cast function for simdizing types in vectorized loops.
 */

#ifndef SIMD_ACCESS_CAST
#define SIMD_ACCESS_CAST

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include "simd_access/reflection.hpp"

namespace simd_access
{

/**
 * Provides a member `type`, which - depending on the index type - is the simdized type of the given template
 * parameter `T` or `T` itself for a scalar index type. Useful for type declarations inside a vectorized loop.
 * @tparam T Type to be optionally simdized.
 * @tparam IndexType Type of the index.
 */
template<class T, class IndexType>
struct simdized_by_index;

/// Specialization of \ref simdized_by_index for scalar index types.
/**
 * @tparam T Type.
 * @tparam IndexType Type of the index.
 */
template<class T, class IndexType> requires std::is_arithmetic_v<IndexType>
struct simdized_by_index<T, IndexType>
{
  using type = T;
};

/// Specialization of \ref simdized_by_index for simd index types.
/**
 * @tparam T Type.
 * @tparam IndexType Type of the index.
 */
template<class T, simd_index IndexType>
struct simdized_by_index<T, IndexType>
{
  using type = decltype(simdized_value<index_model_t<IndexType>>(std::declval<T>()));
};

/// Type which resolves either to `T` or - if `IndexType` is a simd index - to the simdized type of `T`.
/**
 * @tparam T Type to be optionally simdized.
 * @tparam IndexType Type of the index.
 */
template<class T, class IndexType>
using simdized_by_index_t = typename simdized_by_index<T, IndexType>::type;

/**
 * Explicitly casts a value to its simdized value depending on the type of the simd index. A conversion from a scalar
 * value to its simdized value must exist. This can be used to resolve dependent contexts.
 * @tparam IndexType Explicitly given type of the index.
 * @param value Value to be casted.
 * @return If IndexType is a simd index, then the simdized version of value (usually with its inner values broadcasted).
 *   Otherwise `value`.
 */
template<class IndexType>
auto simd_broadcast(auto&& value)
{
  if constexpr (simd_index<IndexType>)
  {
    return simdized_by_index_t<decltype(value), IndexType>(value);
  }
  else
  {
    return value;
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_CAST
