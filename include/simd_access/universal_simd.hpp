// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Universal simd class.
 */

#ifndef SIMD_ACCESS_UNIVERSAL_SIMD
#define SIMD_ACCESS_UNIVERSAL_SIMD

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include "simd_access/reflection.hpp"
#include <utility>
#include <array>
#include <type_traits>

namespace simd_access
{

/// Universal simd class for non-arithmetic value types.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct universal_simd : std::array<T, SimdModel::size()>
{
  /// Static version of `size()` (as provided by `stdx::simd`, but not by `std::array`).
  /**
   * @return `SimdSize`
   */
  static constexpr auto size() { return SimdModel::size(); }

  /// Generator constructor (as provided by `stdx::simd`).
  /**
   * The generator constructor constructs a universal simd where the i-th element is initialized to
   * `generator(std::integral_constant<std::size_t, i>())`.
   * @tparam G Deduced type of the generator functor.
   * @param generator Generator functor.
   */
  template<class G>
  explicit universal_simd(G&& generator) :
    std::array<T, size()>([&]<int... I>(std::integer_sequence<int, I...>) -> std::array<T, size()>
      {
        return {{ generator(std::integral_constant<int, I>())... }};
      } (std::make_integer_sequence<int, size()>())) {}

  /// Default constructor.
  universal_simd() = default;
};

/// Uses a generator to create a scalar value. Overloaded for simd indices.
/**
 * Unlike the `universal_simd` generator constructor the generator passed to `generate_universal` must handle a
 * run-time value index.
 * @param i Integral index value.
 * @param generator Generator functor.
 * @return The result of `generator(i)`.
 */
inline decltype(auto) generate_universal(std::integral auto i, auto&& generator)
{
  return generator(i);
}

/// Uses a generator to create a `universal_simd` value. Overloaded for scalar indices.
/**
 * Unlike the `universal_simd` generator constructor the generator passed to `generate_universal` must handle a
 * run-time value index.
 * @tparam IndexType Deduced type of the simd index.
 * @param idx Simd index.
 * @param generator Generator functor.
 * @return A `universal_simd` containing the values for the simd index.
 */
template<class IndexType>
  requires(!std::integral<IndexType>)
inline decltype(auto) generate_universal(const IndexType& idx, auto&& generator)
{
  return universal_simd<decltype(generator(scalar_index(idx, 0))), index_model_t<IndexType>>([&](auto i)
    {
      return generator(scalar_index(idx, i));
    });
}

/// Accesses a subobject (a member or a member function) of a scalar value. Overloaded for universal simd values.
/**
 * @tparam T Deduced non-simd type of the value.
 * @tparam Func Deduced type of the functor specifying the subobject.
 * @param v Scalar value.
 * @param subobject Functor accessing the subobject.
 * @return The result of `subobject(v)`.
 */
template<class T, class Func>
  requires(!any_simd<T>)
inline decltype(auto) universal_access(T&& v, Func&& subobject)
{
  return subobject(v);
}

/// Accesses a subobject (a member or a member function) of a universal simd value. Overloaded for scalar values.
/**
 * @tparam T Deduced value type of the universal simd.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @tparam Func Deduced type of the functor specifying the subobject.
 * @param v Simd value.
 * @param subobject Functor accessing the subobject of one entry in the simd value.
 * @return A simd value holding the result of the calls `subobject(v[i])` for each vector lane.
 */
template<class T, class SimdModel, class Func>
inline auto universal_access(const simd_access::universal_simd<T, SimdModel>& v, Func&& subobject)
{
  using ScalarType = decltype(subobject(static_cast<const std::unwrap_reference_t<T>&>(v[0])));
  decltype(simdized_value<SimdModel>(std::declval<ScalarType>())) result;
  for (size_t i = 0; i < SimdModel::size(); ++i)
  {
    simd_members([&](auto&& d, auto&& s){ d[i] = s; },
      result, subobject(static_cast<const std::unwrap_reference_t<T>&>(v[i])));
  }
  return result;
}

/// Generalized access of a subobject for scalar and universal simd values.
/**
 * `SIMD_UNIVERSAL_ACCESS` replaces the expression `value expr`, if value might be a universal simd value.
 * @param value A value, which is either a scalar or a universal simd value.
 * @param expr A token sequence, which forms a valid `v expr` sequence for a scalar `v`. If `value` is a universal simd,
 *   then `v` is an entry, otherwise `v` is `value`.
 */
#define SIMD_UNIVERSAL_ACCESS(value, expr) \
  simd_access::universal_access(value, [&](auto&& element) { return element expr; })


} //namespace simd_access

#endif //SIMD_ACCESS_UNIVERSAL_SIMD
