// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions to support structure-of-simd layout
 *
 * The reflection API of simd_access enables the user to handle structures of simd variables. These so-called
 * structure-of-simd variables mimic their scalar counterpart. The user must provide two functions:
 * 1. `simdized_value` takes a scalar variable and returns a simdized (and potentially initialized)
 *   variable of the same type.
 * 2. `simd_members` takes two scalar or simdized variables and iterates over all simdized members calling a functor.
 */

#ifndef SIMD_REFLECTION
#define SIMD_REFLECTION

#include <utility>
#include <vector>

#include "simd_access/base.hpp"
#include "simd_access/location.hpp"
#include "simd_access/index.hpp"

namespace simd_access
{

// Forward declarations
template<class MASK, class T>
struct where_expression;

template<class FN, is_stdx_simd DestType, is_stdx_simd SrcType>
inline void simd_members(DestType& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd DestType>
inline void simd_members(DestType& d, const typename DestType::value_type& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd SrcType>
inline void simd_members(typename SrcType::value_type& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<int SimdSize, simd_arithmetic T>
inline auto simdized_value(T)
{
  return stdx::fixed_size_simd<T, SimdSize>();
}

// overloads for std types, which can't be added after the template definition, since ADL wouldn't found it
template<int SimdSize, class T>
inline auto simdized_value(const std::vector<T>& v)
{
  std::vector<decltype(simdized_value<SimdSize>(std::declval<T>()))> result(v.size());
  return result;
}

template<class DestType, class SrcType, class FN>
inline void simd_members(std::vector<DestType>& d, const std::vector<SrcType>& s, FN&& func)
{
  for (auto e = d.size(), i = 0; i < e; ++i)
  {
    simd_members(d[i], s[i], func);
  }
}

template<int SimdSize, class T, class U>
inline auto simdized_value(const std::pair<T, U>& v)
{
  return std::make_pair(simdized_value<SimdSize>(v.first), simdized_value<SimdSize>(v.second));
}

template<class DestType1, class DestType2, class SrcType1, class SrcType2, class FN>
inline void simd_members(std::pair<DestType1, DestType2>& d, const std::pair<SrcType1, SrcType2>& s, FN&& func)
{
  simd_members(d.first, s.first, func);
  simd_members(d.second, s.second, func);
}


/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize>
  requires (!simd_arithmetic<T>)
inline auto load(const linear_location<T, SimdSize>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(linear_location<std::remove_reference_t<decltype(src)>, SimdSize>(&src));
    });
  return result;
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx, auto&& subobject)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (int i = 0; i < idx.size(); ++i)
  {
    simd_members(result, subobject(base[get_index(idx, i)]), [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (int i = 0; i < idx.size(); ++i)
  {
    simd_members(result, base[get_index(idx, i)], [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address of the memory location, to which the first scalar element is about to be stored.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize>
  requires (!simd_arithmetic<T>)
inline void store(const linear_location<T, SimdSize>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(
        linear_location<std::remove_reference_t<decltype(dest)>, SimdSize>(&dest), src);
    });
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an indirect index. The simd
 * elements to be loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline auto load(const indexed_location<T, SimdSize, IndexArray>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(indexed_location<std::remove_reference_t<decltype(src)>, SimdSize, IndexArray>(
        &src, location.indices_));
    });
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an indirect index. The simd
 * elements are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address and indices of the memory location.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline void store(const indexed_location<T, SimdSize, IndexArray>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(indexed_location<std::remove_reference_t<decltype(dest)>, SimdSize, IndexArray>(&dest, location.indices_), src);
    });
}

/**
 * Returns a `where_expression` for structure-of-simd types, which are unsupported by stdx::simd.
 * @tparam MASK Deduced type of the simd mask.
 * @tparam T Deduced structure-of-simd type.
 * @param mask The value of the simd mask.
 * @param dest Reference to the structure-of-simd value, to which `mask` is applied.
 */
template<class M, class T>
  requires((!simd_arithmetic<T>) && (!is_stdx_simd<T>))
inline auto where(const M& mask, T& dest)
{
  return where_expression<M, T>(mask, dest);
}

/**
 * Extends `stdx::where_expression` for structure-of-simd types.
 * @tparam M Type of the simd mask.
 * @tparam T Structure-of-simd type.
 */
template<class M, class T>
struct where_expression
{
  M mask_;
  T& destination_;

  where_expression(const M& m, T& dest) :
    mask_(m),
    destination_(dest)
  {}

  auto& operator=(const T& source) &&
  {
    simd_members(destination_, source, [&](auto& d, const auto& s)
      {
        using stdx::where;
        using simd_access::where;
        where(mask_, d) = s;
      });
    return *this;
  }
};

} //namespace simd_access

#endif //SIMD_REFLECTION