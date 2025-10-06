// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions to support structure-of-simd layout
 *
 * The reflection API of simd_access enables the user to handle structures of simd variables. These so-called
 * structure-of-simd variables mimic their scalar counterpart. The user must provide two functions:
 * 1. `simdized_value` takes a scalar variable and returns a simdized (and potentially initialized)
 *   variable of the same type.
 * 2. `simd_members` takes a pack of scalar or simdized variables and iterates over all simdized members
 *   calling a functor.
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
///@cond
// Forward declarations
template<class MASK, class T>
struct where_expression;

template<class FN, simd_arithmetic... Types>
inline void simd_members(FN&& func, Types&&... values)
{
  func(values...);
}

template<class FN, stdx_simd... Types>
inline void simd_members(FN&& func, Types&&... values)
{
  func(values...);
}

template<class FN, stdx_simd DestType>
inline void simd_members(FN&& func, DestType& d, const typename DestType::value_type& s)
{
  func(d, s);
}

template<class FN, stdx_simd SrcType>
inline void simd_members(FN&& func, typename SrcType::value_type& d, const SrcType& s)
{
  func(d, s);
}

template<class SimdModel, simd_arithmetic T>
inline auto simdized_value(T)
{
  return stdx::rebind_simd_t<T, SimdModel>();
}

// overloads for std types, which can't be added after the template definition, since ADL wouldn't found it
template<class SimdModel, class T>
inline auto simdized_value(const std::vector<T>& v)
{
  std::vector<decltype(simdized_value<SimdModel>(std::declval<T>()))> result(v.size());
  return result;
}


template<simd_access::specialization_of<std::vector>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  auto&& d = std::get<0>(std::forward_as_tuple(std::forward<Args>(values)...));
  for (decltype(d.size()) i = 0, e = d.size(); i < e; ++i)
  {
    simd_members(func, values[i] ...);
  }
}

template<class SimdModel, class T, class U>
inline auto simdized_value(const std::pair<T, U>& v)
{
  return std::make_pair(simdized_value<SimdModel>(v.first), simdized_value<SimdModel>(v.second));
}

template<simd_access::specialization_of<std::pair>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  simd_members(func, values.first ...);
  simd_members(func, values.second ...);
}
///@endcond

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdModel Simd type acting as type model.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, class T, class SimdModel>
  requires (!simd_arithmetic<T>)
inline auto load(const linear_location<T, SimdModel>& location)
{
  auto result = simdized_value<SimdModel>(*location.base_);
  simd_members([&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(linear_location<std::remove_reference_t<decltype(src)>, SimdModel>{&src});
    },
    result, *location.base_);
  return result;
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Linear index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<class BaseType, simd_index IndexType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const IndexType& idx, auto&& subobject)
{
  decltype(simdized_value<index_model_t<IndexType>>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members([&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      },
      result, subobject(base[scalar_index(idx, i)]));
  }
  return result;
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Linear index.
 * @return A simd value.
 */
template<class BaseType, simd_index IndexType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const IndexType& idx)
{
  decltype(simdized_value<index_model_t<IndexType>>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members([&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      },
      result, base[scalar_index(idx, i)]);
  }
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address of the memory location, to which the first scalar element is about to be stored.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, class SimdModel>
  requires (!simd_arithmetic<T>)
inline void store(const linear_location<T, SimdModel>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdModel>(std::declval<T>()))& source = expr;
  simd_members([&](auto&& dest, auto&& src)
    {
      store<ElementSize>(
        linear_location<std::remove_reference_t<decltype(dest)>, SimdModel>{&dest}, src);
    },
    *location.base_, source);
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an indirect index. The simd
 * elements to be loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, class T, class SimdModel, class IndexArray>
  requires (!simd_arithmetic<T>)
inline auto load(const indexed_location<T, SimdModel, IndexArray>& location)
{
  auto result = simdized_value<SimdModel>(*location.base_);
  simd_members([&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(indexed_location<std::remove_reference_t<decltype(src)>, SimdModel, IndexArray>{
        &src, location.indices_});
    },
    result, *location.base_);
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an indirect index. The simd
 * elements are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address and indices of the memory location.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, class SimdModel, class IndexArray>
  requires (!simd_arithmetic<T>)
inline void store(const indexed_location<T, SimdModel, IndexArray>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdModel>(std::declval<T>()))& source = expr;
  simd_members([&](auto&& dest, auto&& src)
    {
      using location_type = indexed_location<std::remove_reference_t<decltype(dest)>, SimdModel, IndexArray>;
      store<ElementSize>(location_type{&dest, location.indices_}, src);
    }, *location.base_, source);
}

/**
 * Returns a `where_expression` for structure-of-simd types, which are unsupported by stdx::simd.
 * @tparam MASK Deduced type of the simd mask.
 * @tparam T Deduced structure-of-simd type.
 * @param mask The value of the simd mask.
 * @param dest Reference to the structure-of-simd value, to which `mask` is applied.
 * @return A `where_expression` combining `mask`and `dest`.
 */
template<class M, class T>
  requires((!simd_arithmetic<T>) && (!stdx_simd<T>))
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
  /// Simd mask.
  M mask_;
  /// Reference to the masked value.
  T& destination_;

  /// Constructor.
  /**
   * @param m Simd mask.
   * @param dest Reference to the masked value.
   */
  where_expression(const M& m, T& dest) :
    mask_(m),
    destination_(dest)
  {}

  /// Assignment operator. Only those vector lanes elements are assigned to destination, which are set in the mask.
  /**
   * @param source Value to be assigned.
   * @return Reference to this.
   */
  auto& operator=(const T& source) &&
  {
    simd_members([&](auto& d, const auto& s)
      {
        using stdx::where;
        using simd_access::where;
        where(mask_, d) = s;
      }, destination_, source);
    return *this;
  }
};

} //namespace simd_access

#endif //SIMD_REFLECTION
