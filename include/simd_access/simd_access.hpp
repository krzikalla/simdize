// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Helper functions and a macro to mark a simd access.
 */

#ifndef SIMD_ACCESS_MAIN
#define SIMD_ACCESS_MAIN

#include <concepts>

#include "simd_access/base.hpp"
#include "simd_access/cast.hpp"
#include "simd_access/element_access.hpp"
#include "simd_access/index.hpp"
#include "simd_access/load_store.hpp"
#include "simd_access/simd_loop.hpp"
#include "simd_access/reflection.hpp"
#include "simd_access/universal_simd.hpp"
#include "simd_access/value_access.hpp"

namespace simd_access
{

/// Helper class to distinguish between lvalues and rvalues in SIMD_ACCESS macro.
/**
 * This helper class is specialized for lvalues and rvalues by using `isLvalue`.
 * @tparam isLvalue True, if an lvalue.
 */
template<bool isLvalue>
struct LValueSeparator;

/// Specialization, if the expression in SIMD_ACCESS results in an lvalue.
template<>
struct LValueSeparator<true>
{
  /// Non-simd access to an array element.
  /**
   * @param base Array base.
   * @param i Integral index.
   * @return The result of the expression `b[i]`.
   */
  static decltype(auto) to_simd(auto&& base, std::integral auto i)
  {
    return base[i];
  }

  /// Non-simd access to a member of an array element.
  /**
   * @param base Array base.
   * @param i Integral index.
   * @param subobject A functor yielding a member of the array element.
   * @return The result of the expression `subobject(b[i])` (usually `b[i].member`).
   */
  static decltype(auto) to_simd(auto&& base, std::integral auto i, auto&& subobject)
  {
    return subobject(base[i]);
  }

  /// Computes the base address of a given array for a linear simd access.
  /**
   * @tparam SimdModel Simd type acting as type model.
   * @tparam IndexType Deduced integral type of the scalar index.
   * @param base_addr Array base.
   * @param i A linear simd index.
   * @return The address of the starting array element in the element sequence defined by `i`.
   */
  template<class SimdModel, std::integral IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdModel, IndexType>& i)
  {
    return &base_addr[i.index_];
  }

  /// Computes the base address of a given array for an indirect simd access using indices in `stdx::simd`.
  /**
   * @tparam IndexType Deduced integral type of the scalar index.
   * @tparam Abi Deduced abi of the `simd` paramter.
   * @param base_addr Array base.
   * @return The address of the first array element.
   */
  template<std::integral IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&)
  {
    return &base_addr[0];
  }

  /// Computes the base address of a member of array elements for an linear simd access.
  /**
   * @tparam SimdModel Simd type acting as type model.
   * @tparam IndexType Deduced integral type of the scalar index.
   * @param base_addr Array base.
   * @param i A linear simd index.
   * @param subobject A functor yielding a member of the array element.
   * @return The address of the member of the starting array element in the element sequence defined by `i`.
   */
  template<class SimdModel, std::integral IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdModel, IndexType>& i, auto&& subobject)
  {
    return &subobject(base_addr[i.index_]);
  }

  /// Computes the base address of a member of array elements for an indirect simd access using indices in `stdx::simd`.
  /**
   * @tparam IndexType Deduced integral type of the scalar index.
   * @tparam Abi Deduced abi of the `simd` paramter.
   * @param base_addr Array base.
   * @param subobject A functor yielding a member of the array element.
   * @return The address of the member of the first array element.
   */
  template<std::integral IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&, auto&& subobject)
  {
    return &subobject(base_addr[0]);
  }

  /// Creates a value access object for a linear simd access.
  /**
   * @tparam ElementSize Size of an array element.
   * @tparam T Deduced type of the simd-accessed element.
   * @tparam SimdModel Simd type acting as type model.
   * @tparam IndexType Deduced integral type of the scalar index.
   * @param base Pointer to the first element (or one of its members) in the sequence defined by `i`.
   * @return A value access object (see \ref value_access), which can be used as lhs in assignments.
   */
  template<size_t ElementSize, class T, class SimdModel, std::integral IndexType>
  static auto get_direct_value_access(T* base, const index<SimdModel, IndexType>&)
  {
    return make_value_access<ElementSize>(linear_location<T, SimdModel>{base});
  }

  /// Creates a value access object for an indirect simd access using indices in `stdx::simd`.
  /**
   * @tparam ElementSize Size of an array element.
   * @tparam T Deduced type of the simd-accessed element.
   * @tparam IndexType Deduced integral type of the scalar index.
   * @tparam Abi Deduced abi of the `simd` paramter.
   * @param base Pointer to the first array element or one of its members.
   * @param idx SIMD index.
   * @return A value access object (see \ref value_access), which can be used as lhs in assignments.
   */
  template<size_t ElementSize, class T, std::integral IndexType, class Abi>
  static auto get_direct_value_access(T* base, const stdx::simd<IndexType, Abi>& idx)
  {
    using location_type = indexed_location<T, stdx::simd<IndexType, Abi>, stdx::simd<IndexType, Abi>>;
    return make_value_access<ElementSize>(location_type{base, idx});
  }

  /// Creates a value access object for an arbitrary simd access.
  /**
   * @tparam IndexType Deduced type of the simd index.
   * @tparam Func Deduced optional functor for member accesses.
   * @param base Array base.
   * @param indices Simd index.
   * @param subobject Optional functor yielding a member of the array element.
   * @return A value access object (see \ref value_access), which can be used as lhs in assignments.
   */
  template<class IndexType, class... Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(auto&& base, const IndexType& indices, Func&&... subobject)
  {
    return get_direct_value_access<sizeof(decltype(base[0]))>(get_base_address(base, indices, subobject...), indices);
  }
};

/// Specialization, if the expression in SIMD_ACCESS results in an rvalue.
template<>
struct LValueSeparator<false>
{
  /// Non-simd access to an array element.
  /**
   * @param base Array base.
   * @param i Integral index.
   * @return The result of the expression `b[i]`.
   */
  static decltype(auto) to_simd(auto&& base, std::integral auto i)
  {
    return base[i];
  }

  /// Non-simd access to a member of an array element.
  /**
   * @param base Array base.
   * @param i Integral index.
   * @param subobject A functor yielding a member of the array element.
   * @return The result of the expression `subobject(b[i])` (usually `b[i].member`).
   */
  static decltype(auto) to_simd(auto&& base, std::integral auto i, auto&& subobject)
  {
    return subobject(base[i]);
  }

  /// Type of an array element.
  /**
   * @tparam T Array type.
   */
  template<class T>
  using BaseType = std::decay_t<decltype(std::declval<T>()[0])>;

  /// Type of a subobject (specified by Functor) of an array element.
  /**
   * @tparam T Array type.
   * @tparam Func Functor type for member accesses.
   */
  template<class T, class Func>
  using BaseTypeFn = std::decay_t<decltype(std::declval<Func>()(std::declval<T>()[0]))>;

  /// Creates a simd value for an arbitrary simd access.
  /**
   * @tparam T Deduced type of the simd-accessed element.
   * @tparam IndexType Deduced type of the simd index.
   * @param base Array base.
   * @param idx Simd index.
   * @return A simd value holding the `base` array elements defined by `idx`.
   */
  template<class T, class IndexType>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx)
  {
    return load_rvalue<BaseType<T>>(base, idx);
  }

  /// Creates a simd value for an arbitrary simd access to a member of array elements.
  /**
   * @tparam T Deduced type of the simd-accessed element.
   * @tparam IndexType Deduced type of the simd index.
   * @tparam Func Deduced functor for member accesses.
   * @param base Array base.
   * @param idx Simd index.
   * @param subelement Functor yielding a member of the array element.
   * @return A simd value holding the member values of `base` array elements defined by `idx`.
   */
  template<class T, class IndexType, class Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx, Func&& subelement)
  {
    return load_rvalue<BaseTypeFn<T, Func>>(base, idx, subelement);
  }
};


/**
 * This function mocks a globally overloaded operator[]. It can be used instead of the SIMD_ACCESS macro,
 * if no named subobject is accessed.
 * @tparam T Deduced type of the base array.
 * @tparam IndexType Deduced type of the index to the base array. Can be a simd index or an ordinary integral index.
 * @param base The base array.
 * @param index The index to the base array.
 * @return A simd access object suitable for the value category.
 */
template<class T, class IndexType>
inline decltype(auto) sa(T&& base, const IndexType& index)
{
  return LValueSeparator<std::is_lvalue_reference_v<decltype(base[0])>>::to_simd(base, index);
}

/// Unified generator function for a simd value.
/**
 * @tparam T Deduced type of the value. The type has a `to_simd()` member function.
 * @param value The value (an internal helper struct), which is converted to a simd value.
 * @return A simd value.
 */
template<has_to_simd T>
inline auto to_simd(const T& value)
{
  return value.to_simd();
}

/// Overloaded function to enable unified calls.
/**
 * @tparam T Deduced type of the value. The type doesn't have a `to_simd()` member function.
 * @param value Any value.
 * @return `value`.
 */
template<class T> requires(!has_to_simd<T>)
inline auto to_simd(T&& value)
{
  return value;
}

} //namespace simd_access

/// Macro for uniform access to variables for simd and scalar indices. Usable as lvalue and rvalue.
/**
 * This macro defines a simd access to a variable of the form `base[index] subobject` (subobject is optional).
 * TODO: If global operator[] overloading becomes possible, then a decomposition of `base[index]` isn't required
 * anymore. Direct accesses (`base[index]`) and accesses to sub-arrays (`base[index][subindex]`) can be directly
 * written then.
 * TODO: If operator.() overloading becomes possible too, then this macro becomes obsolete, since all expressions can
 * be directly written.
 * @param base The base array.
 * @param index The index to the base array. Can be a simd index or an ordinary integral index.
 * @param ... Possible accessors to data members or elements of a subarray.
 */
#define SIMD_ACCESS(base, index, ...) \
  simd_access::LValueSeparator<std::is_lvalue_reference_v<decltype((base[0] __VA_ARGS__))>>:: \
    to_simd(base, index __VA_OPT__(, [&](auto&& e) -> decltype((e __VA_ARGS__)) { return e __VA_ARGS__; }))

/// Macro for uniform access to variables for simd and scalar indices returning an rvalue.
/**
 * Use in deduced contexts (e.g. a call to a template function), which expect a scalar value or a `stdx::simd`.
 */
#define SIMD_ACCESS_V(...) simd_access::to_simd(SIMD_ACCESS(__VA_ARGS__))

#endif //SIMD_ACCESS_MAIN
