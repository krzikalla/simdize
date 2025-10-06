// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining locations of simdized data.
 */

#ifndef SIMD_ACCESS_LOCATION
#define SIMD_ACCESS_LOCATION

#include <type_traits>

namespace simd_access
{

/// Specifies a location for a simd variable stored in memory as a consecutive sequence of elements.
/**
 * @tparam T Value type of the simd variable.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct linear_location
{
  /// Generalized access to `T`.
  using value_type = T;
  /// Generalized access to `SimdModel`.
  using simd_model_type = SimdModel;
  /// Pointer to the first element of the sequence.
  T* base_;

  /// Experimental creation of a linear location for a member of `T`.
  /**
   * @tparam Member Pointer to a member variable of `T`.
   * @return A new `linear_location` with `base->*Member` as first element.
   */
  template<auto Member>
  auto member_access() const
  {
    return linear_location<std::remove_reference_t<decltype(std::declval<T>().*Member)>, SimdModel>{&(base_->*Member)};
  }

  /// Creation of a linear location for an element of `T`, if `T` is an array.
  /**
   * @param i Array element index.
   * @return A new `linear_location` with `(*base)[i]` as first element.
   */
  auto array_access(auto i) const
  {
    return linear_location<std::remove_reference_t<decltype((*base_)[i])>, SimdModel>{&((*base_)[i])};
  }
};

/// Specifies a location for a simd variable with the indices of its values stored in an array.
/**
 * @tparam T Value type of the simd variable.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ArrayType Type of the array, which stores the indices.
 */
template<class T, class SimdModel, class ArrayType>
struct indexed_location
{
  /// Generalized access to `T`.
  using value_type = T;
  /// Generalized access to `SimdModel`.
  using simd_model_type = SimdModel;
  /// Pointer to element zero of the sequence.
  T* base_;
  /// Reference to the index array.
  const ArrayType& indices_;

  /// Experimental creation of an indexed location for a member of `T`.
  /**
   * @tparam Member Pointer to a member variable of `T`.
   * @return A new `indexed_location` with `base->*Member` as element zero.
   */
  template<auto Member>
  auto member_access() const
  {
    return indexed_location<std::remove_reference_t<decltype(std::declval<T>().*Member)>, SimdModel, ArrayType>
      {&(base_->*Member), indices_};
  }

  /// Creation of an indexed location for an element of `T`, if `T` is an array.
  /**
   * @param i Array element index.
   * @return A new `indexed_location` with `(*base)[i]` as element zero.
   */
  auto array_access(auto i) const
  {
    return indexed_location<std::remove_reference_t<decltype((*base_)[i])>, SimdModel, ArrayType>
      {&((*base_)[i]), indices_};
  }
};

} //namespace simd_access

#endif //SIMD_ACCESS_LOCATION



