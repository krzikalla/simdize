// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

#include "simd_access/base.hpp"
#include "simd_access/location.hpp"

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdModel Simd type acting as type model.
 * @tparam IndexType Type of the scalar index.
 */
template<class SimdModel, class IndexType = size_t>
struct index
{
  /// Generalized access to `SimdModel`.
  using simd_model_type = SimdModel;

  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static auto size() { return SimdModel::size(); }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_ + i.
   */
  auto scalar_index(int i) const { return index_ + IndexType(i); }

  /// The index, at which the sequence starts.
  IndexType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to a consecutive sequence of elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<linear_location<T, SimdModel>, sizeof(T)>(linear_location<T, SimdModel>{data + index_});
  }

  /// Transforms this to a simd value.
  /**
   * @return The value represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return stdx::rebind_simd_t<IndexType, SimdModel>([this](auto i){ return IndexType(index_ + i); });
  }
};

template<class PotentialIndexType>
concept simd_index =
  (stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(std::remove_cvref_t<PotentialIndexType> x) { []<class SimdModel, class IndexType>(index<SimdModel, IndexType>&){}(x); };

/// TODO: Introduce masked_index to support e.g. residual masked loops.

/// Returns the scalar index of a specific vector lane for a linear index.
/**
 * @tparam SimdModel Simd type acting as type model.
 * @tparam IndexType Deduced type of the scalar index.
 * @param idx Linear simd index.
 * @param i Vector lane.
 * @return The scalar index at vector lane `i`, i.e. `idx.start + i`.
 */
template<class SimdModel, class IndexType>
inline auto scalar_index(const index<SimdModel, IndexType>& idx, auto i)
{
  return idx.scalar_index(i);
}

/// Returns the scalar index of a specific vector lane for an indirect index u.
/**
 * @tparam IndexType Deduced integral type of the scalar index.
 * @tparam SimdModel Deduced abi of the `simd` paramter.
 * @param idx Indirect simd index.
 * @param i Vector lane.
 * @return The scalar index at vector lane `i`, i.e. `idx[i]`.
 */
template<std::integral IndexType, class SimdModel>
inline auto scalar_index(const stdx::simd<IndexType, SimdModel>& idx, auto i)
{
  return idx[i];
}

/// Returns true, if the argument is a simd index (i.e. fullfills the concept `simd_index`).
/**
 * @param idx Potential simd index.
 * @return True, if the type of idx fulfills `simd_index` concept.
 */
constexpr inline auto is_simd_index(auto&& idx)
{
  return simd_index<decltype(idx)>;
}

template<class T>
struct index_model;

template<class SimdModel, class IndexType>
struct index_model<index<SimdModel, IndexType>>
{
  using type = SimdModel;
};

template<std::integral Index, class Abi>
struct index_model<stdx::simd<Index, Abi>>
{
  using type = stdx::simd<Index, Abi>;
};

template<simd_index T>
using index_model_t = typename index_model<T>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX
