// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Helper functions and a macro to mark a simd access.
 */

#ifndef SIMD_ACCESS_MAIN
#define SIMD_ACCESS_MAIN

#include <concepts>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief A helper type and a cast function for simdizing types in vectorized loops.
 */

#ifndef SIMD_ACCESS_CAST
#define SIMD_ACCESS_CAST

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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
  auto scalar_index(auto i) const { return index_ + IndexType(i); }

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Elementwise access of simd elements.
 */

#ifndef SIMD_ACCESS_ELEMENT_ACCESS
#define SIMD_ACCESS_ELEMENT_ACCESS

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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
inline auto element(const simd_index auto& index, int i)
{
  return scalar_index(index, i);
}

/// Return a single value of a simd value.
/**
 * @param value Simd value.
 * @param i Index in the simd value.
 * @return The scalar value at the i'th position of `value`.
 */
inline decltype(auto) element(any_simd auto&& value, int i)
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
  for (int i = 0; i < x.size(); ++i)
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
  for (int i = 0; i < x.size(); ++i)
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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <experimental/bits/simd.h>

namespace simd_access
{

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @tparam Abi Deduced Abi tag type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class Abi>
inline void store(const linear_location<T, SimdModel>& location, const stdx::simd<T, Abi>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (int i = 0; i < source.size(); ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @tparam Abi Deduced Abi tag type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class Abi, class ArrayType>
inline void store(const indexed_location<T, SimdModel, ArrayType>& location, const stdx::simd<T, Abi>& source)
{
  // scatter with indirect indices
  for (int i = 0; i < source.size(); ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * location.indices_[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel>
inline auto load(const linear_location<T, SimdModel>& location)
{
  using ResultType = stdx::rebind_simd_t<std::remove_const_t<T>, SimdModel>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(location.base_, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(location.base_) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class ArrayType>
inline auto load(const indexed_location<T, SimdModel, ArrayType>& location)
{
  using ResultType = stdx::rebind_simd_t<std::remove_const_t<T>, SimdModel>;
  // gather with indirect indices
  return ResultType([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx)
{
  using ResultType = stdx::rebind_simd_t<BaseType, index_model_t<IndexType>>;
  return ResultType([&](auto i) { return base[scalar_index(idx, i)]; });
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx, auto&& subobject)
{
  using ResultType = stdx::rebind_simd_t<BaseType, index_model_t<IndexType>>;
  return ResultType([&](auto i)
  {
    return subobject(base[scalar_index(idx, i)]);
  });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions looping over a given function in simd-style in a linear or indirect fashion.
 */

#ifndef SIMD_ACCESS_LOOP
#define SIMD_ACCESS_LOOP

#include <concepts>
#include <experimental/bits/simd.h>
#include <type_traits>
// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

namespace simd_access
{

/// Type for scalar residual loop policy.
using ScalarResidualLoopT = std::integral_constant<int, 0>;
/// Type for vector residual loop policy.
using VectorResidualLoopT = std::integral_constant<int, 1>;
/// Value for scalar residual loop policy.
constexpr auto ScalarResidualLoop = ScalarResidualLoopT();
/// Value for vector residual loop policy.
constexpr auto VectorResidualLoop = VectorResidualLoopT();

/**
 * Linear simd-ized iteration over a function. The function is first called with a simd index and the remainder
 * loop is called with an integral index.
 * @tparam SimdModel Simd type acting as type model.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<class SimdModel, auto ... Args, typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(std::integral auto start, std::integral auto end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdModel, IndexType> simd_i{IndexType(start)};
  const auto simdSize = simd_i.size();
  const auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : simdSize;
  for (; simd_i.index_ + simdSize < end + endOffset; simd_i.index_ += simdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (IndexType i = simd_i.index_; i < end; ++i)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(i);
      }
      else
      {
        fn.template operator()<Args...>(i);
      }
    }
  }
}

/**
 * Linear simd-ized iteration over a function. The function is first called with an integral index until the
 * `alignTestFn` returns true for a specific index. From there on `alignTestFn` isn't called anymore and the function
 * is called with a simd index. The remainder loop is called with an integral index again.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param alignTestFn Generic function to be called. Takes one scalar argument of the common type of `start` and `end`.
 *   Once it returns true, it isn't called anymore and the function starts to call `fn` with simd indices
 *   (including the index for which `alignTestFn` returned `true`).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 */
template<class SimdModel, auto ... Args>
inline void aligning_loop(std::integral auto start, std::integral auto end, auto&& alignTestFn, auto&& fn)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdModel, IndexType> simd_i{IndexType(start)};
  for (; simd_i.index_ < end && !alignTestFn(simd_i.index_); ++simd_i.index_)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i.index_);
    }
    else
    {
      fn.template operator()<Args...>(simd_i.index_);
    }
  }
  const auto simdSize = simd_i.size();
  for (; simd_i.index_ + simdSize < end + 1; simd_i.index_ += simdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  for (; simd_i.index_ < end; ++simd_i.index_)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i.index_);
    }
    else
    {
      fn.template operator()<Args...>(simd_i.index_);
    }
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an stdx::simd
 * and the remainder loop is called with an integral index.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes one argument, whose type is either
 *   `stdx::simd<IntegralType, SimdSize>` or `IntegralType` (which is `*start`).
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<class SimdModel, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  size_t i = 0, i_end = end - start;
  using SimdIndexType = stdx::rebind_simd_t<std::decay_t<decltype(*start)>, SimdModel>;
  const auto simdSize = SimdIndexType::size();
  const auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : simdSize;
  for (; i + simdSize < i_end + endOffset; i += simdSize)
  {
    SimdIndexType simd_i([&](auto j) { return *(start + i + j); });
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (; i < i_end; ++i)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(*(start + i));
      }
      else
      {
        fn.template operator()<Args...>(*(start + i));
      }
    }
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an stdx::simd
 * and the remainder loop is called with an integral index.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes two arguments. The first is the linear index starting at 0, its
 *   type is either `index<SimdSize, size_t>` or `size_t`. The second argument is the indirect index, its type is
 *   either `stdx::simd<IntegralType, SimdSize>` or `IntegralType` (which is `*start`).
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<class SimdModel, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop_with_linear_index(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  size_t i_end = end - start;
  index<SimdModel, size_t> i{0};
  const auto simdSize = i.size();
  const auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : simdSize;
  for (; i.index_ + simdSize < i_end + endOffset; i.index_ += simdSize)
  {
    using SimdIndexType = stdx::rebind_simd_t<std::decay_t<decltype(*start)>, SimdModel>;
    SimdIndexType simd_i([&](auto j) { return *(start + i.index_ + j); });
    if constexpr (sizeof...(Args) == 0)
    {
      fn(i, simd_i);
    }
    else
    {
      fn.template operator()<Args...>(i, simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (; i.index_ < i_end; ++i.index_)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(i.index_, *(start + i.index_));
      }
      else
      {
        fn.template operator()<Args...>(i.index_, *(start + i.index_));
      }
    }
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Universal simd class.
 */

#ifndef SIMD_ACCESS_UNIVERSAL_SIMD
#define SIMD_ACCESS_UNIVERSAL_SIMD

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Class for accessing simd values via simd indices.
 */

#ifndef SIMD_ACCESS_VALUE_ACCESS
#define SIMD_ACCESS_VALUE_ACCESS

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Base class for operator overloads
 * This file demonstrates a possible approach to overload the member access operator.
 */

#ifndef SIMD_ACCESS_OPERATOR_OVERLOAD
#define SIMD_ACCESS_OPERATOR_OVERLOAD

#include <type_traits>
// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

namespace simd_access
{

/// Base class for overloaded member operator.
/**
 * This CRTP base class is needed, if the member operator is overloaded and the types, whose members are potentially
 * accessed might not be classes.
 * @tparam T Type of the class, whose members are accessed.
 * @tparam Derived subclass of this class. It provides a member function `dot_overload` using a less
 * restrictive template syntax.
 */
template<typename T, typename Derived>
struct member_overload;

/// Empty specialization for non-class types. Avoids syntax errors on the template parameter for any non-class T.
template<typename T, typename Derived> requires (!std::is_class_v<T>)
struct member_overload<T, Derived> {};

/// Specialization for class types.
template<typename T, typename Derived> requires (std::is_class_v<T>)
struct member_overload<T, Derived>
{
  /// Overloaded member operator, i.e. operator.()
  /**
   * An overload of `operator.()` for data members must have exactly the form below. The type of the object, whose
   * member is accessed, must be specified in the template argument to enable the compiler to look up for
   * the appropriate member in the expression using the overloaded `operator.()`.
   * @tparam Member Pointer to the member of the class T.
   * @return The value returned by `Derived::dot_overload`.
   */
  template<auto T::*Member>
  auto dot()
  {
    return (static_cast<Derived*>(this))->template dot_overload<Member>();
  }
};

/// Base class for overloaded type cast operator.
/**
 * This CRTP base class restricts the availability of the cast `operator auto()` to value types, which can be simd
 * value types. This prevents instantiation errors (gcc #115216).
 * @tparam T Type of scalar value.
 * @tparam Derived subclass of this class. It provides a member function `to_simd`.
 */
template<typename T, typename Derived>
struct cast_overload;

/// Empty specialization for non-simd_arithmetic types. There is no implicit cast to a simd type.
template<typename T, typename Derived> requires(!simd_arithmetic<T>)
struct cast_overload<T, Derived> {};

/// Specialization for simd_arithmetic types.
template<simd_arithmetic T, typename Derived>
struct cast_overload<T, Derived>
{
  /// Cast operator to a simd value.
  /** Transforms this to a simd value.
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  operator auto() const
  {
    return (static_cast<const Derived*>(this))->to_simd();
  }
};

} //namespace simd_access

#endif //SIMD_ACCESS_OPERATOR_OVERLOAD

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, class SimdModel>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x)
    { []<class T, class SimdModel>(universal_simd<T, SimdModel>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdModel>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdModel Simd type acting as type model.
 */
template<simd_arithmetic T, class SimdModel>
struct auto_simd<T, SimdModel>
{
  /// stdx::simd type.
  using type = stdx::rebind_simd_t<T, SimdModel>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdModel Simd type acting as type model.
 */
template<class T, class SimdModel>
using auto_simd_t = typename auto_simd<T, SimdModel>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <experimental/bits/simd.h>

namespace simd_access
{

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @tparam Abi Deduced Abi tag type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class Abi>
inline void store(const linear_location<T, SimdModel>& location, const stdx::simd<T, Abi>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (int i = 0; i < source.size(); ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @tparam Abi Deduced Abi tag type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class Abi, class ArrayType>
inline void store(const indexed_location<T, SimdModel, ArrayType>& location, const stdx::simd<T, Abi>& source)
{
  // scatter with indirect indices
  for (int i = 0; i < source.size(); ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * location.indices_[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam SimdModel Deduced simd type acting as type model.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel>
inline auto load(const linear_location<T, SimdModel>& location)
{
  using ResultType = stdx::rebind_simd_t<std::remove_const_t<T>, SimdModel>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(location.base_, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(location.base_) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdModel Simd type acting as type model.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, class SimdModel, class ArrayType>
inline auto load(const indexed_location<T, SimdModel, ArrayType>& location)
{
  using ResultType = stdx::rebind_simd_t<std::remove_const_t<T>, SimdModel>;
  // gather with indirect indices
  return ResultType([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx)
{
  using ResultType = stdx::rebind_simd_t<BaseType, index_model_t<IndexType>>;
  return ResultType([&](auto i) { return base[scalar_index(idx, i)]; });
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx, auto&& subobject)
{
  using ResultType = stdx::rebind_simd_t<BaseType, index_model_t<IndexType>>;
  return ResultType([&](auto i)
  {
    return subobject(base[scalar_index(idx, i)]);
  });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE

namespace simd_access
{

/// Creates a binary operator overload for a simd value access as member.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_BIN_OP( op ) \
  auto operator op(const auto& source) { return to_simd() op source; }

/// Creates a binary assignment operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_BIN_ASSIGNMENT_OP( op ) \
  void operator op##=(const auto& source) && { store<ElementSize>(location_, to_simd() op source); }

/// Creates a binary assignment and a binary operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_MEMBER_OPS( op ) \
  VALUE_ACCESS_BIN_OP( op ) \
  VALUE_ACCESS_BIN_ASSIGNMENT_OP( op )

/// Creates a global binary operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_SCALAR_BIN_OP( op ) \
  template<class Location, size_t ElementSize> \
  inline auto operator op(const auto& o1, const value_access<Location, ElementSize>& o2) \
  { \
    return o1 op o2.to_simd(); \
  }

/// Class representing a simd-access (read or write) to a memory location.
/**
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 * @tparam Location Type of the location of the simd data.
 */
template<class Location, size_t ElementSize>
class value_access :
  public member_overload<typename Location::value_type, value_access<Location, ElementSize>>,
  public cast_overload<typename Location::value_type, value_access<Location, ElementSize>>
{
public:
  /// Assignment operator.
  /** Writes a simd value to the simdized memory location represented by this.
   * Implicit conversion is not supported, it should be done explicit by the user.
   * Assign chains are not supported (i.e. the operator returns nothing).
   * @param source Simd value, whose content is written.
   */
  void operator=(const auto& source) &&
  {
    store<ElementSize>(location_, source);
  }

  VALUE_ACCESS_MEMBER_OPS(+)
  VALUE_ACCESS_MEMBER_OPS(-)
  VALUE_ACCESS_MEMBER_OPS(*)
  VALUE_ACCESS_MEMBER_OPS(/)

  /// Transforms this to a simd value.
  /**
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return load<ElementSize>(location_);
  }

  /// Experimental implementation of overloaded member operator, i.e. operator.()
  /**
   * Since `T` might be of non-class type, one cannot specify `auto T::*Member` as a template argument here.
   * @tparam Member Pointer to the member of the class T.
   * @return A `value_access` with a base address pointing to the accessed member of `base_`.
   */
  template<auto Member>
  auto dot_overload()
  {
    return make_value_access<ElementSize>(location_.template member_access<Member>());
  }

  /// Implementation of subscript operator for accesses to sub-array elements of `Location::value_type`.
  /**
   * @param i An index usable as index type for sub-array elements of `Location::value_type`.
   * @return A `value_access` with a base address pointing to the accessed array element of `base_`.
   */
  auto operator[](auto i) const
  {
    return make_value_access<ElementSize>(location_.array_access(i));
  }

  /// Constructor.
  /**
   * @param location The location specification of the accessed simd variable.
   */
  value_access(const Location& location) :
    location_(location)
  {}

private:
  /// The location specification of the accessed simd variable.
  Location location_;
};

/// Factory function for `value_access`.
/**
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 * @tparam Location Deduced type of the location of the simd data.
 * @param location The location specification of the accessed simd variable.
 * @return A `value_access` object representing a simd-access (read or write) to a memory location.
 */
template<size_t ElementSize, class Location>
inline auto make_value_access(const Location& location)
{
  return value_access<Location, ElementSize>(location);
}

VALUE_ACCESS_SCALAR_BIN_OP(+)
VALUE_ACCESS_SCALAR_BIN_OP(-)
VALUE_ACCESS_SCALAR_BIN_OP(*)
VALUE_ACCESS_SCALAR_BIN_OP(/)

template<class PotentialSimdType>
concept has_to_simd =
  requires(PotentialSimdType x) { x.to_simd(); };

} //namespace simd_access

#endif //SIMD_ACCESS_VALUE_ACCESS

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
