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
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class SimdModel>(stdx::simd<T, SimdModel>&){}(x); };

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
