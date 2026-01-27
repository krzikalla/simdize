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
#include "simd_access/index.hpp"

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
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdModel, IntegralType>`
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
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdModel, IntegralType>`
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
 *   type is either `index<SimdModel, size_t>` or `size_t`. The second argument is the indirect index, its type is
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

/**
 * Linear two-dimensional simd-ized iteration over a function. The loop consist of an outer and an inner loop.
 * First, the function is called with a simd index for the outer loop, but all remainding iterations are left out.
 * Then, the function is called with a simd index for the inner loop and for the remainding iterations with two
 * integral indices.
 * @tparam SimdModel Simd type acting as type model.
 * @param start_outer Start of the iteration range [start, end) of the outer loop.
 * @param end_outer End of the iteration range [start, end) of the outer loop.
 * @param start_inner Start of the iteration range [start, end) of the inner loop.
 * @param end_inner End of the iteration range [start, end) of the inner loop.
 * @param fn Generic function to be called. Takes two arguments, whose types are either `index<SimdModel, IntegralType>`
 *   or `IntegralType`. The first argument corresponds to the outer loop, the argument corresponds to the inner loop.
 */
template<class SimdModel, auto ... Args>
inline void loop_2d(std::integral auto start_outer, std::integral auto end_outer,
  std::integral auto start_inner, std::integral auto end_inner, auto&& fn)
{
  using IndexTypeOuter = std::common_type_t<decltype(start_outer), decltype(start_outer)>;
  using IndexTypeInner = std::common_type_t<decltype(start_inner), decltype(end_inner)>;
  const auto simdSize = SimdModel::size();
  auto aligned_end_outer = start_outer + ((end_outer - start_outer) / simdSize) * simdSize;
  loop<SimdModel, Args...>(start_outer, aligned_end_outer, [&](auto i)
  {
    for (IndexTypeInner j = start_inner; j < end_inner; ++j)
    {
      fn(i, j);
    }
  }, VectorResidualLoop);
  for (IndexTypeOuter i = aligned_end_outer; i < end_outer; ++i)
  {
    loop<SimdModel, Args...>(start_inner, end_inner, [&](auto j)
    {
      fn(i, j);
    });
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP
