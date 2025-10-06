
#include "benchmark/benchmark.h"
#include <experimental/bits/simd.h>
#include <vector>
#include <iostream>

#include "helper_bm.hpp"
#include "simd_access/simd_access.hpp"


void Reduce_Scalar(benchmark::State& state)
{
  auto arraySize = state.range(0);
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    double result = .0;
    for (size_t i = 0, e = testData.size(); i < e; ++i)
    {
      result += dataPtr[i];
    }
    benchmark::DoNotOptimize(result);
    assert(result == arraySize * (arraySize + 1) / 2.0);
  }
}


namespace my_reduce
{

auto reduce(const simd_access::simd_arithmetic auto& x, auto)
{
  return x;
}

}


void Reduce_SimpleSimd(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    double result = .0;
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
    {
      using stdx::reduce;
      using my_reduce::reduce;
      result += reduce(SIMD_ACCESS_V(dataPtr, i), std::plus{});
    });
    benchmark::DoNotOptimize(result);
    assert(result == arraySize * (arraySize + 1) / 2.0);
  }
}


void Reduce_SophisticatedSimd(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    auto p = testData.data();
    SimdModel intermediateResult(.0);
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
    {
      if constexpr (simd_access::is_simd_index(i))
      {
        intermediateResult += SIMD_ACCESS(dataPtr, i);
      }
      else
      {
        intermediateResult[0] += dataPtr[i];
      }
    });
    double result = stdx::reduce(intermediateResult, std::plus{});
    benchmark::DoNotOptimize(result);
    assert(result == arraySize * (arraySize + 1) / 2.0);
  }
}


#define BM_READ( name ) BENCHMARK( name )->Unit(benchmark::kMicrosecond)->Arg(103)->Arg(4003)

BM_READ(Reduce_Scalar);
BM_READ(Reduce_SimpleSimd);
BM_READ(Reduce_SophisticatedSimd);
