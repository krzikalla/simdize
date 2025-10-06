
#include "benchmark/benchmark.h"
#include <vector>
#include <iostream>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace sa = simd_access;

struct PointID
{
  double* data_;
  int     idx_;

  auto GetValue() const { return data_[idx_]; }
};

struct PointIDCollection
{
  auto GetPoint(int i) const
  {
    return PointID{data_, i};
  }

  double* data_;
};

void UniversalSimd(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<double> testData(arraySize, .0);
  benchmark::DoNotOptimize(testData.data());
  PointIDCollection collection{testData.data()};
  for (auto _ : state)
  {
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
      {
        auto point = sa::generate_universal(i, [&](auto idx) { return collection.GetPoint(idx); });
        auto result = SIMD_UNIVERSAL_ACCESS(point, .GetValue());
        benchmark::DoNotOptimize(result);
      });
  }
}

void UniversalScalar(benchmark::State& state)
{
  auto arraySize = state.range(0);
  std::vector<double> testData(arraySize, .0);
  benchmark::DoNotOptimize(testData.data());
  PointIDCollection collection{testData.data()};
  for (auto _ : state)
  {
    for (int i = 0; i < testData.size(); ++i)
    {
      const auto& point = sa::generate_universal(i, [&](auto idx) { return collection.GetPoint(idx); });
      auto result = SIMD_UNIVERSAL_ACCESS(point, .GetValue());
      benchmark::DoNotOptimize(result);
    }
  }
}

void UniversalScalarPure(benchmark::State& state)
{
  auto arraySize = state.range(0);
  std::vector<double> testData(arraySize, .0);
  for (int i = 0; i < testData.size(); ++i)
  {
    testData[i] = arraySize * (arraySize + i) / 2;
  }
  PointIDCollection collection{testData.data()};
  for (auto _ : state)
  {
    for (int i = 0; i < testData.size(); ++i)
    {
      const auto& point = collection.GetPoint(i);
      auto result = point.GetValue();
      benchmark::DoNotOptimize(result);
    }
  }
}

BENCHMARK(UniversalSimd)->Unit(benchmark::kMicrosecond)->Arg(32);
BENCHMARK(UniversalScalar)->Unit(benchmark::kMicrosecond)->Arg(32);
BENCHMARK(UniversalScalarPure)->Unit(benchmark::kMicrosecond)->Arg(32);
