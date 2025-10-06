
#include "benchmark/benchmark.h"
#include <memory>
#include <vector>
#include <iostream>

#include "helper_bm.hpp"
#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"


void AligningLoop_Unaligned(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  const auto vec_size = SimdModel::size();
  auto testData =
    std::unique_ptr<double>(reinterpret_cast<double*>(std::aligned_alloc(vec_size * sizeof(double),
      arraySize * sizeof(double))));
  auto dataPtr = testData.get();
  GenerateNWithIndex(dataPtr, arraySize, [](auto i) { return double(i + 1); });
  HeatCache(dataPtr, dataPtr + arraySize);
  for (auto _ : state)
  {
    simd_access::loop<SimdModel>(1, arraySize, [&](auto i)
      {
        auto result = SIMD_ACCESS_V(dataPtr, i);
        benchmark::DoNotOptimize(result);
      });
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}


void AligningLoop_Aligned(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  const auto vec_size = SimdModel::size();
  auto testData =
    std::unique_ptr<double>(reinterpret_cast<double*>(std::aligned_alloc(vec_size * sizeof(double),
      arraySize * sizeof(double))));
  auto dataPtr = testData.get();
  GenerateNWithIndex(dataPtr, arraySize, [](auto i) { return double(i + 1); });
  HeatCache(dataPtr, dataPtr + arraySize);
  for (auto _ : state)
  {
    simd_access::aligning_loop<SimdModel>(1, arraySize, [&](auto i) { return i % vec_size == 0; },
      [&](auto i)
      {
        auto result = SIMD_ACCESS_V(dataPtr, i);
        benchmark::DoNotOptimize(result);
      });
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

#define BM_READ( name ) BENCHMARK( name )->Unit(benchmark::kMicrosecond)->Arg(100)->Arg(4000)

BM_READ(AligningLoop_Unaligned);
BM_READ(AligningLoop_Aligned);
