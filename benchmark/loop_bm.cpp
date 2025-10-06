
#include "benchmark/benchmark.h"
#include <vector>
#include <iostream>

#include "helper_bm.hpp"
#include "simd_access/simd_access.hpp"

#if (1)
#define CHECK_RESULT(...)
#else
#ifdef NDEBUG
#error "checking requires assert"
#endif

template<int vec_size>
void CHECK_RESULT(size_t index, const stdx::fixed_size_simd<double, vec_size>& result)
{
  for (int i = 0; i < vec_size; ++i)
  {
    if (result[i] != double(index + i + 1))
    {
      std::cout << "At " << index + i << " result was " << result[i] << std::endl;
      assert(false);
    }
  }
}

#endif

void Loop_IntrinsicScatteredSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<std::pair<double,double>> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return std::pair{double(i + 1), 3.14}; });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
      {
        SimdModel result;
#if (0)
        __m128i vindex {0x200000000ul, 0x600000004ul};
        stdx::__data(result) = _mm256_i32gather_pd(&(dataPtr + i.scalar_index(0))->first, vindex, 8);
        benchmark::DoNotOptimize(result);
        CHECK_RESULT(i.scalar_index(0), result);
#endif
      }, simd_access::VectorResidualLoop);
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_ScatteredSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<std::pair<double,double>> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return std::pair{double(i + 1), 3.14}; });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
      {
        auto result = SIMD_ACCESS(dataPtr, i, .first).to_simd();
        benchmark::DoNotOptimize(result);
        CHECK_RESULT(i.scalar_index(0), result);
      }, simd_access::VectorResidualLoop);
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    simd_access::loop<SimdModel>(0, testData.size(), [&](auto i)
      {
        auto result = SIMD_ACCESS(dataPtr, i).to_simd();
        benchmark::DoNotOptimize(result);
        CHECK_RESULT(i.scalar_index(0), result);
      }, simd_access::VectorResidualLoop);
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearInlinedSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    for (size_t i = 0, e = testData.size(); i < e; i += SimdModel::size())
    {
      auto result = SimdModel(dataPtr + i, stdx::element_aligned);
      benchmark::DoNotOptimize(result);
      CHECK_RESULT(i, result);
    };
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearScalarReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  std::vector<double> testData(arraySize);
  GenerateNWithIndex(testData.begin(), arraySize, [](auto i) { return double(i + 1); });
  HeatCache(testData);
  for (auto _ : state)
  {
    for (size_t i = 0, e = testData.size(); i < e; ++i)
    {
      auto result = testData[i];
      benchmark::DoNotOptimize(result);
      CHECK_RESULT(i, result);
    };
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

#define BM_READ( name ) BENCHMARK( name )->Unit(benchmark::kMicrosecond)->Arg(100)->Arg(4000)

BM_READ(Loop_IntrinsicScatteredSimdReadAccess);
BM_READ(Loop_ScatteredSimdReadAccess);
BM_READ(Loop_LinearSimdReadAccess);
BM_READ(Loop_LinearInlinedSimdReadAccess);
BM_READ(Loop_LinearScalarReadAccess);
