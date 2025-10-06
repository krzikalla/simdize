
#include "benchmark/benchmark.h"
#include <vector>
#include <iostream>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace sa = simd_access;

template<class T>
struct Point {
  T x;
  int padding; // prevent vertical vectorization
  T y;
  Point(T xx, T yy) : x(xx), y(yy) {}
  Point() {}
  auto operator+(const Point& op2) const { return Point{ x + op2.x, y + op2.y }; }
};

template<class SimdModel, class T>
inline auto simdized_value(const Point<T>& t)
{
  using sa::simdized_value;
  return Point{simdized_value<SimdModel>(t.x), simdized_value<SimdModel>(t.y)};
}

template<simd_access::specialization_of<Point>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  using sa::simd_members;
  simd_members(func, values.x ...);
  simd_members(func, values.y ...);
}


void ReflectionSimd(benchmark::State& state)
{
  auto arraySize = state.range(0);
  using SimdModel = stdx::simd<double>;
  std::vector<Point<double>> x(arraySize, Point{.0, .0}), y(arraySize, Point{.0, .0}), z(arraySize);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());
    simd_access::loop<SimdModel>(0, z.size(), [&](auto i)
    {
      SIMD_ACCESS(z, i) = SIMD_ACCESS(x, i) + SIMD_ACCESS(y, i);
    });
    benchmark::DoNotOptimize(z.data());
  }
}

void ReflectionScalar(benchmark::State& state)
{
  auto arraySize = state.range(0);
  std::vector<Point<double>> x(arraySize), y(arraySize), z(arraySize);
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(y.data());
    for (int i = 0; i < z.size(); ++i)
    {
      z[i] = x[i] + y[i];
    }
    benchmark::DoNotOptimize(z.data());
  }
}

BENCHMARK(ReflectionSimd)->Arg(16);
BENCHMARK(ReflectionScalar)->Arg(16);
