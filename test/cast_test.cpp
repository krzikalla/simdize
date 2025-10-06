
#include <gtest/gtest.h>

#include "simd_access/simd_access.hpp"

namespace {

template<class T>
struct TestStruct
{
  T x;
};

template<class T>
TestStruct<T> operator*(const T& v1, const TestStruct<T>& v2)
{
  return TestStruct{v1 * v2.x};
}

template<class SimdModel, class T>
inline auto simdized_value(const TestStruct<T>& t)
{
  using simd_access::simdized_value;
  return TestStruct<decltype(simdized_value<SimdModel>(t.x))>();
}

template<simd_access::specialization_of<TestStruct>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  using simd_access::simd_members;
  simd_members(func, values.x ...);
}

}

TEST(Cast, DependentContext)
{
  const int size = 103;
  TestStruct<double> src[size], dest[size];
  using SimdModel = stdx::simd<double>;
  for (int i = 0; i < size; ++i)
  {
    src[i].x = i;
    // this compiles fine:
    dest[i] = 3.0 * src[i];
  }

  simd_access::loop<SimdModel>(0, size, [&](auto i)
    {
      // this doesn't compile, an explicit cast is needed
      //SIMD_ACCESS(dest, i) = 2.0 * SIMD_ACCESS(src, i);
      SIMD_ACCESS(dest, i) = simd_access::simd_broadcast<decltype(i)>(2.0) * SIMD_ACCESS(src, i);
    });

  for (int i = 0; i < size; ++i)
  {
    EXPECT_EQ(dest[i].x, i * 2.0);
  }
}
