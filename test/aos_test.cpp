
#include <gtest/gtest.h>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace {

template<class T>
struct Point
{
  T x, y;
};

template<class T>
Point<T> operator+(const Point<T>& p1, const Point<T>& p2)
{
  return Point<T>{p1.x + p2.x, p1.y + p2.y};
}

template<class SimdModel, class T>
inline auto simdized_value(const Point<T>& p)
{
  using simd_access::simdized_value;
  return Point<decltype(simdized_value<SimdModel>(p.x))>();
}

template<simd_access::specialization_of<Point>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  func(values.x ...);
  func(values.y ...);
}

}

TEST(AosTest, LinearAddition)
{
  static constexpr size_t size = 103;
  Point<double> src1[size], src2[size], dest[size];
  for (int i = 0; i < size; ++i)
  {
    src1[i].x = i;
    src2[i].x = i * 2;
    src1[i].y = i * 3;
    src2[i].y = i * 4;
  }

  using SimdModel = stdx::simd<double>;
  simd_access::loop<SimdModel>(0, size, [&](auto i)
    {
      SIMD_ACCESS(dest, i) = SIMD_ACCESS(src1, i) + SIMD_ACCESS(src2, i);
    });

  for (int i = 0; i < size; ++i)
  {
    EXPECT_EQ(dest[i].x, i * 3);
    EXPECT_EQ(dest[i].y, i * 7);
  }
}

TEST(AosTest, IndirectAddition)
{
  static constexpr size_t size = 10;
  Point<double> src1[size], src2[size], dest[size];
  for (int i = 0; i < size; ++i)
  {
    src1[i].x = i;
    src2[i].x = i * 2;
    src1[i].y = i * 3;
    src2[i].y = i * 4;
  }
  const int indices[10] = { 3, 2, 1, 3, 2, 1, 3, 2, 1, 0 };

  using SimdModel = stdx::simd<double>;
  simd_access::loop_with_linear_index<SimdModel>(indices, indices + size, [&](auto linear_index, auto i)
    {
      SIMD_ACCESS(dest, linear_index) = SIMD_ACCESS(src1, i) + SIMD_ACCESS(src2, i);
    });

  for (int i = 0; i < size; ++i)
  {
    auto expected_result = src1[indices[i]] + src2[indices[i]];
    EXPECT_EQ(dest[i].x, expected_result.x);
    EXPECT_EQ(dest[i].y, expected_result.y);
  }
}
