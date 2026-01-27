
#include <gtest/gtest.h>

#include "simd_access/index.hpp"


TEST(Index, Linear)
{
  using SimdModel = stdx::simd<double>;
  simd_access::index<SimdModel> index{3};
  auto value = index.to_simd();

  for (int i = 0; i < index.size(); ++i)
  {
    EXPECT_EQ(index.scalar_index(i), i + 3);
    EXPECT_EQ(value[i], i + 3);
  }
}

TEST(Index, LinearTypeConversion)
{
  using SimdModel = stdx::simd<double>;
  simd_access::index<SimdModel, int> index{3};
  auto value = index.to_simd();

  for (int i = 0; i < index.size(); ++i)
  {
    EXPECT_EQ(index.scalar_index(i), i + 3);
    EXPECT_EQ(value[i], i + 3);
  }
}

TEST(Index, Operators)
{
  using SimdModel = stdx::simd<double>;
  simd_access::index<SimdModel, int> index{3};

  auto add1 = index + 5;
  auto add2 = 7 + index;

  auto mul1 = index * 5;
  auto mul2 = 7 * index;

  for (int i = 0; i < index.size(); ++i)
  {
    EXPECT_EQ(simd_access::scalar_index(add1, i), i + 3 + 5);
    EXPECT_EQ(simd_access::scalar_index(add2, i), i + 3 + 7);
    EXPECT_EQ(simd_access::scalar_index(mul1, i), (i + 3) * 5);
    EXPECT_EQ(simd_access::scalar_index(mul2, i), (i + 3) * 7);
  }
}
