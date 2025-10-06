
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
