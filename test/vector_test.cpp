
#include <gtest/gtest.h>

#include "simd_access/simd_access.hpp"
#include "simd_access/vector.hpp"


TEST(VectorTest, ArrayAdd)
{
  static constexpr size_t size = 103;
  simd_access::vector<double> src1(size), src2(size), dest(size);
  const auto& csrc1 = src1;
  for (int i = 0; i < size; ++i)
  {
    src1[i] = i;
    src2[i] = i * 2;
  }

  using SimdModel = stdx::simd<double>;  simd_access::loop<SimdModel>(0, size, [&](auto i)
    {
      dest[i] = csrc1[i] + src2[i];
    });

  for (int i = 0; i < size; ++i)
  {
    EXPECT_EQ(dest[i], i * 3);
  }
}
