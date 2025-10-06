
#include <gtest/gtest.h>

#include "simd_access/element_access.hpp"
#include "simd_access/universal_simd.hpp"


TEST(Elementwise, SimpleRead)
{
  double source[5] = { 1, 2, 3, 4, 5 };
  double dest[5] = { };
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> x(source, stdx::element_aligned);

  int linear_index = 0;

  auto read_fn = [&](auto&& y)
    {
      dest[linear_index++] = y;
    };

  simd_access::elementwise(read_fn, x);
  simd_access::elementwise(read_fn, source[4]);

  for (int i = 0; i < 5; ++i)
  {
    EXPECT_EQ(dest[i], i + 1);
  }
}

TEST(Elementwise, MultipleRead)
{
  double x_source[5] = { 1, 3, 5, 7, 9 };
  double y_source[5] = { 2, 4, 6, 8, 10 };
  double dest[10] = { };
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> x(x_source, stdx::element_aligned);
  stdx::fixed_size_simd<double, vec_size> y(y_source, stdx::element_aligned);

  int linear_index = 0;

  auto read_fn = [&](auto&& x_scalar, auto&& y_scalar)
    {
      dest[linear_index++] = x_scalar;
      dest[linear_index++] = y_scalar;
    };

  simd_access::elementwise(read_fn, x, y);
  simd_access::elementwise(read_fn, x_source[4], y_source[4]);

  for (int i = 0; i < 10; ++i)
  {
    EXPECT_EQ(dest[i], i + 1);
  }
}

TEST(Elementwise, SimpleWrite)
{
  double source[5] = { 1, 2, 3, 4, 5 };
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> dest_v (.0);
  double dest_s = .0;

  int linear_index = 0;
  auto write_fn = [&](auto&& y)
    {
      simd_access::element_write(y) = source[linear_index++];
    };

  simd_access::elementwise(write_fn, dest_v);
  simd_access::elementwise(write_fn, dest_s);

  for (int i = 0; i < 4; ++i)
  {
    EXPECT_EQ(dest_v[i], i + 1);
  }
  EXPECT_EQ(dest_s, 5);
}

TEST(Elementwise, SingleElementAccess)
{
  double source[5] = { 1, 2, 3, 4, 5 };
  double scalar_source = 6;
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> x(source, stdx::element_aligned);

  EXPECT_EQ(simd_access::get_element<0>(x), 1);
  EXPECT_EQ(simd_access::get_element<1>(x), 2);
  EXPECT_EQ(simd_access::get_element<2>(x), 3);
  EXPECT_EQ(simd_access::get_element<3>(x), 4);
  EXPECT_EQ(simd_access::get_element<0>(scalar_source), 6);
}

TEST(Elementwise, SingleElementAccessUniversalSimd)
{
  using SimdModel = stdx::fixed_size_simd<double, 4>;
  simd_access::auto_simd_t<std::string, SimdModel> custom_simd;
  custom_simd[0] = "Hi";

  EXPECT_EQ(simd_access::get_element<0>(custom_simd), "Hi");
  EXPECT_EQ(simd_access::get_element<1>(custom_simd), "");
  EXPECT_EQ(simd_access::get_element<2>(custom_simd), "");

  EXPECT_EQ(&simd_access::get_element<0>(custom_simd), &custom_simd[0]);
  EXPECT_EQ(&simd_access::get_element<1>(custom_simd), &custom_simd[1]);
  EXPECT_EQ(&simd_access::get_element<2>(custom_simd), &custom_simd[2]);
}
