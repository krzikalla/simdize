
#include <experimental/bits/simd.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "simd_access/simd_access.hpp"

namespace {

struct TestStruct
{
  double x;
  double y[1];

  double GetX() const
  {
    return x;
  };
};

struct TestData
{
  static constexpr size_t size = 10;
  double a[size];
  double a_subarr[size][1];
  TestStruct s[size];
  std::vector<double> v;

  TestData() :
    v(size)
  {
    for (int i = 0; i < size; ++i)
    {
      a[i] = v[i] = a_subarr[i][0] = s[i].x = s[i].y[0] = i;
    }
  }
};

}

TEST(Macro, UnvectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  for (int i = 0; i < TestData::size; ++i)
  {
    EXPECT_EQ(SIMD_ACCESS(t.a, i), i);
    EXPECT_EQ(SIMD_ACCESS(t.a_subarr, i, [0]), i);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .x), i);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .y[0]), i);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .GetX()), i);
    EXPECT_EQ(SIMD_ACCESS(t.v, i), i);

    EXPECT_EQ(SIMD_ACCESS_V(t.a, i), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.a_subarr, i, [0]), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.s, i, .x), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.s, i, .y[0]), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.s, i, .GetX()), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.v, i), i);

    EXPECT_EQ(sa::sa(t.a, i), i);
    EXPECT_EQ(sa::sa(t.a_subarr, i)[0], i);
    EXPECT_EQ(sa::sa(t.v, i), i);
  }

  for (int i = 0; i < TestData::size; ++i)
  {
    SIMD_ACCESS(t.a, i) = i + 300.;
    SIMD_ACCESS(t.a_subarr, i, [0]) =  i + 301.;
    SIMD_ACCESS(t.s, i, .x) = i + 302.;
    SIMD_ACCESS(t.s, i, .y[0]) = i + 303.;
    SIMD_ACCESS(t.v, i) = i + 304.;
  }
  for (int i = 0; i < TestData::size; ++i)
  {
    EXPECT_EQ(SIMD_ACCESS(t.a, i), i + 300.);
    EXPECT_EQ(SIMD_ACCESS(t.a_subarr, i, [0]), i + 301.);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .x), i + 302.);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .y[0]), i + 303.);
    EXPECT_EQ(SIMD_ACCESS(t.v, i), i + 304.);
  }

  for (int i = 0; i < TestData::size; ++i)
  {
    sa::sa(t.a, i) =  i + 305.;
    sa::sa(t.a_subarr, i)[0] = i + 306.;
    sa::sa(t.v, i) = i + 307.;
  }

  for (int i = 0; i < TestData::size; ++i)
  {
    EXPECT_EQ(sa::sa(t.a, i), i + 305.);
    EXPECT_EQ(sa::sa(t.a_subarr, i)[0], i + 306.);
    EXPECT_EQ(sa::sa(t.v, i), i + 307.);
  }
}

TEST(Macro, DirectVectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  using SimdModel = stdx::simd<double>;
  simd_access::index<SimdModel> index{3};

  SimdModel x_a = SIMD_ACCESS(t.a, index);
  SimdModel x_a_arr = SIMD_ACCESS(t.a_subarr, index, [0]);
  SimdModel x_s_x = SIMD_ACCESS(t.s, index, .x);
  SimdModel x_s_y = SIMD_ACCESS(t.s, index, .y[0]);
  SimdModel x_s_rx = SIMD_ACCESS(t.s, index, .GetX());
  SimdModel x_v = SIMD_ACCESS(t.v, index);

  SimdModel x_sa_a = sa::sa(t.a, index);
  SimdModel x_sa_arr = sa::sa(t.a_subarr, index)[0];
  SimdModel x_sa_v = sa::sa(t.v, index);

  for (int i = 0; i < x_a.size(); ++i)
  {
    EXPECT_EQ(x_a[i], i + 3);
    EXPECT_EQ(x_a_arr[i], i + 3);
    EXPECT_EQ(x_s_x[i], i + 3);
    EXPECT_EQ(x_s_y[i], i + 3);
    EXPECT_EQ(x_s_rx[i], i + 3);
    EXPECT_EQ(x_v[i], i + 3);
    EXPECT_EQ(x_sa_a[i], i + 3);
    EXPECT_EQ(x_sa_arr[i], i + 3);
    EXPECT_EQ(x_sa_v[i], i + 3);
  }
}

TEST(Macro, SimdVectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  using SimdModel = stdx::simd<double>;
  stdx::rebind_simd_t<size_t, SimdModel> index;
  const auto vec_size = index.size();
  for (size_t i = 0; i < vec_size; ++i)
  {
    index[i] = vec_size - i + 3;
  }

  SimdModel x_a = SIMD_ACCESS(t.a, index);
  SimdModel x_a_arr = SIMD_ACCESS(t.a_subarr, index, [0]);
  SimdModel x_s_x = SIMD_ACCESS(t.s, index, .x);
  SimdModel x_s_y = SIMD_ACCESS(t.s, index, .y[0]);
  SimdModel x_s_rx = SIMD_ACCESS(t.s, index, .GetX());
  SimdModel x_v = SIMD_ACCESS(t.v, index);

  SimdModel x_sa_a = sa::sa(t.a, index);
  SimdModel x_sa_arr = sa::sa(t.a_subarr, index)[0];
  SimdModel x_sa_v = sa::sa(t.v, index);

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(x_a[i], vec_size - i + 3);
    EXPECT_EQ(x_a_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_s_x[i], vec_size - i + 3);
    EXPECT_EQ(x_s_y[i], vec_size - i + 3);
    EXPECT_EQ(x_s_rx[i], vec_size - i + 3);
    EXPECT_EQ(x_v[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_a[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_v[i], vec_size - i + 3);
  }
}


TEST(Macro, DeducedSimdVectorizedArrayAccess)
{
  TestData t;
  using SimdModel = stdx::simd<double>;
  simd_access::index<SimdModel> index{5};
  const auto vec_size = index.size();

  auto deducedFunction = [&vec_size](auto&& value)
  {
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(value[i], i + 5);
    }
  };

  deducedFunction(SIMD_ACCESS_V(t.a, index));
  deducedFunction(SIMD_ACCESS_V(t.a_subarr, index, [0]));
  deducedFunction(SIMD_ACCESS_V(t.s, index, .x));
  deducedFunction(SIMD_ACCESS_V(t.s, index, .y[0]));
  deducedFunction(SIMD_ACCESS_V(t.s, index, .GetX()));
  deducedFunction(SIMD_ACCESS_V(t.v, index));
}

TEST(Macro, RValueTest)
{
  struct Test
  {
    double data_[100];
    double operator[](int i) const { return data_[i]; }

  } test;

  std::iota(test.data_, test.data_ + 100, 0);
  for (int i = 0; i < 100; ++i)
  {
    EXPECT_EQ(SIMD_ACCESS(test, i), i);
  }

  using SimdModel = stdx::simd<double>;
  {
    simd_access::index<SimdModel> index{3};
    SimdModel x = SIMD_ACCESS(test, index);
    for (int i = 0; i < x.size(); ++i)
    {
      EXPECT_EQ(x[i], i + 3);
    }
  }

  {
    stdx::rebind_simd_t<size_t, SimdModel> index;
    for (int i = 0; i < index.size(); ++i)
    {
      index[i] = index.size() - i + 3;
    }
    SimdModel x = SIMD_ACCESS(test, index);
    for (int i = 0; i < x.size(); ++i)
    {
      EXPECT_EQ(x[i], x.size() - i + 3);
    }
  }
}

/*
TEST(Macro, NonContinuousStorage)
{
  constexpr size_t vec_size = stdx::native_simd<int>::size();
  std::vector<int> src_values[vec_size];
  const int test_size = 5;
  for (int i = 0; i < vec_size; ++i)
  {
    src_values[i].resize(test_size);
    int startValue = 1 * i;
    for (auto& j : src_values[i])
    {
      j = startValue;
      startValue += (i * 100) + 1;
    }
  }
  for (int j = 0; j < test_size; ++j)
  {
    auto result = SIMD_ACCESS_V(src_values, simd_access::index<vec_size>{0}, [j]);
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(result[i], i + ((i * 100) + 1) * j);
    }
  }
}*/
