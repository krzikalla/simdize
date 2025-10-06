
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access/simd_access.hpp"

namespace {

template<class T>
struct TestStruct
{
  T x;
  T y[2];

  std::pair<T, T> GetPair() const
  {
    return std::pair(x, y[1]);
  }

  TestStruct operator+(const TestStruct& op2) const
  {
    return TestStruct{ x + op2.x, { y[0] + op2.y[0], y[1] + op2.y[1] } };
  }
};

struct TestData
{
  static constexpr size_t size = 103;
  std::vector<TestStruct<double>> v;

  explicit TestData() :
    v(size)
  {
    for (int i = 0; i < size; ++i)
    {
      v[i].x = i;
      v[i].y[0] = i + 1000;
      v[i].y[1] = i + 2000;
    }
  }

  // deliberately return an rvalue:
  TestStruct<double> operator[](int i)
  {
    return v[i];
  }
};

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
  simd_members(func, values.y[0] ...);
  simd_members(func, values.y[1] ...);
}

}


TEST(Reflections, IndexedAccess)
{
  TestData src;
  using SimdModel = stdx::simd<double>;
  simd_access::loop<SimdModel>(0, 100, [&](auto i)
    {
      auto index = i.to_simd();
      auto ts = SIMD_ACCESS_V(src.v, index);
      for (int j = 0; j < index.size(); ++j)
      {
        EXPECT_EQ(ts.x[j], index[j]);
        EXPECT_EQ(ts.y[0][j], index[j] + 1000);
        EXPECT_EQ(ts.y[1][j], index[j] + 2000);
      }
    }, simd_access::VectorResidualLoop);
}

TEST(Reflections, RValueAccess)
{
  TestData src;
  using SimdModel = stdx::simd<double>;
  {
    simd_access::index<SimdModel> index{3};
    auto ts = SIMD_ACCESS(src, index);
    for (int j = 0; j < index.size(); ++j)
    {
      EXPECT_EQ(ts.x[j], j + 3);
      EXPECT_EQ(ts.y[0][j], j + 1003);
      EXPECT_EQ(ts.y[1][j], j + 2003);
    }
  }

  {
    simd_access::index<SimdModel> index{3};
    auto ts = SIMD_ACCESS(src.v, index, .GetPair());
    for (int j = 0; j < index.size(); ++j)
    {
      EXPECT_EQ(ts.first[j], j + 3);
      EXPECT_EQ(ts.second[j], j + 2003);
    }
  }

  {
    stdx::rebind_simd_t<size_t, SimdModel> index;
    for (int i = 0; i < index.size(); ++i)
    {
      index[i] = index.size() - i + 3;
    }
    auto ts = SIMD_ACCESS(src.v, index, .GetPair());
    for (int i = 0; i < index.size(); ++i)
    {
      EXPECT_EQ(ts.first[i], index.size() - i + 3);
      EXPECT_EQ(ts.second[i], index.size() - i + 2003);
    }
  }
}

TEST(Reflections, OperatorOverload)
{
  TestData dest, src1, src2;
  using SimdModel = stdx::simd<double>;  simd_access::loop<SimdModel>(0, src1.v.size(), [&](auto i)
  {
    SIMD_ACCESS(dest.v, i) = SIMD_ACCESS(src1.v, i) + SIMD_ACCESS(src2.v, i);
  });
  for (int i = 0; i < dest.v.size(); ++i)
  {
    EXPECT_EQ(dest.v[i].x, i * 2);
    EXPECT_EQ(dest.v[i].y[0], (i + 1000) * 2);
    EXPECT_EQ(dest.v[i].y[1], (i + 2000) * 2);
  }
}

TEST(Reflections, ConditionalAssignment)
{
  TestData dest, src;
  using SimdModel = stdx::simd<double>;
  using simd_access::where;
  stdx::simd_mask<double, SimdModel::abi_type> mask;
  const auto vec_size = mask.size();
  for (int i = 0; i < vec_size; ++i)
  {
    mask[i] = i % 2 == 0;
  }

  auto loop_size = (src.v.size() / vec_size) * vec_size;
  simd_access::loop<SimdModel>(0, loop_size, [&](auto i)
  {
    auto result = SIMD_ACCESS_V(src.v, i);
    where(mask, result) = SIMD_ACCESS(src.v, i) + SIMD_ACCESS(src.v, i);
    SIMD_ACCESS(dest.v, i) = result;
  }, simd_access::VectorResidualLoop);

  for (int i = 0; i < loop_size; ++i)
  {
    auto factor = (i % 2 == 0) ? 2 : 1;
    EXPECT_EQ(dest.v[i].x, i * factor);
    EXPECT_EQ(dest.v[i].y[0], (i + 1000) * factor);
    EXPECT_EQ(dest.v[i].y[1], (i + 2000) * factor);
  }
}



TEST(Reflections, StructuralReduction)
{
  TestData src;
  TestStruct<double> dest[5] = {}, faultdest[5] = {};
  using SimdModel = stdx::simd<double>;
  const int indices[11] = { 1, 1, 2, 3, 4, 0, 0, 4, 1, 2, 4 };
  const int num_indices[5] = { 2, 3, 2, 1, 3 };

  simd_access::loop<SimdModel>(indices, indices + 11, [&](auto elemIdx)
  {
    auto result = SIMD_ACCESS_V(src.v, elemIdx);
    simd_access::elementwise_with_index([&](auto elemIndexScalar, auto... i)
    {
      simd_members([&](auto& d, const auto& s) { d += simd_access::element(s, i...); }, dest[elemIndexScalar], result);
    }, elemIdx);
    SIMD_ACCESS(faultdest, elemIdx, .x) += result.x;
  });

  int faultdestCounter = 0;
  for (int i = 0; i < 5; ++i)
  {
    EXPECT_EQ(dest[i].x, num_indices[i] * i);
    EXPECT_EQ(dest[i].y[0], num_indices[i] * (i + 1000));
    EXPECT_EQ(dest[i].y[1], num_indices[i] * (i + 2000));
    if (faultdest[i].x != num_indices[i] * i)
    {
      ++faultdestCounter;
    }
  }
  const auto vec_size = SimdModel::size();
  if (vec_size == 2)
  {
    EXPECT_EQ(faultdestCounter, 1);
  }
  else if (vec_size == 4 || vec_size == 8)
  {
    EXPECT_EQ(faultdestCounter, 2);
  }
}
