
#include <experimental/bits/simd.h>
#include <gtest/gtest.h>
#include <utility>

#include "simd_access/base.hpp"
#include "simd_access/reflection.hpp"
#include "simd_access/universal_simd.hpp"

namespace {

template<class T>
struct Point
{
  T x_, y_;

  T GetMul() const { return x_ * y_; }
};

template<class T>
class RestrictiveClass
{
  Point<T> p_;
public:
  RestrictiveClass(const RestrictiveClass&) = delete;
  RestrictiveClass(RestrictiveClass&&) = default;
  explicit RestrictiveClass(Point<T> p) : p_(p) {}

  const auto& GetPoint() const { return p_; }
};

template<class SimdModel, class T>
inline auto simdized_value(const Point<T>& t)
{
  using simd_access::simdized_value;
  return Point{simdized_value<SimdModel>(t.x_), simdized_value<SimdModel>(t.x_)};
}

template<simd_access::specialization_of<Point>... Args>
inline void simd_members(auto&& func, Args&&... values)
{
  using simd_access::simd_members;
  simd_members(func, values.x_ ...);
  simd_members(func, values.y_ ...);
}

}


TEST(UniversalSimdTest, Construct)
{
  using SimdModel = stdx::simd<double>;

  simd_access::universal_simd<Point<double>, SimdModel> v_Point([&](auto i) { return Point<double>{i, i * 2}; });

  simd_access::universal_simd<RestrictiveClass<double>, SimdModel> v_Restrict([&](auto i)
    { return RestrictiveClass<double>(Point<double>{i + 1, (i + 1) * 2}); }
  );

  for (size_t i = 0; i < SimdModel::size(); ++i)
  {
    EXPECT_EQ(v_Point[i].x_, i);
    EXPECT_EQ(v_Point[i].y_, i * 2);
    EXPECT_EQ(v_Restrict[i].GetPoint().x_, i + 1);
    EXPECT_EQ(v_Restrict[i].GetPoint().y_, (i + 1) * 2);
  }
}

TEST(UniversalSimdTest, SimdAccess)
{
  using SimdModel = stdx::simd<double>;

  simd_access::universal_simd<RestrictiveClass<double>, SimdModel> v_Restrict([&](auto i)
    { return RestrictiveClass<double>(Point<double>{i + 3, (i + 3) * 3}); }
  );

  auto result = simd_access::universal_access(v_Restrict, [&](auto&& element) { return element.GetPoint(); });

  for (size_t i = 0; i < SimdModel::size(); ++i)
  {
    EXPECT_EQ(result.x_[i], i + 3);
    EXPECT_EQ(result.y_[i], (i + 3) * 3);
  }
}


TEST(UniversalSimdTest, IndexAccess)
{
  constexpr auto vec_size = 4;
  using SimdModel = stdx::fixed_size_simd<double, vec_size>;

  Point<double> rArray[8];
  for (int i = 0; i < 8; ++i)
  {
    rArray[i].x_ = (i * 2.) + 1.;
    rArray[i].y_ = (i + 1.) * 2.;
  }

  {
    auto result = simd_access::generate_universal(2, [&](auto i) { return rArray[i]; });
    auto mulResult = SIMD_UNIVERSAL_ACCESS(result, .GetMul());
    EXPECT_EQ(result.x_, 5.);
    EXPECT_EQ(result.y_, 6.);
    EXPECT_EQ(mulResult, 30.);
  }

  {
    simd_access::index<SimdModel> index{3};
    auto result = simd_access::generate_universal(index, [&](auto i) { return rArray[i]; });
    auto mulResult = SIMD_UNIVERSAL_ACCESS(result, .GetMul());
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(result[i].x_, rArray[i + 3].x_);
      EXPECT_EQ(result[i].y_, rArray[i + 3].y_);
      EXPECT_EQ(mulResult[i], rArray[i + 3].x_ * rArray[i + 3].y_);
    }
  }

  {
    stdx::rebind_simd_t<size_t, SimdModel> index;
    for (int i = 0; i < vec_size; ++i)
    {
      index[i] = vec_size - i + 3;
    }
    auto result = simd_access::generate_universal(index, [&](auto i) { return rArray[i]; });
    auto mulResult = SIMD_UNIVERSAL_ACCESS(result, .GetMul());
    // test, that SIMD_UNIVERSAL_ACCESS can be embedded:
    auto mulExprResult = SIMD_UNIVERSAL_ACCESS(result, .GetMul()) + SIMD_UNIVERSAL_ACCESS(result, .GetMul());
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(result[i].x_, rArray[vec_size - i + 3].x_);
      EXPECT_EQ(result[i].y_, rArray[vec_size - i + 3].y_);
      EXPECT_EQ(mulResult[i], rArray[vec_size - i + 3].x_ * rArray[vec_size - i + 3].y_);
      EXPECT_EQ(mulExprResult[i], mulResult[i] + mulResult[i]);
    }
  }
}

TEST(UniversalSimdTest, Reference)
{
  RestrictiveClass<double> rArray[4] =
  {
    RestrictiveClass(Point{ 1., 2.}), RestrictiveClass(Point{ 3., 4.}),
    RestrictiveClass(Point{ 5., 6.}), RestrictiveClass(Point{ 7., 8.})
  };

  constexpr size_t vec_size = 2;
  using SimdModel = stdx::fixed_size_simd<double, vec_size>;
  {
    simd_access::index<SimdModel> index{1};
    auto result = simd_access::generate_universal(index, [&](auto i) { return std::cref(rArray[i].GetPoint()); });
    auto xResult = SIMD_UNIVERSAL_ACCESS(result, .x_);

    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(result[i].get().x_, i * 2. + 3.);
      EXPECT_EQ(xResult[i], i * 2. + 3.);
      EXPECT_EQ(result[i].get().y_, i * 2. + 4.);

      EXPECT_EQ(&result[i].get(), &rArray[i + 1].GetPoint());
    }
  }
}