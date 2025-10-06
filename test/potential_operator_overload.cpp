
#include <gtest/gtest.h>

#include "simd_access/simd_loop.hpp"
#include "simd_access/value_access.hpp"
#include "simd_access/simd_access.hpp"

// This file is a compileable demo of a globally overloadable 'operator[]' and an overloadable member access
// operator ('operator.'). The SIMD_ACCESS macro is not required.
namespace {

struct Vector
{
  double x, y;

  // Implements the default behavior of 'operator.'.
  template<auto Vector::*Member>
  auto& dot()
  {
    return this->*Member;
  }

  template<auto Vector::*Member>
  const auto& dot() const
  {
    return this->*Member;
  }
};

template <class T, class Tuple>
struct Index;

template <class T, class... Types>
struct Index<T, std::tuple<T, Types...>> {
    static const std::size_t value = 0;
};

template <class T, class U, class... Types>
struct Index<T, std::tuple<U, Types...>> {
    static const std::size_t value = 1 + Index<T, std::tuple<Types...>>::value;
};

template<typename T, typename... Names>
struct Indexable
{
  T data_[sizeof...(Names)];

  template<typename Name>
  T& operator[](Name)
  {
    return data_[Index<Name, std::tuple<Names...>>::value];
  }

  template<typename Name>
  const T& operator[](Name) const
  {
    return data_[Index<Name, std::tuple<Names...>>::value];
  }
};

struct tX {} nX;
struct tY {} nY;

// Alternative implementation of an vector with named member variables. This doesn't need an overloaded operator.().
using NamedVector = Indexable<double, tX, tY>;

} // namespace


TEST(OperatorOverload, Subscription)
{
  const size_t array_size = 101;
  double source[array_size], destination[array_size];
  for (int i = 0; i < array_size; ++i)
  {
    source[i] = i + 1.0;
  }

  simd_access::loop<stdx::simd<double>>(0, array_size,
    [&](auto i)
    {
      // If 'operator[]' would be globally overloadable, then argument order don't need to be reversed anymore.
      // The following line could simply read:
      // destination[i] = 1.0 / source[i];

      i[destination] = 1.0 / i[source];
    });

  for (size_t i = 0; i < array_size; ++i)
  {
    EXPECT_EQ(destination[i], 1.0 / (i + 1.0));
  }
}

TEST(OperatorOverload, MemberAccess)
{
  const size_t array_size = 103;
  Vector source[array_size], destination[array_size];
  for (int i = 0; i < array_size; ++i)
  {
    source[i].x = i + 1.0;
    source[i].y = i + 2.0;
  }

  simd_access::loop<stdx::simd<double>>(0, array_size,
    [&](auto i)
    {
      // If 'operator.' would be overloadable (in addition to a global 'operator[]'),
      // the following two lines could simply read:
      // destination[i].x = 1.0 / source[i].x;
      // destination[i].y = 1.0 / source[i].y;

      i[destination].template dot<&Vector::x>() = 1.0 / i[source].template dot<&Vector::x>();
      i[destination].template dot<&Vector::y>() = 1.0 / i[source].template dot<&Vector::y>();
    });


  for (size_t i = 0; i < array_size; ++i)
  {
    EXPECT_EQ(destination[i].x, 1.0 / (i + 1.0));
    EXPECT_EQ(destination[i].y, 1.0 / (i + 2.0));
  }
}

TEST(OperatorOverload, NamedMemberAccess)
{
  const size_t array_size = 105;
  NamedVector source[array_size], destination[array_size];
  for (int i = 0; i < array_size; ++i)
  {
    source[i][nX] = i + 1.0;
    source[i][nY] = i + 2.0;
  }

  simd_access::loop<stdx::simd<double>>(0, array_size,
    [&](auto i)
    {
      i[destination][nX] = 1.0 / i[source][nX];
      i[destination][nY] = 1.0 / i[source][nY];
    });

  for (size_t i = 0; i < array_size; ++i)
  {
    EXPECT_EQ(destination[i][nX], 1.0 / (i + 1.0));
    EXPECT_EQ(destination[i][nY], 1.0 / (i + 2.0));
  }
}
