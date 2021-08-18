/******************************************************************************
Copyright (c) 2021 Dmitriy Korchemkin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/
#ifndef POLYNOMIALS_TESTING_HPP
#define POLYNOMIALS_TESTING_HPP

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include <doctest/doctest.h>

#include "polynomials/dense_polynomial.hpp"

#include <ceres/jet.h>
#include <iostream>
#include <random>

template <typename T> struct scalar_type { using type = T; };

template <typename T, int n> struct scalar_type<ceres::Jet<T, n>> {
  using type = T;
};

template <typename T> using scalar_type_t = typename scalar_type<T>::type;

// doctest::Approx is not templated... what a shame
template <typename T> struct inner_type { using type = T; };

template <typename T> struct inner_type<std::complex<T>> { using type = T; };

template <typename T> using inner_type_t = typename inner_type<T>::type;

template <typename T1, typename T2>
bool approximately_equal(const T1 &a, const T2 &b) {
  using std::abs;
  using std::isfinite;
  using std::max;
  using std::sqrt;
  using T = inner_type_t<T1>;
  const T eps = sqrt(std::numeric_limits<T>::epsilon());
  return abs(a - b) < eps * (T(1.) + max(abs(a), abs(b)));
}

template <typename T, int N>
bool approximately_equal(const ceres::Jet<T, N> &a, const ceres::Jet<T, N> &b) {
  bool equal = approximately_equal(a.a, b.a);
  for (int i = 0; i < N && equal; ++i)
    equal &= approximately_equal(a.v[i], b.v[i]);
  return equal;
}

template <typename A, typename B>
bool approximately_equal_matrices(const Eigen::MatrixBase<A> &a,
                                  const Eigen::MatrixBase<B> &b) {
  using std::abs;
  using std::isfinite;
  using std::max;
  using T = inner_type_t<typename A::Scalar>;
  const T eps = sqrt(std::numeric_limits<T>::epsilon());
  const T diff_norm = (a - b).template lpNorm<Eigen::Infinity>();
  const T a_norm = a.template lpNorm<Eigen::Infinity>();
  const T b_norm = b.template lpNorm<Eigen::Infinity>();
  return diff_norm < eps * (T(1.) + max(a_norm, b_norm));
}

struct RngBase {
  static std::mt19937 &rng() {
    static std::mt19937 r(1337);
    return r;
  }
};

template <typename T> struct Random : RngBase {
  T operator()() const {
    std::uniform_real_distribution<T> runif(T(-2.), T(2.));
    return runif(rng());
  }
};
template <typename S, int N> struct Random<ceres::Jet<S, N>> : RngBase {
  using T = ceres::Jet<S, N>;

  T operator()() const {
    std::uniform_real_distribution<S> runif(S(-2.), S(2.));
    T res;
    res.a = runif(rng());
    for (int i = 0; i < N; ++i)
      res.v[i] = runif(rng());
    return res;
  }
};

template <typename RealType> struct PolynomialHolder {
  template <typename... T> PolynomialHolder(const T &...args) : data(args...) {}
  RealType data;
  RealType &mutable_data() { return data; }
  RealType &operator*() { return data; }
  const RealType &operator*() const { return data; }
};

template <typename Mapped> struct PolynomialHolder<Eigen::Map<Mapped>> {
  using RealType = std::remove_const_t<Mapped>;
  using MapType = Eigen::Map<Mapped>;
  template <typename... T>
  PolynomialHolder(const T &...args)
      : real_data(args...), data(real_data.data(), args...) {}

  RealType &mutable_data() { return real_data; }

  MapType &operator*() { return data; }
  const MapType &operator*() const { return data; }

  RealType real_data;
  MapType data;
};

#define CHECK_VALID(exp)                                                       \
  {                                                                            \
    CHECK((exp));                                                              \
    valid &= (exp);                                                            \
  }                                                                            \
  while (false)                                                                \
    ;

struct Dummy {};

// Generic substitutor
template <template <typename...> typename, typename...> struct Substitutor;
template <template <typename...> typename Result, typename... Types,
          typename... Rem>
struct Substitutor<Result, std::tuple<Types...>, Rem...> {

  template <typename T> struct Substitute {
    template <typename... TT> using Next = Result<T, TT...>;

    static bool Check() { return Substitutor<Next, Rem...>::Check(); }
  };
  static bool Check() { return (Substitute<Types>::Check() && ...); }
};

template <template <typename...> typename Result, typename V, typename... Rem>
struct Substitutor<Result, V, Rem...> {
  template <typename... TT> using Next = Result<V, TT...>;
  static bool Check() { return Substitutor<Next, Rem...>::Check(); }
};

template <template <typename...> typename Result, typename... Rem>
struct Substitutor<Result, Dummy, Rem...> {
  static bool Check() { return Result<Rem...>::Check(); }
};

template <template <typename...> typename Result>
struct Substitutor<Result, Dummy> {
  static bool Check() { return Result<>::Check(); }
};

template <template <typename...> typename Result> struct Substitutor<Result> {
  static bool Check() { return Result<>::Check(); }
};

// Traverses degrees
template <template <typename...> typename Test, typename... Args>
struct DegreeIterator;

template <template <typename...> typename Test, typename deg_a, typename Scalar,
          typename... Args>
struct DegreeIterator<Test, deg_a, Scalar, Args...> {
  template <typename... TT> using TestDeg = Test<deg_a, Scalar, TT...>;

  static bool Check() { return Substitutor<TestDeg, Args...>::Check(); }
};

// Traverses vector types
template <template <typename...> typename Test, typename...>
struct TypeIterator;

template <template <typename...> typename Test, typename deg_t, typename Scalar,
          typename... Args>
struct TypeIterator<Test, deg_t, Scalar, Args...> {
  static constexpr Eigen::Index deg = deg_t::value;

  using PolyTypes =
      std::tuple<polynomials::DensePoly<Scalar, deg, deg>,
                 polynomials::DensePoly<Scalar, polynomials::Dynamic, deg>,
                 polynomials::DensePoly<Scalar, polynomials::Dynamic,
                                        polynomials::Dynamic>>;

  template <typename... TT> using TestDynamic = Test<deg_t, Scalar, TT...>;

  static bool Check() {
    return Substitutor<TestDynamic, PolyTypes, Dummy, Args...>::Check();
  }
};

template <template <typename...> typename Test, typename...> struct MapIterator;

template <template <typename...> typename Test, typename Poly, typename deg_t,
          typename Scalar, typename... Args>
struct MapIterator<Test, deg_t, Scalar, Poly, Args...> {
  static constexpr Eigen::Index deg = deg_t::value;
  using PolyType = Poly;
  using MapTypes =
      std::tuple<PolyType, Eigen::Map<PolyType>, Eigen::Map<const PolyType>>;

  template <typename... TT> using TestDynamic = Test<TT..., deg_t, Scalar>;

  static bool Check() {
    return Substitutor<TestDynamic, Args..., MapTypes>::Check();
  }
};

template <template <typename...> typename Test, typename... Args>
struct PolyIterator {
  template <typename... TT> using M = MapIterator<Test, TT...>;
  template <typename... TT> using MV = TypeIterator<M, TT...>;
  template <typename... TT> using MVD = DegreeIterator<MV, TT...>;

  static bool Check() { return Substitutor<MVD, Args...>::Check(); }
};

using TestDegrees =
    std::tuple<std::integral_constant<int, 5>, std::integral_constant<int, 3>,
               std::integral_constant<int, 0>>;
template <typename T> using JetType = ceres::Jet<T, 4>;

template <template <typename...> typename Test, typename ScalarP1,
          typename ScalarP2, typename ScalarType>
struct PolyPolyScalarTest {

  template <typename... Args> using P = PolyIterator<Test, Args...>;
  template <typename... Args> using PP = PolyIterator<P, Args...>;

  static bool Check() {
    return PP<TestDegrees, std::tuple<ScalarP1>, Dummy, TestDegrees,
              std::tuple<ScalarP2>, Dummy,
              std::tuple<ScalarType, JetType<ScalarType>>>::Check();
  }
};

template <template <typename...> typename Test, typename ScalarP1,
          typename ScalarType>
struct PolyScalarScalarTest {

  template <typename... Args> using P = PolyIterator<Test, Args...>;

  static bool Check() {
    return P<TestDegrees, std::tuple<ScalarP1, JetType<ScalarP1>>, Dummy,
             std::tuple<ScalarType, JetType<ScalarType>>,
             std::tuple<ScalarType, JetType<ScalarType>>>::Check();
  }
};

#endif
