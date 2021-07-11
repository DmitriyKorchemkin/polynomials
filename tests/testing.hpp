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
template <typename T> bool approximately_equal(const T &a, const T &b) {
  using std::abs;
  using std::max;
  using std::sqrt;
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
template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct TestBase {
  using ScalarP1 = typename Poly1::Scalar;
  using ScalarP2 = typename Poly2::Scalar;
  using RS1 = Random<Scalar1>;
  using RSP1 = Random<ScalarP1>;
  using RSP2 = Random<ScalarP2>;

  RS1 rs1;
  RS1 rs2;
  RSP1 rsp1;
  RSP2 rsp2;

  TestBase() : poly1(deg1, low_deg1), poly2(deg2, low_deg2) {
    for (int i = 0; i < poly1.mutable_data().total_coeffs(); ++i)
      poly1.mutable_data().coeffs()[i] = rsp1();
    for (int i = 0; i < poly2.mutable_data().total_coeffs(); ++i)
      poly2.mutable_data().coeffs()[i] = rsp2();
  }

  PolynomialHolder<Poly1> poly1;
  PolynomialHolder<Poly2> poly2;
};

template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct PlusTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto sum_fg = f + g;
    const auto exp_degree = std::max(f.degree(), g.degree());
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fg = sum_fg(s);
      const auto exp = f(s) + g(s);
      CHECK_VALID(approximately_equal(got_fg, exp));
    }
    return valid;
  }
};

template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct MinusTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto sub_fg = f - g;
    const auto exp_degree = std::max(f.degree(), g.degree());
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fg = sub_fg(s);
      const auto exp_fg = f(s) - g(s);
      CHECK_VALID(approximately_equal(got_fg, exp_fg));
    }
    return valid;
  }
};
template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct MulTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto mul_fg = f * g;
    const auto exp_degree = f.degree() + g.degree();
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fg = mul_fg(s);
      const auto exp = f(s) * g(s);
      CHECK_VALID(approximately_equal(got_fg, exp));
    }
    return valid;
  }
};
template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct PlusScalarTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();

    auto v = Base::rs2();
    auto sum_fs = f + v;
    auto sum_sf = v + f;
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fs = sum_fs(s);
      const auto got_sf = sum_sf(s);
      const auto exp_f = f(s) + v;
      CHECK_VALID(approximately_equal(got_fs, exp_f));
      CHECK_VALID(approximately_equal(got_sf, exp_f));
    }
    return valid;
  }
};
template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct MinusScalarTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();
    const auto v = Base::rs2();
    const auto sub_fs = f - v;
    const auto sub_sf = v - f;
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fs = sub_fs(s);
      const auto got_sf = sub_sf(s);
      const auto exp_fs = f(s) - v;
      const auto exp_sf = v - f(s);
      CHECK_VALID(approximately_equal(got_fs, exp_fs));
      CHECK_VALID(approximately_equal(got_sf, exp_sf));
    }
    return valid;
  }
};

template <typename Poly1, int deg1, int low_deg1, typename Poly2, int deg2,
          int low_deg2, typename Scalar1, typename Scalar2>
struct MulScalarTest
    : TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2> {
  using Base =
      TestBase<Poly1, deg1, low_deg1, Poly2, deg2, low_deg2, Scalar1, Scalar2>;
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();
    const auto v = Base::rs2();
    const auto mul_fs = f * v;
    const auto mul_sf = v * f;
    for (int i = 0; i < 10 * exp_degree; ++i) {
      const auto s = Base::rs1();
      const auto got_fs = mul_fs(s);
      const auto got_sf = mul_sf(s);
      const auto exp_f = f(s) * v;
      CHECK_VALID(approximately_equal(got_fs, exp_f));
      CHECK_VALID(approximately_equal(got_sf, exp_f));
    }
    return valid;
  }
};

template <template <typename> typename Test, typename... Scalars>
struct ShuffleScalarTest {
  bool test() const {
    bool results[] = {Test<Scalars>().test()...};
    for (auto &res : results)
      if (!res)
        return false;
    return true;
  }
};

template <template <typename, typename> typename Test, typename... Scalars>
struct ShuffleScalarsTest {
  template <typename T> struct Shuffler {
    template <typename S> using TestT = Test<T, S>;

    bool test() const { return ShuffleScalarTest<TestT, Scalars...>().test(); }
  };

  bool test() const {
    bool results[] = {Shuffler<Scalars>().test()...};
    for (auto &res : results)
      if (!res)
        return false;
    return true;
  }
};

template <template <typename, typename, typename> typename Test, typename...>
struct PolynomialShuffler;

template <template <typename, typename, typename> typename Test,
          typename... Polynomials, typename... Scalars>
struct PolynomialShuffler<Test, std::tuple<Polynomials...>,
                          std::tuple<Scalars...>> {
  template <typename P> struct Shuffler {
    template <typename S1, typename S2> using TestP = Test<P, S1, S2>;
    bool test() const { return ShuffleScalarsTest<TestP, Scalars...>().test(); }
  };

  bool test() const {
    bool results[] = {Shuffler<Polynomials>().test()...};
    for (auto &res : results)
      if (!res)
        return false;
    return true;
  }
};

template <template <typename, typename, typename, typename> typename Test,
          typename...>
struct PolynomialsShuffler;

template <template <typename, typename, typename, typename> typename Test,
          typename... Polynomials1, typename... Polynomials2,
          typename... Scalars>
struct PolynomialsShuffler<Test, std::tuple<Polynomials1...>,
                           std::tuple<Polynomials2...>,
                           std::tuple<Scalars...>> {
  template <typename P1> struct Shuffler {
    template <typename P2, typename S1, typename S2>
    using TestP = Test<P1, P2, S1, S2>;
    bool test() const {
      return PolynomialShuffler<TestP, std::tuple<Polynomials2...>,
                                std::tuple<Scalars...>>()
          .test();
    }
  };

  bool test() const {
    bool results[] = {Shuffler<Polynomials1>().test()...};
    for (auto &res : results)
      if (!res)
        return false;
    return true;
  }
};

template <template <typename, int, int, typename, int, int, typename, typename>
          typename Test,
          typename Scalar, typename PolyScalar1, typename PolyScalar2, int...>
struct DegreeIterator;

template <template <typename, int, int, typename, int, int, typename, typename>
          typename Test,
          typename Scalar, typename PolyScalar1, typename PolyScalar2,
          int... degrees>
struct DegreeIterator {
  template <int degree1, int low_degree1, int degree2, int low_degree2,
            bool = low_degree2 <= degree2>
  struct LowerDegree2Iterator;
  template <int degree1, int low_degree1, int degree2, int low_degree2>
  struct LowerDegree2Iterator<degree1, low_degree1, degree2, low_degree2,
                              true> {
    template <typename P1, typename P2, typename S1, typename S2>
    using TestD =
        Test<P1, degree1, low_degree1, P2, degree2, low_degree2, S1, S2>;
    using DP1 = polynomials::DensePoly<PolyScalar1, degree1, low_degree1>;
    using DP2 = polynomials::DensePoly<PolyScalar2, degree2, low_degree2>;
    bool test() const {
      return PolynomialsShuffler<
                 TestD, std::tuple<DP1, Eigen::Map<DP1>, Eigen::Map<const DP1>>,
                 std::tuple<DP2, Eigen::Map<DP2>, Eigen::Map<const DP2>>,
                 std::tuple<Scalar, ceres::Jet<Scalar, 4>>>()
          .test();
    }
  };

  template <int degree1, int low_degree1, int degree2, int low_degree2>
  struct LowerDegree2Iterator<degree1, low_degree1, degree2, low_degree2,
                              false> {
    bool test() const { return true; }
  };

  template <int degree1, int low_degree1, int degree2> struct Degree2Iterator {
    template <int low_degree2>
    using Inner =
        LowerDegree2Iterator<degree1, low_degree1, degree2, low_degree2>;
    bool test() const {
      bool results[] = {Inner<degrees>().test()...};
      for (auto &res : results)
        if (!res)
          return false;
      return true;
    }
  };
  template <int degree1, int low_degree1, bool = low_degree1 <= degree1>
  struct LowerDegree1Iterator;
  template <int degree1, int low_degree1>
  struct LowerDegree1Iterator<degree1, low_degree1, true> {
    template <int degree2>
    using Inner = Degree2Iterator<degree1, low_degree1, degree2>;
    bool test() const {
      bool results[] = {Inner<degrees>().test()...};
      for (auto &res : results)
        if (!res)
          return false;
      return true;
    }
  };
  template <int degree1, int low_degree1>
  struct LowerDegree1Iterator<degree1, low_degree1, false> {
    bool test() const { return true; }
  };
  template <int degree1> struct Degree1Iterator {
    template <int low_degree1>
    using Inner = LowerDegree1Iterator<degree1, low_degree1>;
    bool test() const {
      bool results[] = {Inner<degrees>().test()...};
      for (auto &res : results)
        if (!res)
          return false;
      return true;
    }
  };
  bool test() const {
    bool results[] = {Degree1Iterator<degrees>().test()...};
    for (auto &res : results)
      if (!res)
        return false;
    return true;
  }
};

#endif
