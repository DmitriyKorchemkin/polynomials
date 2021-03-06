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
#ifndef POLYNOMIALS_DENSE_POLYNOMIAL_TESTS_HPP
#define POLYNOMIALS_DENSE_POLYNOMIAL_TESTS_HPP

#include "testing.hpp"

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar>
struct TestBase {
  using ScalarP1 = typename Poly1::Scalar;
  using ScalarP2 = typename Poly2::Scalar;
  using RS = Random<Scalar>;
  using RSP1 = Random<ScalarP1>;
  using RSP2 = Random<ScalarP2>;

  RS rs;
  RSP1 rsp1;
  RSP2 rsp2;

  TestBase() : poly1(deg1), poly2(deg2) {
    for (int i = 0; i < poly1.mutable_data().total_coeffs(); ++i)
      poly1.mutable_data().coeffs()[i] = rsp1();
    for (int i = 0; i < poly2.mutable_data().total_coeffs(); ++i)
      poly2.mutable_data().coeffs()[i] = rsp2();
  }

  PolynomialHolder<Poly1> poly1;
  PolynomialHolder<Poly2> poly2;
};

template <typename... args> struct PlusTest;

template <typename Scalar, typename Poly1, typename deg1, typename PolyScalar1,
          typename Poly2, typename deg2, typename PolyScalar2>
struct PlusTest<Scalar, Poly1, deg1, PolyScalar1, Poly2, deg2, PolyScalar2>
    : TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar> {
  using Base = TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar>;
  static bool Check() { return PlusTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto sum_fg = f + g;
    const auto exp_degree = std::max(f.degree(), g.degree());
    for (int i = 0; i < 2 * exp_degree; ++i) {
      const auto s = Base::rs();
      const auto got_fg = sum_fg(s);
      const auto exp = f(s) + g(s);
      CHECK_VALID(approximately_equal(got_fg, exp));
    }
    return valid;
  }
};

template <typename... args> struct MinusTest;

template <typename Scalar, typename Poly1, typename deg1, typename PolyScalar1,
          typename Poly2, typename deg2, typename PolyScalar2>
struct MinusTest<Scalar, Poly1, deg1, PolyScalar1, Poly2, deg2, PolyScalar2>
    : TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar> {
  using Base = TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar>;
  static bool Check() { return MinusTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto sub_fg = f - g;
    const auto exp_degree = std::max(f.degree(), g.degree());
    for (int i = 0; i < 2 * exp_degree; ++i) {
      const auto s = Base::rs();
      const auto got_fg = sub_fg(s);
      const auto exp_fg = f(s) - g(s);
      CHECK_VALID(approximately_equal(got_fg, exp_fg));
    }
    return valid;
  }
};
template <typename... args> struct MulTest;

template <typename Scalar, typename Poly1, typename deg1, typename PolyScalar1,
          typename Poly2, typename deg2, typename PolyScalar2>
struct MulTest<Scalar, Poly1, deg1, PolyScalar1, Poly2, deg2, PolyScalar2>
    : TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar> {
  using Base = TestBase<Poly1, deg1::value, Poly2, deg2::value, Scalar>;
  static bool Check() { return MulTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    const auto mul_fg = f * g;
    const auto exp_degree = f.degree() + g.degree();
    for (int i = 0; i < 2 * exp_degree; ++i) {
      const auto s = Base::rs();
      const auto got_fg = mul_fg(s);
      const auto exp = f(s) * g(s);
      CHECK_VALID(approximately_equal(got_fg, exp));
    }
    return valid;
  }
};




template <typename... args> struct CopyTest;

template <typename P>
struct is_map : std::false_type {};

template <typename P>
struct is_map<Eigen::Map<P>> : std::true_type {};

template <typename P>
constexpr bool is_map_v = is_map<P>::value;

template <typename P> struct is_const_map : std::false_type {};

template <typename P>
struct is_const_map<Eigen::Map<const P>> : std::true_type {};

template <typename P> constexpr bool is_const_map_v = is_const_map<P>::value;

template <typename Poly1, int deg1, typename Poly2, int deg2>
constexpr bool is_copyconstructible() {
  if (!std::is_same_v<typename Poly1::Scalar, typename Poly2::Scalar>) return false;
  if (Poly1::DegreeAtCompileTime != polynomials::Dynamic && deg1 < deg2) return false;
  if (Poly1::MaxDegreeAtCompileTime != polynomials::Dynamic && Poly1::MaxDegreeAtCompileTime < deg2) return false;
  if (is_map_v<Poly1>) return false;
  return true;
}

template <typename Poly1, int deg1, typename Poly2, int deg2>
constexpr bool
    is_copyconstructible_v = is_copyconstructible<Poly1, deg1, Poly2, deg2>();

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar,
          bool assignable = is_copyconstructible_v<Poly1, deg1, Poly2, deg2>>
struct CopyTestImpl;

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar>
struct CopyTestImpl<Poly1, deg1, Poly2, deg2, Scalar, false>{
  static constexpr bool Check() {
    return true;
  }
};

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar>
struct CopyTestImpl<Poly1, deg1, Poly2, deg2, Scalar, true> : TestBase<Poly1, deg1, Poly2, deg2, Scalar> {
  using Base = TestBase<Poly1, deg1, Poly2, deg2, Scalar>;

  static bool Check() {
    return CopyTestImpl().test();
  }

  bool test() const {
    bool valid = true;
    const auto &g = *Base::poly2;
    const Poly1 ff(g);
    const auto exp_degree = g.degree();
    for (int i = 0; i < 2*exp_degree; ++i) {
      const auto s = Base::rs();
      const auto got_g = ff(s);
      const auto exp_g = g(s);
      CHECK_VALID(approximately_equal(got_g, exp_g));
    }
    return valid;
  }
};

template <typename Scalar, typename Poly1, typename deg1, typename PolyScalar1,
          typename Poly2, typename deg2, typename PolyScalar2>
struct CopyTest<Scalar, Poly1, deg1, PolyScalar1, Poly2, deg2, PolyScalar2> {
  static constexpr bool Check() {
    return CopyTestImpl<Poly1, deg1::value, Poly2, deg2::value,
                        Scalar>::Check();
  }
};

template <typename Poly1, int deg1, typename Poly2, int deg2>
constexpr bool is_assignable() {
  if (!std::is_same_v<typename Poly1::Scalar, typename Poly2::Scalar>)
    return false;
  if (Poly1::DegreeAtCompileTime != polynomials::Dynamic && deg1 < deg2)
    return false;
  if (Poly1::MaxDegreeAtCompileTime != polynomials::Dynamic &&
      Poly1::MaxDegreeAtCompileTime < deg2)
    return false;
  if (is_const_map_v<Poly1>)
    return false;
  if (is_map_v<Poly1> && deg1 < deg2)
    return false;
  return true;
}

template <typename Poly1, int deg1, typename Poly2, int deg2>
constexpr bool is_assignable_v = is_assignable<Poly1, deg1, Poly2, deg2>();

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar,
          bool assignable = is_assignable_v<Poly1, deg1, Poly2, deg2>>
struct AssignTestImpl;

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar>
struct AssignTestImpl<Poly1, deg1, Poly2, deg2, Scalar, false> {
  static constexpr bool Check() { return true; }
};

template <typename Poly1, int deg1, typename Poly2, int deg2, typename Scalar>
struct AssignTestImpl<Poly1, deg1, Poly2, deg2, Scalar, true>
    : TestBase<Poly1, deg1, Poly2, deg2, Scalar> {
  using Base = TestBase<Poly1, deg1, Poly2, deg2, Scalar>;

  static bool Check() { return AssignTestImpl().test(); }

  bool test() {
    bool valid = true;
    auto &f = *Base::poly1;
    const auto &g = *Base::poly2;
    f = g;
    const auto exp_degree = g.degree();
    for (int i = 0; i < 2 * exp_degree; ++i) {
      const auto s = Base::rs();
      const auto got_g = f(s);
      const auto exp_g = g(s);
      CHECK_VALID(approximately_equal(got_g, exp_g));
    }
    return valid;
  }
};

template <typename... args> struct AssignTest;

template <typename Scalar, typename Poly1, typename deg1, typename PolyScalar1,
          typename Poly2, typename deg2, typename PolyScalar2>
struct AssignTest<Scalar, Poly1, deg1, PolyScalar1, Poly2, deg2, PolyScalar2> {
  static constexpr bool Check() {
    return AssignTestImpl<Poly1, deg1::value, Poly2, deg2::value,
                          Scalar>::Check();
  }
};

#endif
