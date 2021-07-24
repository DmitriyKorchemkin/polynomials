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
#ifndef POLYNOMIALS_DENSE_SCALAR_TESTS_HPP
#define POLYNOMIALS_DENSE_SCALAR_TESTS_HPP

#include "testing.hpp"
template <typename Poly1, int deg1, typename Scalar1, typename Scalar2>
struct ScalarTestBase {
  using ScalarP1 = typename Poly1::Scalar;
  using RS1 = Random<Scalar1>;
  using RS2 = Random<Scalar2>;
  using RSP1 = Random<ScalarP1>;

  RS1 rs1;
  RS2 rs2;
  RSP1 rsp1;

  ScalarTestBase() : poly1(deg1) {
    for (int i = 0; i < poly1.mutable_data().total_coeffs(); ++i)
      poly1.mutable_data().coeffs()[i] = rsp1();
  }

  PolynomialHolder<Poly1> poly1;
};

template <typename...> struct PlusScalarTest;

template <typename Poly1, typename deg1, typename Scalar1, typename Scalar2,
          typename PolyScalar>
struct PlusScalarTest<Scalar1, Scalar2, Poly1, deg1, PolyScalar>
    : ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2> {
  using Base = ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2>;
  static bool Check() { return PlusScalarTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();

    auto v = Base::rs2();
    auto sum_fs = f + v;
    auto sum_sf = v + f;
    for (int i = 0; i < 2 * exp_degree; ++i) {
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

template <typename...> struct MinusScalarTest;

template <typename Poly1, typename deg1, typename Scalar1, typename Scalar2,
          typename PolyScalar>
struct MinusScalarTest<Scalar1, Scalar2, Poly1, deg1, PolyScalar>
    : ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2> {
  using Base = ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2>;
  static bool Check() { return MinusScalarTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();
    const auto v = Base::rs2();
    const auto sub_fs = f - v;
    const auto sub_sf = v - f;
    for (int i = 0; i < 2 * exp_degree; ++i) {
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

template <typename...> struct MulScalarTest;

template <typename Poly1, typename deg1, typename Scalar1, typename Scalar2,
          typename PolyScalar>
struct MulScalarTest<Scalar1, Scalar2, Poly1, deg1, PolyScalar>
    : ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2> {
  using Base = ScalarTestBase<Poly1, deg1::value, Scalar1, Scalar2>;
  static bool Check() { return MulScalarTest().test(); }
  bool test() const {
    bool valid = true;
    const auto &f = *Base::poly1;
    const auto exp_degree = f.degree();
    const auto v = Base::rs2();
    const auto mul_fs = f * v;
    const auto mul_sf = v * f;
    for (int i = 0; i < 2 * exp_degree; ++i) {
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

#endif
