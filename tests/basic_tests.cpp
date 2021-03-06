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

#include "polynomials/dense_polynomial.hpp"

#include "testing.hpp"

using namespace polynomials;
using Quadric = DensePoly<double, 2>;
using Cubic = DensePoly<double, 3>;
using PolyX = DensePoly<double, Dynamic>;
using QuadricMap = Eigen::Map<Quadric>;
using CubicMap = Eigen::Map<Cubic>;
using PolyXMap = Eigen::Map<PolyX>;
using QuadricCMap = Eigen::Map<const Quadric>;
using CubicCMap = Eigen::Map<const Cubic>;
using PolyXCMap = Eigen::Map<const PolyX>;

TEST_CASE("PolyOps") {

  Quadric q;
  Cubic c;
  PolyX x;
  q << 1., -2., 3.;
  c << -4., 5., -6., 7.;

  bool valid = true;

  const auto sum = q + c;
  x = q + c;
  static_assert(decltype(sum)::DegreeAtCompileTime == 3);
  for (double X = -5.; X <= 5.; X += 1.) {
    CHECK_VALID(approximately_equal(sum(X), x(X)));
    CHECK_VALID(approximately_equal(sum(X), q(X) + c(X)));
  }

  const auto sub = q - c;
  x = q - c;
  static_assert(decltype(sub)::DegreeAtCompileTime == 3);
  for (double X = -5.; X <= 5.; X += 1.) {
    CHECK_VALID(approximately_equal(sub(X), x(X)));
    CHECK_VALID(approximately_equal(sub(X), q(X) - c(X)));
  }

  const auto mul = q * c;
  x = q * c;
  static_assert(decltype(mul)::DegreeAtCompileTime == 5);
  for (double X = -5.; X <= 5.; X += 1.) {
    CHECK_VALID(approximately_equal(mul(X), x(X)));
    CHECK_VALID(approximately_equal(mul(X), q(X) * c(X)));
  }
}

TEST_CASE("Roots") {
  bool valid = true;

  Quadric q;
  q << -6., -1., 1.;

  const auto q1_cr = q.roots();
  const auto q1_rr = q.real_roots();
  const auto q1_pr = q.positive_real_roots();
  CHECK_VALID(q1_cr.size() == 2);
  CHECK_VALID(q1_rr.size() == 2);
  CHECK_VALID(q1_pr.size() == 1);

  q << 6., 1., 1.;
  const auto q2_cr = q.roots();
  const auto q2_rr = q.real_roots();
  CHECK_VALID(q2_cr.size() == 2);
  CHECK_VALID(q2_rr.size() == 0);

  Cubic c;
  c << -5., 3., 1., 1.;

  const auto c1_cr = c.roots();
  const auto c1_rr = c.real_roots();
  const auto c1_pr = c.positive_real_roots();
  CHECK_VALID(c1_cr.size() == 3);
  CHECK_VALID(c1_rr.size() == 1);
  CHECK_VALID(c1_pr.size() == 1);
  CHECK_VALID(approximately_equal(c1_pr[0], 1.));
}

TEST_CASE("Jet roots real") {
  bool valid = true;

  using Jet = ceres::Jet<double, 3>;
  using JetQuadric = polynomials::DensePoly<Jet, 2>;

  JetQuadric jq;
  jq << Jet(-6., 0), Jet(-1., 1), Jet(1., 2);

  auto real_roots = jq.real_roots();
  CHECK_VALID(real_roots.size() == 2);

  if (real_roots[0] > real_roots[1])
    std::swap(real_roots[0], real_roots[1]);

  CHECK_VALID(approximately_equal(real_roots[0].a, -2.))
  CHECK_VALID(approximately_equal(real_roots[1].a, 3.))

  Eigen::Matrix<double, 2, 3> J_exp, J_got;
  J_exp.row(0) << 0.2, -0.4, 0.8;
  J_exp.row(1) << -0.2, -0.6, -1.8;

  J_got.row(0) = real_roots[0].v.transpose();
  J_got.row(1) = real_roots[1].v.transpose();
  CHECK_VALID(approximately_equal_matrices(J_exp, J_got));
}

TEST_CASE("Jet roots complex") {
  bool valid = true;

  using Jet = ceres::Jet<double, 3>;
  using JetQuadric = polynomials::DensePoly<Jet, 2>;

  JetQuadric jq;
  jq << Jet(2., 0), Jet(2., 1), Jet(1., 2);
  auto complex_roots = jq.roots();
  if (complex_roots[0].a.imag() > complex_roots[1].a.imag())
    std::swap(complex_roots[0], complex_roots[1]);

  CHECK_VALID(approximately_equal(complex_roots[0].a, std::complex(-1., -1.)));
  CHECK_VALID(approximately_equal(complex_roots[1].a, std::complex(-1., 1.)));

  Eigen::Matrix<std::complex<double>, 2, 3> J_exp_cmplx, J_got_cmplx;

  J_exp_cmplx.row(0) << std::complex(0., -0.5), std::complex(-0.5, 0.5),
      std::complex(1., 0.);
  J_exp_cmplx.row(1) << std::complex(0., 0.5), std::complex(-0.5, -0.5),
      std::complex(1., 0.);

  J_got_cmplx.row(0) = complex_roots[0].v.transpose();
  J_got_cmplx.row(1) = complex_roots[1].v.transpose();
  CHECK_VALID(approximately_equal_matrices(J_exp_cmplx, J_got_cmplx));
}

TEST_CASE("BasicProps") {
  static_assert(Quadric::DegreeAtCompileTime == 2);
  static_assert(Cubic::DegreeAtCompileTime == 3);
  static_assert(PolyX::DegreeAtCompileTime == Dynamic);
  static_assert(Quadric::Base::DegreeAtCompileTime == 2);
  static_assert(Cubic::Base::DegreeAtCompileTime == 3);
  static_assert(PolyX::Base::DegreeAtCompileTime == Dynamic);

  static_assert(QuadricMap::DegreeAtCompileTime == 2);
  static_assert(CubicMap::DegreeAtCompileTime == 3);
  static_assert(PolyXMap::DegreeAtCompileTime == Dynamic);
  static_assert(QuadricMap::Base::DegreeAtCompileTime == 2);
  static_assert(CubicMap::Base::DegreeAtCompileTime == 3);
  static_assert(PolyXMap::Base::DegreeAtCompileTime == Dynamic);

  static_assert(QuadricCMap::DegreeAtCompileTime == 2);
  static_assert(CubicCMap::DegreeAtCompileTime == 3);
  static_assert(PolyXCMap::DegreeAtCompileTime == Dynamic);
  static_assert(QuadricCMap::Base::DegreeAtCompileTime == 2);
  static_assert(CubicCMap::Base::DegreeAtCompileTime == 3);
  static_assert(PolyXCMap::Base::DegreeAtCompileTime == Dynamic);
}
