/******************************************************************************
Copyright (c) 2021-2022 Dmitriy Korchemkin

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

#ifndef POLYNOMIALS_ROOTS_HPP
#define POLYNOMIALS_ROOTS_HPP

#include <Eigen/Dense>
#include <iosfwd>
#include <optional>

#include "polynomials/assert.hpp"
#include "polynomials/types.hpp"

#ifndef CERES_PUBLIC_JET_H
namespace ceres {
template <typename Scalar, int N> struct Jet;
}
#endif

namespace polynomials {

template <typename Scalar> struct RealRootFilter {
  std::optional<Scalar> operator()(const std::complex<Scalar> &c,
                                   const Scalar &tolerance) const {
    using std::abs;
    const auto re = c.real();
    const auto im = c.imag();
    const auto abs_re = abs(re);
    const auto abs_im = abs(im);
    if (abs_im < abs_re * tolerance)
      return re;
    return std::nullopt;
  }

  std::optional<Scalar> operator()(const std::complex<Scalar> &c) const {
    const auto re = c.real();
    const auto im = c.imag();
    if (im == Scalar(0.))
      return re;
    return std::nullopt;
  }
};

template <typename Scalar> struct PositiveRealRootFilter {
  std::optional<Scalar> operator()(const std::complex<Scalar> &c) const {
    auto real = rrf(c);
    if (!real)
      return std::nullopt;
    if (*real >= Scalar(0.))
      return real;
    return std::nullopt;
  }

  std::optional<Scalar> operator()(const std::complex<Scalar> &c,
                                   const Scalar &tolerance) const {
    auto real = rrf(c, tolerance);
    if (!real)
      return std::nullopt;
    if (*real >= Scalar(0.))
      return real;
    return std::nullopt;
  }

  RealRootFilter<Scalar> rrf;
};

template <typename Polynomial,
          bool fixed_degree = Polynomial::DegreeAtCompileTime != Dynamic>
struct QuotientRingMulXImpl;

template <typename Polynomial>
struct QuotientRingMulX : public QuotientRingMulXImpl<Polynomial> {
  using Base = QuotientRingMulXImpl<Polynomial>;

  template <typename... Args> QuotientRingMulX(Args... args) : Base(args...) {}
};

template <typename Polynomial> struct QuotientRingMulXImpl<Polynomial, true> {
  static constexpr Index DegreeAtCompileTime = Polynomial::DegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Polynomial::MaxDegreeAtCompileTime;
  using Scalar = typename Polynomial::Scalar;

  using CompanionMatrix =
      Eigen::Matrix<Scalar, DegreeAtCompileTime, DegreeAtCompileTime>;
  using ComplexRoots = typename CompanionMatrix::EigenvaluesReturnType;
  using RealRoots =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, DegreeAtCompileTime, 1>;

  QuotientRingMulXImpl(const Polynomial &p) {
    const Scalar lead_term = p[DegreeAtCompileTime];
    matrix
        .template bottomLeftCorner<DegreeAtCompileTime - 1,
                                   DegreeAtCompileTime - 1>()
        .setIdentity();
    matrix.col(DegreeAtCompileTime - 1) =
        -p.coeffs().template head<DegreeAtCompileTime>() *
        (Scalar(1.) / lead_term);
    matrix.row(0).template head<DegreeAtCompileTime - 1>().setZero();
  }

  operator CompanionMatrix() const { return matrix; }

  ComplexRoots complex_roots() const { return matrix.eigenvalues(); }

  template <typename Op, typename... Args>
  RealRoots filter_roots(Args... args) const {
    Index count_valid = 0;
    Scalar rr[DegreeAtCompileTime];
    const auto roots = complex_roots();
    const Op filter;
    for (Index i = 0; i < DegreeAtCompileTime; ++i) {
      auto real = filter(roots[i], args...);
      if (!real)
        continue;
      rr[count_valid++] = *real;
    }
    return Eigen::Map<const RealRoots>(rr, count_valid, 1);
  }

  template <typename... Args> RealRoots real_roots(Args... args) const {
    return filter_roots<RealRootFilter<Scalar>, Args...>(args...);
  }

  template <typename... Args>
  RealRoots positive_real_roots(Args... args) const {
    return filter_roots<PositiveRealRootFilter<Scalar>, Args...>(args...);
  }

private:
  CompanionMatrix matrix;
};

template <typename Polynomial> struct QuotientRingMulXImpl<Polynomial, false> {
  static constexpr Index DegreeAtCompileTime = Polynomial::DegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Polynomial::MaxDegreeAtCompileTime;
  using Scalar = typename Polynomial::Scalar;

  using CompanionMatrix =
      Eigen::Matrix<Scalar, DegreeAtCompileTime, DegreeAtCompileTime, 0,
                    MaxDegreeAtCompileTime, MaxDegreeAtCompileTime>;
  using ComplexRoots = typename CompanionMatrix::EigenvaluesReturnType;
  using RealRoots =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, MaxDegreeAtCompileTime, 1>;

  QuotientRingMulXImpl(const Polynomial &p) {
    const Index degree = p.degree();
    matrix.resize(degree, degree);
    const Scalar lead_term = p[degree];
    matrix.bottomLeftCorner(degree - 1, degree - 1).setIdentity();
    matrix.col(DegreeAtCompileTime - 1) =
        -p.coeffs().head(degree) * (Scalar(1.) / lead_term);
    matrix.row(0).head(degree - 1).setZero();
  }

  operator CompanionMatrix() const { return matrix; }

  ComplexRoots complex_roots() const { return matrix.eigenvalues(); }

  RealRoots real_roots() const {
    Index count_real = 0;
    const Index degree = dim();
    Eigen::Matrix<Scalar, MaxDegreeAtCompileTime, 1> rr(degree, 1);

    const auto roots = complex_roots();

    for (Index i = 0; i < degree; ++i) {
      auto im = roots[i].imag();
      if (im != Scalar(0.))
        continue;
      auto re = roots[i].real();
      rr[count_real++] = re;
    }
    return Eigen::Map<const RealRoots>(rr, count_real, 1);
  }

  RealRoots real_roots(const Scalar &tolerance) const {
    Index count_real = 0;
    const Index degree = dim();
    Eigen::Matrix<Scalar, MaxDegreeAtCompileTime, 1> rr(degree, 1);

    const auto roots = complex_roots();

    for (Index i = 0; i < degree; ++i) {
      auto re = roots[i].real();
      auto are = std::abs(re);
      auto aim = std::abs(roots[i].imag);
      if (!(aim < tolerance * are))
        continue;
      rr[count_real++] = re;
    }
    return Eigen::Map<const RealRoots>(rr, count_real, 1);
  }

  Index dim() const { return matrix.rows(); }

private:
  CompanionMatrix matrix;
};

template <typename Poly, template <typename> typename Algo, typename Scalar>
struct RootFinder {
  using Algorithm = Algo<Poly>;
  using ComplexRoots = typename Algorithm::ComplexRoots;
  using RealRoots = typename Algorithm::RealRoots;

  template <typename... Args>
  static RealRoots real_roots(const Poly &p, Args... args) {
    return Algorithm(p).real_roots(args...);
  }

  template <typename... Args>
  static RealRoots positive_real_roots(const Poly &p, Args... args) {
    return Algorithm(p).positive_real_roots(args...);
  }

  template <typename... Args>
  static ComplexRoots complex_roots(const Poly &p, Args... args) {
    return Algorithm(p).complex_roots(args...);
  }
};

template <typename Poly, template <typename> typename Algo, typename Scalar,
          int N>
struct RootFinder<Poly, Algo, ceres::Jet<Scalar, N>> {
  using ScalarPoly = DensePoly<Scalar, Poly::DegreeAtCompileTime,
                               Poly::MaxDegreeAtCompileTime>;
  using ScalarRootFinder = RootFinder<ScalarPoly, Algo>;
  using ScalarAlgorithm = Algo<ScalarPoly>;
  using ScalarComplexRoots = typename ScalarAlgorithm::ComplexRoots;
  using ScalarRealRoots = typename ScalarAlgorithm::RealRoots;
  using Jet = ceres::Jet<Scalar, N>;
  using ComplexJet = ceres::Jet<typename ScalarComplexRoots::Scalar, N>;

  using JetComplexRoots = Eigen::Matrix<ComplexJet, Poly::DegreeAtCompileTime,
                                        1, 0, Poly::MaxDegreeAtCompileTime, 1>;
  using JetRealRoots =
      Eigen::Matrix<Jet, Eigen::Dynamic, 1, 0, Poly::MaxDegreeAtCompileTime, 1>;

  static ScalarPoly cast(const Poly &p) {
    ScalarPoly sp(p.degree());
    const Index num_coeffs = p.degree() + 1;
    for (Index i = 0; i < num_coeffs; ++i)
      sp.coeffs()[i] = p.coeffs()[i].a;
    return sp;
  }

  template <typename T>
  static ceres::Jet<T, N>
  process_root(const Poly &poly, const ScalarPoly &scalar_poly, const T &root) {
    ceres::Jet<T, N> res;
    res.a = root;
    res.v.setZero();
    const T denom(-scalar_poly.df(root));
    const auto &coeffs = poly.coeffs();
    const Index num_coeffs = coeffs.size();
    T root_pow = T(1.);
    for (Index i = 0; i < num_coeffs; ++i) {
      // dx / da_k = -x^k / f'
      // dx / dp_i = \sum_k dx / da_k da_k / dp_i
      // dx / dp = (\sum_k -x^k [da_k / dp]) / f'

      res.v += root_pow * coeffs[i].v;
      root_pow *= root;
    }
    res.v /= denom;
    return res;
  }

  template <typename... Args>
  static JetComplexRoots complex_roots(const Poly &p, Args... args) {
    const ScalarPoly scalar = cast(p);
    const ScalarComplexRoots scalar_roots =
        ScalarRootFinder::complex_roots(scalar, args...);
    const Index num_roots = scalar_roots.size();
    JetComplexRoots jet(num_roots);
    for (Index i = 0; i < num_roots; ++i)
      jet[i] = process_root(p, scalar, scalar_roots[i]);
    return jet;
  }

  template <typename... Args>
  static JetRealRoots real_roots(const Poly &p, Args... args) {
    const ScalarPoly scalar = cast(p);
    const ScalarRealRoots scalar_roots =
        ScalarRootFinder::real_roots(scalar, args...);
    const Index num_real_roots = scalar_roots.size();
    JetRealRoots jet(num_real_roots);
    for (Index i = 0; i < num_real_roots; ++i)
      jet[i] = process_root(p, scalar, scalar_roots[i]);
    return jet;
  }

  template <typename... Args>
  static JetRealRoots positive_real_roots(const Poly &p, Args... args) {
    const ScalarPoly scalar = cast(p);
    const ScalarRealRoots scalar_roots =
        ScalarRootFinder::positive_real_roots(scalar, args...);
    const Index num_real_roots = scalar_roots.size();
    JetRealRoots jet(num_real_roots);
    for (Index i = 0; i < num_real_roots; ++i)
      jet[i] = process_root(p, scalar, scalar_roots[i]);
    return jet;
  }
};

} // namespace polynomials

#endif
