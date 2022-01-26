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

#ifndef POLYNOMIALS_DENSE_OPERATORS_HPP
#define POLYNOMIALS_DENSE_OPERATORS_HPP

#include "polynomials/dense_polynomial.hpp"

namespace polynomials {
template <typename DerivedLhs, typename DerivedRhs>
auto operator*(const DensePolyBase<DerivedLhs> &lhs,
               const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_mul_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  using MulTraits = polynomial_mul_trait<DensePolyBase<DerivedLhs>,
                                         DensePolyBase<DerivedRhs>>;
  using Result = typename MulTraits::ResultType;
  Result res(lhs.degree() + rhs.degree());

  if constexpr (MulTraits::DegreeAtCompileTime != Dynamic) {
    using Lhs = typename MulTraits::Lhs;
    using Rhs = typename MulTraits::Rhs;
    // using Scalar = typename MulTraits::Scalar;
    constexpr Index tail = Rhs::CoeffsCompileTime;
    constexpr Index head = Lhs::CoeffsCompileTime - 1;
    static_assert(tail != Dynamic && tail > 0);
    static_assert(head != Dynamic && head >= 0);

    auto &c = res.coeffs();
    c.template tail<tail>() = rhs.coeffs() * lhs[Lhs::DegreeAtCompileTime];
    c.template head<head>() = lhs.coeffs().template head<head>() * rhs[0];

    constexpr Index mul_rhs_len = Rhs::CoeffsCompileTime - 1;
    for (Index lhs_degree = 0; lhs_degree < Lhs::DegreeAtCompileTime;
         ++lhs_degree) {
      c.template segment<mul_rhs_len>(lhs_degree + 1) +=
          rhs.coeffs().template tail<mul_rhs_len>() * lhs[lhs_degree];
    }
  } else {
    const Index tail = rhs.total_coeffs();
    const Index head = lhs.total_coeffs() - 1;
    POLYNOMIALS_ASSERT(tail != Dynamic && tail > 0,
                       "Invalid multiplication size");
    POLYNOMIALS_ASSERT(head != Dynamic && head >= 0,
                       "Invalid multiplication size");

    auto &c = res.coeffs();
    c.tail(tail) = rhs.coeffs() * lhs[lhs.degree()];
    c.head(head) = lhs.coeffs().head(head) * rhs[0];

    const Index mul_rhs_len = rhs.total_coeffs() - 1;
    for (Index lhs_degree = 0; lhs_degree < lhs.degree(); ++lhs_degree) {
      c.segment(lhs_degree + 1, mul_rhs_len) +=
          rhs.coeffs().tail(mul_rhs_len) * lhs[lhs_degree];
    }
  }
  return res;
}

template <typename DerivedLhs, typename DerivedRhs>
auto operator+(const DensePolyBase<DerivedLhs> &lhs,
               const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  return sum_like_op<DerivedLhs, DerivedRhs, AddOp>(lhs, rhs);
}

template <typename DerivedLhs, typename DerivedRhs>
auto operator-(const DensePolyBase<DerivedLhs> &lhs,
               const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  return sum_like_op<DerivedLhs, DerivedRhs, SubOp>(lhs, rhs);
}


template <typename Derived1>
template <typename Derived2>
Derived1 &
DensePolyBase<Derived1>::operator=(const DensePolyBase<Derived2> &rhs) {
  using Lhs = DensePolyBase<Derived1>;
  using Rhs = DensePolyBase<Derived2>;
  static_assert(potentially_assignable_v<Lhs, Rhs>,
                "Incompatible types to assign");

  return AssignOp<Lhs, Rhs>()(*this, rhs).derived();
}

template <typename Derived, char var = 'x'>
std::ostream &operator<<(std::ostream &os, const DensePolyBase<Derived> &poly) {
  using Poly = DensePolyBase<Derived>;
  using Scalar = typename Poly::Scalar;
  const Scalar zero(0);
  const Scalar one(1);
  const Scalar neg_one(-1);

  bool not_first = false;
  bool showpos = os.flags() & std::ios_base::showpos;
  for (Index i = 0; i <= poly.degree(); ++i) {
    const auto &C = poly[i];
    if (C == zero)
      continue;

    bool need_plus = not_first && !showpos && C > zero;
    if (need_plus)
      os << "+";
    not_first = true;
    if ((C != one && C != neg_one) || i == 0) {
      os << C;
      if (i == 0)
        continue;
      os << '*';
    } else if (C == neg_one) {
      os << '-';
    }
    os << var;

    if (i == 1)
      continue;
    os << '^' << i;
  }
  if (!not_first)
    os << '0';
  return os;
}

} // namespace polynomials

#endif
