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

#ifndef POLYNOMIALS_DENSE_OPERATORS_HPP
#define POLYNOMIALS_DENSE_OPERATORS_HPP

#include <iosfwd>

namespace polynomials {
template <typename Lhs, typename Rhs> struct polynomial_mul_trait;

template <typename DerivedLhs, typename DerivedRhs>
struct polynomial_mul_trait<DensePolyBase<DerivedLhs>,
                            DensePolyBase<DerivedRhs>> {
  using Lhs = DensePolyBase<DerivedLhs>;
  using Rhs = DensePolyBase<DerivedRhs>;
  using Scalar =
      typename Eigen::ScalarBinaryOpTraits<typename Lhs::Scalar,
                                           typename Rhs::Scalar>::ReturnType;
  static constexpr Index DegreeAtCompileTime =
      Lhs::DegreeAtCompileTime == Eigen::Dynamic ||
              Rhs::DegreeAtCompileTime == Eigen::Dynamic
          ? Eigen::Dynamic
          : Lhs::DegreeAtCompileTime + Rhs::DegreeAtCompileTime;
  static constexpr Index LowDegreeAtCompileTime =
      Lhs::LowDegreeAtCompileTime == Eigen::Dynamic ||
              Rhs::LowDegreeAtCompileTime == Eigen::Dynamic
          ? Eigen::Dynamic
          : Lhs::LowDegreeAtCompileTime + Rhs::LowDegreeAtCompileTime;

  using ResultType =
      DensePoly<Scalar, DegreeAtCompileTime, LowDegreeAtCompileTime>;
};

template <typename Lhs, typename Rhs> struct polynomial_sum_trait;

template <typename DerivedLhs, typename DerivedRhs>
struct polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                            DensePolyBase<DerivedRhs>> {
  using Lhs = DensePolyBase<DerivedLhs>;
  using Rhs = DensePolyBase<DerivedRhs>;
  using Scalar =
      typename Eigen::ScalarBinaryOpTraits<typename Lhs::Scalar,
                                           typename Rhs::Scalar>::ReturnType;
  static constexpr Index DegreeAtCompileTime =
      max_or_dynamic(Lhs::DegreeAtCompileTime, Rhs::DegreeAtCompileTime);
  static constexpr Index LowDegreeAtCompileTime =
      min_or_dynamic(Lhs::LowDegreeAtCompileTime, Rhs::LowDegreeAtCompileTime);

  using ResultType =
      DensePoly<Scalar, DegreeAtCompileTime, LowDegreeAtCompileTime>;
};

struct AddOp {
  template <typename Lhs, typename Rhs>
  auto operator()(const Lhs &lhs, const Rhs &rhs) const -> decltype(lhs + rhs) {
    return lhs + rhs;
  }
  template <typename D> const D &left_op(const D &d) const { return d; }
  template <typename D> const D &right_op(const D &d) const { return d; }
};

struct SubOp {
  template <typename Lhs, typename Rhs>
  auto operator()(const Lhs &lhs, const Rhs &rhs) const -> decltype(lhs - rhs) {
    return lhs - rhs;
  }
  template <typename D> const D &left_op(const D &d) const { return d; }
  template <typename D> auto right_op(const D &d) const -> decltype(-d) {
    return -d;
  }
};

template <typename DerivedLhs, typename DerivedRhs, typename Op>
auto sum_like_op(const DensePolyBase<DerivedLhs> &lhs,
                 const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  using SumTraits = polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                         DensePolyBase<DerivedRhs>>;
  using Result = typename SumTraits::ResultType;
  using ResultScalar = typename Result::Scalar;
  const auto l_deg = lhs.degree();
  const auto r_deg = rhs.degree();
  const auto s_deg = std::max(l_deg, r_deg);
  const auto l_low_deg = lhs.low_degree();
  const auto r_low_deg = rhs.low_degree();
  const auto s_low_deg = std::min(l_low_deg, r_low_deg);
  Result sum(s_deg, s_low_deg);
  auto &s_coeffs = sum.coeffs();
  auto &l_coeffs = lhs.coeffs();
  auto &r_coeffs = rhs.coeffs();

  const Op op;

  // setup higher degrees
  if (s_deg != l_deg) {
    if constexpr (SumTraits::DegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index tail_len =
          SumTraits::DegreeAtCompileTime - SumTraits::Lhs::DegreeAtCompileTime;
      constexpr Index tail_present =
          std::min(tail_len, SumTraits::Rhs::CoeffsCompileTime);
      constexpr Index tail_zero = tail_len - tail_present;
      if (tail_present) {
        s_coeffs.template tail<tail_present>() =
            op.right_op(r_coeffs.template tail<tail_present>())
                .template cast<ResultScalar>();
      }
      if (tail_zero) {
        constexpr Index tail_zero_base = SumTraits::Lhs::DegreeAtCompileTime -
                                         SumTraits::LowDegreeAtCompileTime + 1;

        s_coeffs.template segment<tail_zero>(tail_zero_base).setZero();
      }
    } else {
      const Index tail_len = s_deg - r_deg;
      s_coeffs.tail(tail_len) = op.right_op(r_coeffs.tail(tail_len));
    }
  }
  if (s_deg != r_deg) {
    if constexpr (SumTraits::DegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index tail_len =
          SumTraits::DegreeAtCompileTime - SumTraits::Rhs::DegreeAtCompileTime;
      constexpr Index tail_present =
          std::min(tail_len, SumTraits::Lhs::CoeffsCompileTime);
      constexpr Index tail_zero = tail_len - tail_present;

      if (tail_present) {
        s_coeffs.template tail<tail_present>() =
            op.left_op(l_coeffs.template tail<tail_present>())
                .template cast<ResultScalar>();
      }
      if (tail_zero) {
        constexpr Index tail_zero_base = SumTraits::Rhs::DegreeAtCompileTime -
                                         SumTraits::LowDegreeAtCompileTime + 1;

        s_coeffs.template segment<tail_zero>(tail_zero_base).setZero();
      }
    } else {
      const Index tail_len = s_deg - l_deg;
      s_coeffs.tail(tail_len) = op.left_op(l_coeffs.tail(tail_len));
    }
  }
  // setup lower degrees
  if (s_low_deg != l_low_deg) {
    if constexpr (SumTraits::LowDegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index head_len = SumTraits::Lhs::LowDegreeAtCompileTime -
                                 SumTraits::LowDegreeAtCompileTime;
      constexpr Index head_present_len =
          std::min(head_len, SumTraits::Rhs::CoeffsCompileTime);
      constexpr Index head_zero_len = head_len - head_present_len;
      if (head_present_len) {
        s_coeffs.template head<head_present_len>() =
            op.right_op(r_coeffs.template head<head_present_len>())
                .template cast<ResultScalar>();
      }
      if (head_zero_len) {
        s_coeffs.template segment<head_zero_len>(head_present_len).setZero();
      }
    } else {
      const Index head_len = l_low_deg - s_low_deg;
      s_coeffs.head(head_len) = op.right_op(r_coeffs.head(head_len));
    }
  }
  if (s_low_deg != r_low_deg) {
    if constexpr (SumTraits::LowDegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index head_len = SumTraits::Rhs::LowDegreeAtCompileTime -
                                 SumTraits::LowDegreeAtCompileTime;
      constexpr Index head_present_len =
          std::min(head_len, SumTraits::Lhs::CoeffsCompileTime);
      constexpr Index head_zero_len = head_len - head_present_len;
      if (head_present_len) {
        s_coeffs.template head<head_present_len>() =
            op.left_op(l_coeffs.template head<head_present_len>())
                .template cast<ResultScalar>();
      }
      if (head_zero_len) {
        s_coeffs.template segment<head_zero_len>(head_present_len).setZero();
      }
    } else {
      const Index head_len = r_low_deg - s_low_deg;
      s_coeffs.head(head_len) = op.left_op(l_coeffs.head(head_len));
    }
  }
  // setup middle
  if constexpr (SumTraits::LowDegreeAtCompileTime != Eigen::Dynamic &&
                SumTraits::DegreeAtCompileTime != Eigen::Dynamic) {
    constexpr Index mid_low = std::max(SumTraits::Lhs::LowDegreeAtCompileTime,
                                       SumTraits::Rhs::LowDegreeAtCompileTime);
    constexpr Index mid_high = std::min(SumTraits::Lhs::DegreeAtCompileTime,
                                        SumTraits::Rhs::DegreeAtCompileTime);
    constexpr Index mid_len = mid_high - mid_low + 1;

    if constexpr (mid_len > 0) {
      s_coeffs.template segment<mid_len>(mid_low -
                                         SumTraits::LowDegreeAtCompileTime) =
          op(l_coeffs.template segment<mid_len>(
                 mid_low - SumTraits::Lhs::LowDegreeAtCompileTime),
             r_coeffs.template segment<mid_len>(
                 mid_low - SumTraits::Rhs::LowDegreeAtCompileTime));
    }
  } else {
    const Index mid_low = std::max(l_low_deg, r_low_deg);
    const Index mid_high = std::min(l_deg, r_deg);
    const Index mid_len = mid_high - mid_low + 1;
    if (mid_len > 0) {
      s_coeffs.segment(mid_low, mid_len) =
          op(l_coeffs.segment(mid_low - l_low_deg, mid_len),
             r_coeffs.segment(mid_low - r_low_deg, mid_len));
    }
  }
  return sum;
}

template <typename DerivedLhs, typename DerivedRhs>
auto operator*(const DensePolyBase<DerivedLhs> &lhs,
               const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_mul_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  using MulTraits = polynomial_mul_trait<DensePolyBase<DerivedLhs>,
                                         DensePolyBase<DerivedRhs>>;
  using Result = typename MulTraits::ResultType;
  Result res(lhs.degree() + rhs.degree(), lhs.low_degree() + rhs.low_degree());

  if constexpr (MulTraits::DegreeAtCompileTime != Eigen::Dynamic &&
                MulTraits::LowDegreeAtCompileTime != Eigen::Dynamic) {
    using Lhs = typename MulTraits::Lhs;
    using Rhs = typename MulTraits::Rhs;
    // using Scalar = typename MulTraits::Scalar;
    constexpr Index tail = Rhs::CoeffsCompileTime;
    constexpr Index head = Lhs::CoeffsCompileTime - 1;
    static_assert(tail != Eigen::Dynamic && tail > 0);
    static_assert(head != Eigen::Dynamic && head >= 0);

    auto &c = res.coeffs();
    c.template tail<tail>() = rhs.coeffs() * lhs[Lhs::DegreeAtCompileTime];
    c.template head<head>() =
        lhs.coeffs().template head<head>() * rhs[Rhs::LowDegreeAtCompileTime];

    constexpr Index mul_rhs_len = Rhs::CoeffsCompileTime - 1;
    for (Index lhs_degree = Lhs::LowDegreeAtCompileTime;
         lhs_degree < Lhs::DegreeAtCompileTime; ++lhs_degree) {
      c.template segment<mul_rhs_len>(lhs_degree - Lhs::LowDegreeAtCompileTime +
                                      1) +=
          rhs.coeffs().template tail<mul_rhs_len>() * lhs[lhs_degree];
    }
  } else {
    const Index tail = rhs.total_coeffs();
    const Index head = lhs.total_coeffs() - 1;
    POLYNOMIALS_ASSERT(tail != Eigen::Dynamic && tail > 0,
                       "Invalid multiplication size");
    POLYNOMIALS_ASSERT(head != Eigen::Dynamic && head >= 0,
                       "Invalid multiplication size");

    auto &c = res.coeffs();
    c.tail(tail) = rhs.coeffs() * lhs[lhs.degree()];
    c.head(head) = rhs.coeffs().head() * lhs[lhs.low_degree()];

    const Index mul_rhs_len = rhs.total_coeffs() - 1;
    for (Index lhs_degree = lhs.low_degree(); lhs_degree < lhs.degree();
         ++lhs_degree) {
      c.segment(lhs_degree - lhs.low_degree() + 1, mul_rhs_len) +=
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

template <typename Derived, char var = 'x'>
std::ostream &operator<<(std::ostream &os, const DensePolyBase<Derived> &poly) {
  using Poly = DensePolyBase<Derived>;
  using Scalar = typename Poly::Scalar;
  const Scalar zero(0);
  const Scalar one(1);
  const Scalar neg_one(-1);

  bool not_first = false;
  bool showpos = os.flags() & std::ios_base::showpos;
  for (Index i = poly.low_degree(); i <= poly.degree(); ++i) {
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
