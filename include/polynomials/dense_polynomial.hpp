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
#ifndef POLYNOMIALS_DENSE_POLYNOMIAL_HPP
#define POLYNOMIALS_DENSE_POLYNOMIAL_HPP

#include "polynomials/assert.hpp"

#include <Eigen/Dense>

#include <iostream>

namespace polynomials {

using Index = Eigen::Index;
template <typename T, Index DegreeAtCompileTime,
          Index LowDegreeAtCompileTime = 0,
          Index MaxDegreeAtCompileTime = DegreeAtCompileTime, int Options = 0>
struct DensePoly;
} // namespace polynomials

namespace Eigen {
namespace internal {
template <typename Scalar_, Index DegreeAtCompileTime,
          Index LowDegreeAtCompileTime, Index MaxDegreeAtCompileTime,
          int Options_>
struct traits<
    polynomials::DensePoly<Scalar_, DegreeAtCompileTime, LowDegreeAtCompileTime,
                           MaxDegreeAtCompileTime, Options_>> {
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};

template <typename Scalar_, Index DegreeAtCompileTime,
          Index LowDegreeAtCompileTime, Index MaxDegreeAtCompileTime,
          int Options_>
struct traits<
    Map<polynomials::DensePoly<Scalar_, DegreeAtCompileTime,
                               LowDegreeAtCompileTime, MaxDegreeAtCompileTime>,
        Options_>> {

  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};
template <typename Scalar_, Index DegreeAtCompileTime,
          Index LowDegreeAtCompileTime, Index MaxDegreeAtCompileTime,
          int Options_>
struct traits<Map<const polynomials::DensePoly<Scalar_, DegreeAtCompileTime,
                                               LowDegreeAtCompileTime,
                                               MaxDegreeAtCompileTime>,
                  Options_>> {

  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};

} // namespace internal
} // namespace Eigen

namespace polynomials {

template <typename Derived> struct DensePolyBase {
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  template <typename T>
  using OpType = typename Eigen::ScalarBinaryOpTraits<Scalar, T>::ReturnType;
  static constexpr Index DegreeAtCompileTime = Derived::DegreeAtCompileTime;
  static constexpr Index LowDegreeAtCompileTime =
      Derived::LowDegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Derived::MaxDegreeAtCompileTime;
  static constexpr Index CoeffsCompileTime = Derived::CoeffsCompileTime;
  static constexpr Index MaxCoeffsCompileTime = Derived::MaxCoeffsCompileTime;

  template <typename T = Scalar>
  auto operator()(const T &at) const -> OpType<T> {
    using Res = OpType<T>;
    const auto &C = coeffs();

    const Index n_coeffs = total_coeffs();
    Res result(C[n_coeffs - 1]);

    for (Index i = n_coeffs - 2; i >= 0; --i) {
      result *= at;
      result += C[i];
    }

    const Index low_coeff = low_degree();
    for (Index i = 0; i < low_coeff; ++i) {
      result *= at;
    }
    return result;
  }

  auto operator[](const Index &i) const { return coeffs()[i - low_degree()]; }
  auto &operator[](const Index &i) { return coeffs()[i - low_degree()]; }

  const auto &coeffs() const { return derived().coeffs(); }
  auto &coeffs() { return derived().coeffs(); }

  constexpr auto low_degree() const {
    return LowDegreeAtCompileTime == Eigen::Dynamic ? derived().low_degree()
                                                    : LowDegreeAtCompileTime;
  }
  constexpr Index degree() const {
    return DegreeAtCompileTime == Eigen::Dynamic ? derived().degree()
                                                 : DegreeAtCompileTime;
  }
  constexpr Index total_coeffs() const {
    return CoeffsCompileTime == Eigen::Dynamic ? coeffs().size()
                                               : CoeffsCompileTime;
  }

protected:
  auto &derived() const { return *static_cast<const Derived *>(this); }
  auto &derived() { return *static_cast<Derived *>(this); }
};

template <typename T> struct is_dense_poly : std::false_type {};

template <typename T>
struct is_dense_poly<DensePolyBase<T>> : std::true_type {};

template <Index sz> struct LowDegreeValue {
  LowDegreeValue(const Index &new_sz) {
    (void)new_sz;
    POLYNOMIALS_ASSERT(sz == new_sz, "Unable to resize from size " +
                                         std::to_string(sz) +
                                         " to size: " + std::to_string(new_sz));
  }
  constexpr Index low_degree_value() const { return sz; }
  void reset_low_degree(const Index &new_sz) {
    (void)new_sz;
    POLYNOMIALS_ASSERT(sz == new_sz, "Unable to resize from size " +
                                         std::to_string(sz) +
                                         " to size: " + std::to_string(new_sz));
  }
};

template <> struct LowDegreeValue<Eigen::Dynamic> {
  LowDegreeValue(const Index &sz) : sz(sz) {}

  Index low_degree_value() const { return sz; }
  void reset_low_degree(const Index &new_sz) { sz = new_sz; }

protected:
  Index sz;
};

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

constexpr Index max_num_coeffs(const Index LowDegreeAtCompileTime,
                               const Index MaxDegreeAtCompileTime) {
  if (MaxDegreeAtCompileTime == Eigen::Dynamic)
    return Eigen::Dynamic;
  if (LowDegreeAtCompileTime != Eigen::Dynamic)
    return MaxDegreeAtCompileTime - LowDegreeAtCompileTime + 1;

  return MaxDegreeAtCompileTime;
}
constexpr Index num_coeffs_compile_time(const Index DegreeAtCompileTime,
                                        const Index LowDegreeAtCompileTime) {

  if (DegreeAtCompileTime != Eigen::Dynamic &&
      LowDegreeAtCompileTime != Eigen::Dynamic)
    return DegreeAtCompileTime - LowDegreeAtCompileTime + 1;
  return Eigen::Dynamic;
}

constexpr Index max_or_dynamic(const Index &a, const Index &b) {
  if (a == Eigen::Dynamic || b == Eigen::Dynamic)
    return Eigen::Dynamic;
  return std::max(a, b);
}
constexpr Index min_or_dynamic(const Index &a, const Index &b) {
  if (a == Eigen::Dynamic || b == Eigen::Dynamic)
    return Eigen::Dynamic;
  return std::min(a, b);
}

template <typename T, Index DegreeAtCompileTime_, Index LowDegreeAtCompileTime_,
          Index MaxDegreeAtCompileTime_, int Options>
struct DensePoly
    : DensePolyBase<DensePoly<T, DegreeAtCompileTime_, LowDegreeAtCompileTime_,
                              MaxDegreeAtCompileTime_, Options>>,
      LowDegreeValue<LowDegreeAtCompileTime_> {
  using Scalar = T;
  static constexpr Index DegreeAtCompileTime = DegreeAtCompileTime_;
  static constexpr Index LowDegreeAtCompileTime = LowDegreeAtCompileTime_;
  static constexpr Index MaxDegreeAtCompileTime = MaxDegreeAtCompileTime_;

  static constexpr Index CoeffsCompileTime =
      num_coeffs_compile_time(DegreeAtCompileTime, LowDegreeAtCompileTime);
  static constexpr Index MaxCoeffsCompileTime =
      max_num_coeffs(LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  using Coeffs = Eigen::Matrix<Scalar, CoeffsCompileTime, 1, Options,
                               MaxCoeffsCompileTime, 1>;
  using LowCoeffDegree = LowDegreeValue<LowDegreeAtCompileTime>;

  DensePoly(const Index degree = DegreeAtCompileTime,
            const Index low_degree = LowDegreeAtCompileTime)
      : LowCoeffDegree(low_degree), coeffs_(degree - low_degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }
  Coeffs &coeffs() { return coeffs_; }

  constexpr Index low_degree() const {
    return LowCoeffDegree::low_degree_value();
  }

  constexpr Index degree() const {
    return DegreeAtCompileTime == Eigen::Dynamic ? low_degree() + coeffs_.size()
                                                 : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }
  Scalar *data() { return coeffs_.data(); }

protected:
  Coeffs coeffs_;
};
} // namespace polynomials

namespace Eigen {

template <typename T, Index DegreeAtCompileTime_, Index LowDegreeAtCompileTime_,
          Index MaxDegreeAtCompileTime_, int Options>
class Map<
    polynomials::DensePoly<T, DegreeAtCompileTime_, LowDegreeAtCompileTime_,
                           MaxDegreeAtCompileTime_, Options>>
    : public polynomials::DensePolyBase<Map<polynomials::DensePoly<
          T, DegreeAtCompileTime_, LowDegreeAtCompileTime_,
          MaxDegreeAtCompileTime_, Options>>>,
      polynomials::LowDegreeValue<LowDegreeAtCompileTime_> {
public:
  using Scalar = T;
  using Mapped =
      typename polynomials::DensePoly<T, DegreeAtCompileTime_,
                                      LowDegreeAtCompileTime_,
                                      MaxDegreeAtCompileTime_, Options>;
  static constexpr Index DegreeAtCompileTime = Mapped::DegreeAtCompileTime;
  static constexpr Index LowDegreeAtCompileTime =
      Mapped::LowDegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Mapped::MaxDegreeAtCompileTime;
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsCompileTime;
  static constexpr Index MaxCoeffsCompileTime = Mapped::MaxCoeffsCompileTime;

  using Coeffs = Eigen::Map<typename Mapped::Coeffs>;
  using LowCoeffDegree = typename Mapped::LowCoeffDegree;

  Map(T *data, const Index degree = DegreeAtCompileTime,
      const Index low_degree = LowDegreeAtCompileTime)
      : LowCoeffDegree(low_degree), coeffs_(data, degree - low_degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }
  Coeffs &coeffs() { return coeffs_; }

  constexpr Index low_degree() const {
    return LowCoeffDegree::low_degree_value();
  }

  constexpr Index degree() const {
    return DegreeAtCompileTime == Eigen::Dynamic ? low_degree() + coeffs_.size()
                                                 : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }
  Scalar *data() { return coeffs_.data(); }

protected:
  Coeffs coeffs_;
};

template <typename T, Index DegreeAtCompileTime_, Index LowDegreeAtCompileTime_,
          Index MaxDegreeAtCompileTime_, int Options>
class Map<const polynomials::DensePoly<T, DegreeAtCompileTime_,
                                       LowDegreeAtCompileTime_,
                                       MaxDegreeAtCompileTime_, Options>>
    : public polynomials::DensePolyBase<Map<const polynomials::DensePoly<
          T, DegreeAtCompileTime_, LowDegreeAtCompileTime_,
          MaxDegreeAtCompileTime_, Options>>>,
      polynomials::LowDegreeValue<LowDegreeAtCompileTime_> {
public:
  using Scalar = T;
  using Mapped =
      typename polynomials::DensePoly<T, DegreeAtCompileTime_,
                                      LowDegreeAtCompileTime_,
                                      MaxDegreeAtCompileTime_, Options>;
  static constexpr Index DegreeAtCompileTime = Mapped::DegreeAtCompileTime;
  static constexpr Index LowDegreeAtCompileTime =
      Mapped::LowDegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Mapped::MaxDegreeAtCompileTime;
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsCompileTime;
  static constexpr Index MaxCoeffsCompileTime = Mapped::MaxCoeffsCompileTime;

  using Coeffs = Eigen::Map<const typename Mapped::Coeffs>;
  using LowCoeffDegree = typename Mapped::LowCoeffDegree;

  Map(const T *data, const Index degree = DegreeAtCompileTime,
      const Index low_degree = LowDegreeAtCompileTime)
      : LowCoeffDegree(low_degree), coeffs_(data, degree - low_degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }

  constexpr Index low_degree() const {
    return LowCoeffDegree::low_degree_value();
  }

  constexpr Index degree() const {
    return DegreeAtCompileTime == Eigen::Dynamic ? low_degree() + coeffs_.size()
                                                 : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }

protected:
  const Coeffs coeffs_;
};

} // namespace Eigen

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

template <typename Poly, typename Scalar> struct polynomial_scalar_op_trait;

template <typename Derived, typename Rhs>
struct polynomial_scalar_op_trait<DensePolyBase<Derived>, Rhs> {
  using Poly = DensePolyBase<Derived>;
  using OtherScalar = Rhs;
  using Scalar = typename Eigen::ScalarBinaryOpTraits<typename Poly::Scalar,
                                                      OtherScalar>::ReturnType;
  static constexpr Index DegreeAtCompileTime = Poly::DegreeAtCompileTime;
  static constexpr Index LowDegreeAtCompileTime = Poly::LowDegreeAtCompileTime;

  using ResultType =
      DensePoly<Scalar, DegreeAtCompileTime, LowDegreeAtCompileTime>;
  using ResultTypeWithIntercept = DensePoly<Scalar, DegreeAtCompileTime, 0>;
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

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator+(const DensePolyBase<Derived> &lhs, const OtherScalar &rhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultTypeWithIntercept {
  using OpTraits =
      polynomial_scalar_op_trait<DensePolyBase<Derived>, OtherScalar>;
  using ResultBase = typename OpTraits::ResultType;
  using Result = typename OpTraits::ResultTypeWithIntercept;

  const auto deg = lhs.degree();
  Result res(deg);
  if constexpr (OpTraits::Poly::CoeffsCompileTime != Eigen::Dynamic) {
    constexpr Index tail_size = OpTraits::Poly::CoeffsCompileTime;
    constexpr Index deg = OpTraits::Poly::DegreeAtCompileTime;
    constexpr Index head_size = deg + 1 - tail_size;
    res.coeffs().template tail<tail_size>() =
        lhs.coeffs().template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().template head<head_size>().setZero();
    }
  } else {
    const Index tail_size = lhs.total_coeffs();
    const Index head_size = deg + 1 - tail_size;
    res.coeffs().tail(tail_size) =
        lhs.coeffs().template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().head(head_size).setZero();
    }
  }
  res.coeffs()[0] += rhs;
  return res;
}

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator+(const OtherScalar &rhs, const DensePolyBase<Derived> &lhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultTypeWithIntercept {
  return lhs + rhs;
}

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator-(const DensePolyBase<Derived> &lhs, const OtherScalar &rhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultTypeWithIntercept {
  using OpTraits =
      polynomial_scalar_op_trait<DensePolyBase<Derived>, OtherScalar>;
  using ResultBase = typename OpTraits::ResultType;
  using Result = typename OpTraits::ResultTypeWithIntercept;

  const auto deg = lhs.degree();
  Result res(deg);
  if constexpr (OpTraits::Poly::CoeffsCompileTime != Eigen::Dynamic) {
    constexpr Index tail_size = OpTraits::Poly::CoeffsCompileTime;
    constexpr Index deg = OpTraits::Poly::DegreeAtCompileTime;
    constexpr Index head_size = deg + 1 - tail_size;
    res.coeffs().template tail<tail_size>() =
        lhs.coeffs().template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().template head<head_size>().setZero();
    }
  } else {
    const Index tail_size = lhs.total_coeffs();
    const Index head_size = deg + 1 - tail_size;
    res.coeffs().tail(tail_size) =
        lhs.coeffs().template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().head(head_size).setZero();
    }
  }
  res.coeffs()[0] -= rhs;
  return res;
}

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator-(const OtherScalar &rhs, const DensePolyBase<Derived> &lhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultTypeWithIntercept {
  using OpTraits =
      polynomial_scalar_op_trait<DensePolyBase<Derived>, OtherScalar>;
  using ResultBase = typename OpTraits::ResultType;
  using Result = typename OpTraits::ResultTypeWithIntercept;

  const auto deg = lhs.degree();
  Result res(deg);
  if constexpr (OpTraits::Poly::CoeffsCompileTime != Eigen::Dynamic) {
    constexpr Index tail_size = OpTraits::Poly::CoeffsCompileTime;
    constexpr Index deg = OpTraits::Poly::DegreeAtCompileTime;
    constexpr Index head_size = deg + 1 - tail_size;
    res.coeffs().template tail<tail_size>() =
        (-lhs.coeffs()).template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().template head<head_size>().setZero();
    }
  } else {
    const Index tail_size = lhs.total_coeffs();
    const Index head_size = deg + 1 - tail_size;
    res.coeffs().tail(tail_size) =
        (-lhs.coeffs()).template cast<typename ResultBase::Scalar>();
    if (head_size) {
      res.coeffs().head(head_size).setZero();
    }
  }
  res.coeffs()[0] += rhs;
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

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator*(const DensePolyBase<Derived> &lhs, const OtherScalar &rhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultType {
  using OpTraits =
      polynomial_scalar_op_trait<DensePolyBase<Derived>, OtherScalar>;
  using Result = typename OpTraits::ResultType;

  const auto deg = lhs.degree();
  const auto low_deg = lhs.low_degree();
  Result res(deg, low_deg);
  res.coeffs() = lhs.coeffs() * rhs;
  return res;
}

template <typename Derived, typename OtherScalar,
          std::enable_if_t<
              !std::is_base_of_v<DensePolyBase<OtherScalar>, OtherScalar>,
              bool> = true>
auto operator*(const OtherScalar &rhs, const DensePolyBase<Derived> &lhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultType {
  return lhs * rhs;
}

template <typename Derived, typename OtherScalar>
auto operator/(const DensePolyBase<Derived> &lhs, const OtherScalar &rhs) ->
    typename polynomial_scalar_op_trait<DensePolyBase<Derived>,
                                        OtherScalar>::ResultType {
  using OpTraits =
      polynomial_scalar_op_trait<DensePolyBase<Derived>, OtherScalar>;
  using Result = typename OpTraits::ResultType;

  const auto deg = lhs.degree();
  const auto low_deg = lhs.low_degree();
  Result res(deg, low_deg);
  res.coeffs() = lhs.coeffs() / rhs;
  return res;
}

} // namespace polynomials
#endif
