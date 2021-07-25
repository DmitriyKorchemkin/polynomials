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

#include <Eigen/Dense>
#include <iosfwd>

#include "polynomials/assert.hpp"
#include "polynomials/types.hpp"

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
      (Lhs::DegreeAtCompileTime == Dynamic ||
       Rhs::DegreeAtCompileTime == Dynamic)
          ? Dynamic
          : Lhs::DegreeAtCompileTime + Rhs::DegreeAtCompileTime;

  using ResultType = DensePoly<Scalar, DegreeAtCompileTime>;
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

  using ResultType = DensePoly<Scalar, DegreeAtCompileTime>;

  static constexpr Index CoeffsCompileTime = ResultType::CoeffsCompileTime;
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

template <typename Lhs, typename Rhs, typename Op,
          bool constsize =
              polynomial_sum_trait<Lhs, Rhs>::CoeffsCompileTime != Dynamic>
struct SumLikeOpImpl;

// static-sized version
template <typename Result, typename Lhs, typename Rhs, typename Op,
          bool from_lhs>
struct TailOp;

template <typename Result, typename Lhs, typename Rhs, typename Op,
          bool has_intersection>
struct MidOp;

template <typename Result, typename Lhs, typename Rhs, typename Op>
struct TailOp<Result, Lhs, Rhs, Op, false> {
  using ResultScalar = typename Result::Scalar;
  void operator()(Result &res, const Lhs &lhs, const Rhs &rhs,
                  const Op &op) const {
    (void)lhs;
    static_assert(Rhs::DegreeAtCompileTime == Result::DegreeAtCompileTime);
    constexpr Index max_degree = Rhs::DegreeAtCompileTime;
    constexpr Index min_degree = Lhs::DegreeAtCompileTime;
    constexpr Index tail_len = max_degree - min_degree;
    constexpr Index tail_available = std::min(Rhs::CoeffsCompileTime, tail_len);
    static_assert(tail_available != Dynamic);
    if (tail_available == 0)
      return;

    res.coeffs().template tail<tail_available>() =
        op.right_op(rhs.coeffs())
            .template tail<tail_available>()
            .template cast<ResultScalar>();
  }
};
template <typename Result, typename Lhs, typename Rhs, typename Op>
struct TailOp<Result, Lhs, Rhs, Op, true> {
  using ResultScalar = typename Result::Scalar;
  void operator()(Result &res, const Lhs &lhs, const Rhs &rhs,
                  const Op &op) const {
    (void)rhs;
    static_assert(Lhs::DegreeAtCompileTime == Result::DegreeAtCompileTime);
    constexpr Index max_degree = Lhs::DegreeAtCompileTime;
    constexpr Index min_degree = Rhs::DegreeAtCompileTime;
    constexpr Index tail_len = max_degree - min_degree;
    constexpr Index tail_available = std::min(Lhs::CoeffsCompileTime, tail_len);
    static_assert(tail_available != Dynamic);
    if (tail_available == 0)
      return;

    res.coeffs().template tail<tail_available>() =
        op.left_op(lhs.coeffs())
            .template tail<tail_available>()
            .template cast<ResultScalar>();
  }
};

template <typename Result, typename Lhs, typename Rhs, typename Op>
struct MidOp<Result, Lhs, Rhs, Op, true> {
  static constexpr Index min_degree =
      std::min(Lhs::DegreeAtCompileTime, Rhs::DegreeAtCompileTime);
  void operator()(Result &res, const Lhs &lhs, const Rhs &rhs,
                  const Op &op) const {
    static_assert(min_degree != Dynamic);

    constexpr Index len = min_degree + 1;
    static_assert(len >= 0);
    if constexpr (len == 0)
      return;

    res.coeffs().template head<len>() = op(lhs.coeffs().template head<len>(),
                                           rhs.coeffs().template head<len>());
  }
};

template <typename DerivedLhs, typename DerivedRhs, typename Op>
struct SumLikeOpImpl<DensePolyBase<DerivedLhs>, DensePolyBase<DerivedRhs>, Op,
                     true> : private Op {
  using Lhs = DensePolyBase<DerivedLhs>;
  using Rhs = DensePolyBase<DerivedRhs>;
  using SumTraits = polynomial_sum_trait<Lhs, Rhs>;
  using Result = typename SumTraits::ResultType;
  using ResultScalar = typename Result::Scalar;
  static constexpr Index min_degree =
      std::min(Lhs::DegreeAtCompileTime, Rhs::DegreeAtCompileTime);

  SumLikeOpImpl(const Lhs &lhs, const Rhs &rhs) : lhs(lhs), rhs(rhs) {}

  Result operator()() const {
    Result res;
    static_assert(Result::DegreeAtCompileTime != Dynamic);
    MidOp<Result, Lhs, Rhs, Op, true>()(res, lhs, rhs, *this);
    TailOp<Result, Lhs, Rhs, Op, min_degree == Rhs::DegreeAtCompileTime>()(
        res, lhs, rhs, *this);
    return res;
  }

  const Lhs &lhs;
  const Rhs &rhs;
};

// dynamic-sized version
template <typename Result, typename Lhs, typename Rhs, typename Op>
struct TailOpDynamic {
  using ResultScalar = typename Result::Scalar;
  void operator()(Result &res, const Lhs &lhs, const Rhs &rhs,
                  const Op &op) const {
    const auto rhs_degree = rhs.degree();
    const auto lhs_degree = lhs.degree();
    const auto res_degree = res.degree();
    if (rhs_degree == res_degree) {
      const auto max_degree = rhs_degree;
      const auto min_degree = lhs_degree;
      const auto tail_len = max_degree - min_degree;
      const auto tail_available = std::min(rhs.total_coeffs(), tail_len);
      if (tail_available == 0)
        return;

      res.coeffs().tail(tail_available) = op.right_op(rhs.coeffs())
                                              .tail(tail_available)
                                              .template cast<ResultScalar>();
    } else {
      POLYNOMIALS_ASSERT(lhs_degree == res_degree,
                         "Invalid degree: " << lhs_degree << ' ' << rhs_degree
                                            << ' ' << res_degree);
      const auto max_degree = lhs_degree;
      const auto min_degree = rhs_degree;
      const auto tail_len = max_degree - min_degree;
      const auto tail_available = std::min(lhs.total_coeffs(), tail_len);
      if (tail_available == 0)
        return;

      res.coeffs().tail(tail_available) = op.left_op(lhs.coeffs())
                                              .tail(tail_available)
                                              .template cast<ResultScalar>();
    }
  }
};

template <typename Result, typename Lhs, typename Rhs, typename Op>
struct MidOpDynamic {

  using ResultScalar = typename Result::Scalar;
  void operator()(Result &res, const Lhs &lhs, const Rhs &rhs,
                  const Op &op) const {
    const auto rhs_degree = rhs.degree();
    const auto lhs_degree = lhs.degree();
    const auto min_degree = std::min(lhs_degree, rhs_degree);

    const auto len = min_degree + 1;
    POLYNOMIALS_ASSERT(len >= 0, "Invalid mid length");
    if (len == 0)
      return;

    res.coeffs().head(len) = op(lhs.coeffs().head(len), rhs.coeffs().head(len));
  }
};

// dynamic-sized version
template <typename DerivedLhs, typename DerivedRhs, typename Op>
struct SumLikeOpImpl<DensePolyBase<DerivedLhs>, DensePolyBase<DerivedRhs>, Op,
                     false> : private Op {
  using Lhs = DensePolyBase<DerivedLhs>;
  using Rhs = DensePolyBase<DerivedRhs>;
  using SumTraits = polynomial_sum_trait<Lhs, Rhs>;
  using Result = typename SumTraits::ResultType;
  using ResultScalar = typename Result::Scalar;

  SumLikeOpImpl(const Lhs &lhs, const Rhs &rhs) : lhs(lhs), rhs(rhs) {}

  Result operator()() const {
    Result res(std::max(lhs.degree(), rhs.degree()));
    MidOpDynamic<Result, Lhs, Rhs, Op>()(res, lhs, rhs, *this);
    TailOpDynamic<Result, Lhs, Rhs, Op>()(res, lhs, rhs, *this);
    return res;
  }

  const Lhs &lhs;
  const Rhs &rhs;
};

template <typename DerivedLhs, typename DerivedRhs, typename Op>
auto sum_like_op(const DensePolyBase<DerivedLhs> &lhs,
                 const DensePolyBase<DerivedRhs> &rhs) ->
    typename polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                  DensePolyBase<DerivedRhs>>::ResultType {
  using SumTraits = polynomial_sum_trait<DensePolyBase<DerivedLhs>,
                                         DensePolyBase<DerivedRhs>>;
  using Impl =
      SumLikeOpImpl<typename SumTraits::Lhs, typename SumTraits::Rhs, Op>;
  return Impl(lhs, rhs)();
}

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

template <typename T> struct is_map : std::false_type {};

template <typename T> struct is_map<Eigen::Map<T>> : std::true_type {};

template <typename T> constexpr bool is_map_v = is_map<T>::value;

template <typename T> struct is_const_map : std::false_type {};

template <typename T>
struct is_const_map<Eigen::Map<const T>> : std::true_type {};

template <typename T> constexpr bool is_const_map_v = is_const_map<T>::value;

template <typename Lhs, typename Rhs> constexpr bool potentially_assignable() {
  if (!std::is_same_v<typename Lhs::Scalar, typename Rhs::Scalar>)
    return false;
  if (Lhs::MaxDegreeAtCompileTime != Dynamic &&
      Lhs::MaxDegreeAtCompileTime < Rhs::MaxDegreeAtCompileTime)
    return false;
  if (is_const_map_v<Lhs>)
    return false;
  return true;
}

template <typename Lhs, typename Rhs>
constexpr bool potentially_assignable_v = potentially_assignable<Lhs, Rhs>();

template <typename Lhs, typename Rhs,
          bool Static = Lhs::DegreeAtCompileTime !=
                        Dynamic &&Rhs::DegreeAtCompileTime != Dynamic>
struct AssignOp;

template <typename Lhs, typename Rhs> struct AssignOp<Lhs, Rhs, true> {
  Lhs &operator()(Lhs &lhs, const Rhs &rhs) const {
    static_assert(Lhs::DegreeAtCompileTime >= Rhs::DegreeAtCompileTime,
                  "Cannot assign polynomial with larger degree to polynomial "
                  "with smaller");
    constexpr Index head = Rhs::CoeffsCompileTime;
    constexpr Index tail = Lhs::CoeffsCompileTime - Rhs::CoeffsCompileTime;
    if constexpr (head >= 0) {
      lhs.coeffs().template head<head>() = rhs.coeffs();
    }
    if constexpr (tail >= 0) {
      lhs.coeffs().template tail<tail>().setZero();
    }
    return lhs;
  }
};

template <typename Lhs, typename Rhs> struct AssignOp<Lhs, Rhs, false> {
  Lhs &operator()(Lhs &lhs, const Rhs &rhs) const {
    static_assert(Lhs::DegreeAtCompileTime == Dynamic ||
                  Rhs::DegreeAtCompileTime <= Lhs::DegreeAtCompileTime);
    if constexpr (Lhs::MaxDegreeAtCompileTime != Dynamic &&
                  Rhs::DegreeAtCompileTime == Dynamic) {
      POLYNOMIALS_ASSERT(Lhs::MaxDegreeAtCompileTime >= rhs.degree(),
                         "Not enough storage to assign "
                             << rhs.degree() << "-degree polynomial");
    }
    if constexpr (is_map_v<typename Lhs::Derived>) {
      POLYNOMIALS_ASSERT(lhs.degree() >= rhs.degree(),
                         "Eigen::Map is not resizeable, but need to assign "
                             << rhs.degree() << "-degree polynomial to "
                             << lhs.degree() << " map");

    } else {

      lhs.resize(Lhs::DegreeAtCompileTime == Dynamic
                     ? rhs.degree()
                     : Lhs::DegreeAtCompileTime);
    }
    const Index head = rhs.total_coeffs();
    const Index tail = lhs.total_coeffs() - head;
    if (head) {
      lhs.coeffs().head(head) = rhs.coeffs();
    }
    if (tail) {
      lhs.coeffs().tail(tail).setZero();
    }
    return lhs;
  }
};

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
