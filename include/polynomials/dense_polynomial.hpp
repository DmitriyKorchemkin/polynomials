#ifndef POLYNOMIALS_DENSE_POLYNOMIAL_HPP
#define POLYNOMIALS_DENSE_POLYNOMIAL_HPP

#include "polynomials/assert.hpp"

#include <Eigen/Dense>

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
  static constexpr Index CoeffsCompileTime = Derived::CoeffsAtCompileTime;
  static constexpr Index MaxCoeffsAtCompileTime =
      Derived::MaxCoeffsAtCompileTime;

  template <typename T = Scalar> auto operator()(const T &at) -> OpType<T> {
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

  auto &operator[](const Index &i) const { return coeffs()[i - low_degree()]; }
  auto &operator[](const Index &i) { return coeffs()[i - low_degree()]; }

  auto &coeffs() const { return derived().coeffs(); }
  auto &coeffs() { return derived().coeffs(); }

  constexpr auto low_degree() const { return derived().low_degree(); }
  constexpr Index degree() const { return coeffs().size() + low_degree() - 1; }
  constexpr Index total_coeffs() const { return coeffs().size(); }

protected:
  auto &derived() const { return *static_cast<const Derived *>(this); }
  auto &derived() { return *static_cast<Derived *>(this); }
};

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

  static constexpr Index CoeffsAtCompileTime =
      num_coeffs_compile_time(DegreeAtCompileTime, LowDegreeAtCompileTime);
  static constexpr Index MaxCoeffsAtCompileTime =
      max_num_coeffs(LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  using Coeffs = Eigen::Matrix<Scalar, CoeffsAtCompileTime, 1, Options,
                               MaxCoeffsAtCompileTime, 1>;
  using LowCoeffDegree = LowDegreeValue<LowDegreeAtCompileTime>;

  DensePoly(const Index degree = DegreeAtCompileTime,
            const Index low_degree = LowDegreeAtCompileTime)
      : LowCoeffDegree(low_degree), coeffs_(degree - low_degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }
  Coeffs &coeffs() { return coeffs_; }

  constexpr Index low_degree() const {
    return LowCoeffDegree::low_degree_value();
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
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsAtCompileTime;
  static constexpr Index MaxCoeffsAtCompileTime =
      Mapped::MaxCoeffsAtCompileTime;

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
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsAtCompileTime;
  static constexpr Index MaxCoeffsAtCompileTime =
      Mapped::MaxCoeffsAtCompileTime;

  using Coeffs = Eigen::Map<const typename Mapped::Coeffs>;
  using LowCoeffDegree = typename Mapped::LowCoeffDegree;

  Map(const T *data, const Index degree = DegreeAtCompileTime,
      const Index low_degree = LowDegreeAtCompileTime)
      : LowCoeffDegree(low_degree), coeffs_(data, degree - low_degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }

  constexpr Index low_degree() const {
    return LowCoeffDegree::low_degree_value();
  }

  const Scalar *data() const { return coeffs_.data(); }

protected:
  const Coeffs coeffs_;
};

} // namespace Eigen

namespace polynomials {

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
      s_coeffs.template tail<tail_len>() =
          op.right_op(r_coeffs.template tail<tail_len>());
    } else {
      const Index tail_len = s_deg - r_deg;
      s_coeffs.tail(tail_len) = op.right_op(r_coeffs.tail(tail_len));
    }
  }
  if (s_deg != r_deg) {
    if constexpr (SumTraits::DegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index tail_len =
          SumTraits::DegreeAtCompileTime - SumTraits::Rhs::DegreeAtCompileTime;
      s_coeffs.template tail<tail_len>() =
          op.left_op(l_coeffs.template tail<tail_len>());
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
      s_coeffs.template head<head_len>() =
          op.right_op(r_coeffs.template head<head_len>());
    } else {
      const Index head_len = l_low_deg - s_low_deg;
      s_coeffs.head(head_len) = op.right_op(r_coeffs.head(head_len));
    }
  }
  if (s_low_deg != r_low_deg) {
    if constexpr (SumTraits::LowDegreeAtCompileTime != Eigen::Dynamic) {
      constexpr Index head_len = SumTraits::Rhs::LowDegreeAtCompileTime -
                                 SumTraits::LowDegreeAtCompileTime;
      s_coeffs.template head<head_len>() =
          op.left_op(l_coeffs.template head<head_len>());
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
    s_coeffs.template segment<mid_len>(mid_low) =
        op(l_coeffs.template segment<mid_len>(
               mid_low - SumTraits::Lhs::LowDegreeAtCompileTime),
           r_coeffs.template segment<mid_len>(
               mid_low - SumTraits::Rhs::LowDegreeAtCompileTime));
  } else {
    const Index mid_low = std::max(l_low_deg, r_low_deg);
    const Index mid_high = std::min(l_deg, r_deg);
    const Index mid_len = mid_high - mid_low + 1;
    s_coeffs.segment(mid_low, mid_len) =
        op(l_coeffs.segment(mid_low - l_low_deg, mid_len),
           r_coeffs.segment(mid_low - r_low_deg, mid_len));
  }
  return sum;
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
} // namespace polynomials
#endif
