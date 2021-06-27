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

  //	constexpr auto

protected:
  auto &derived() const { return *static_cast<const Derived *>(this); }
  auto &derived() { return *static_cast<Derived *>(this); }
};

template <Index sz> struct LowDegreeValue {
  LowDegreeValue(const Index &new_sz) {
    POLYNOMIALS_ASSERT(sz == new_sz, "Unable to resize from size " +
                                         std::to_string(sz) +
                                         " to size: " + std::to_string(sz));
  }
  constexpr Index low_degree_value() const { return sz; }
  void reset_low_degree(const Index &new_sz) {
    POLYNOMIALS_ASSERT(sz == new_sz, "Unable to resize from size " +
                                         std::to_string(sz) +
                                         " to size: " + std::to_string(sz));
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

constexpr Index max_num_coeffs(const Index DegreeAtCompileTime,
                               const Index LowDegreeAtCompileTime,
                               const Index MaxDegreeAtCompileTime) {
  if (MaxDegreeAtCompileTime == Eigen::Dynamic)
    return Eigen::Dynamic;
  if (LowDegreeAtCompileTime != Eigen::Dynamic)
    return MaxDegreeAtCompileTime - LowDegreeAtCompileTime + 1;

  return MaxDegreeAtCompileTime;
}
constexpr Index num_coeffs_compile_time(const Index DegreeAtCompileTime,
                                        const Index LowDegreeAtCompileTime,
                                        const Index MaxDegreeAtCompileTime) {
  if (DegreeAtCompileTime != Eigen::Dynamic &&
      LowDegreeAtCompileTime != Eigen::Dynamic)
    return DegreeAtCompileTime - LowDegreeAtCompileTime + 1;
  return Eigen::Dynamic;
}

template <typename T, Index DegreeAtCompileTime, Index LowDegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options>
struct DensePoly
    : DensePolyBase<DensePoly<T, DegreeAtCompileTime, LowDegreeAtCompileTime,
                              MaxDegreeAtCompileTime, Options>>,
      LowDegreeValue<LowDegreeAtCompileTime> {
  using Scalar = T;
  static constexpr Index CoeffsAtCompileTime = num_coeffs_compile_time(
      DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  static constexpr Index MaxCoeffsAtCompileTime = max_num_coeffs(
      DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
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
}; // namespace polynomials

namespace Eigen {

template <typename T, Index DegreeAtCompileTime, Index LowDegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options>
class Map<polynomials::DensePoly<T, DegreeAtCompileTime, LowDegreeAtCompileTime,
                                 MaxDegreeAtCompileTime, Options>>
    : public polynomials::DensePolyBase<Map<
          polynomials::DensePoly<T, DegreeAtCompileTime, LowDegreeAtCompileTime,
                                 MaxDegreeAtCompileTime, Options>>>,
      polynomials::LowDegreeValue<LowDegreeAtCompileTime> {
public:
  using Scalar = T;
  static constexpr Index CoeffsAtCompileTime =
      polynomials::num_coeffs_compile_time(
          DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  static constexpr Index MaxCoeffsAtCompileTime = polynomials::max_num_coeffs(
      DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  using Coeffs = Eigen::Map<Eigen::Matrix<Scalar, CoeffsAtCompileTime, 1,
                                          Options, MaxCoeffsAtCompileTime, 1>>;
  using LowCoeffDegree = polynomials::LowDegreeValue<LowDegreeAtCompileTime>;

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

template <typename T, Index DegreeAtCompileTime, Index LowDegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options>
class Map<
    const polynomials::DensePoly<T, DegreeAtCompileTime, LowDegreeAtCompileTime,
                                 MaxDegreeAtCompileTime, Options>>
    : public polynomials::DensePolyBase<Map<const polynomials::DensePoly<
          T, DegreeAtCompileTime, LowDegreeAtCompileTime,
          MaxDegreeAtCompileTime, Options>>>,
      polynomials::LowDegreeValue<LowDegreeAtCompileTime> {
public:
  using Scalar = T;
  static constexpr Index CoeffsAtCompileTime =
      polynomials::num_coeffs_compile_time(
          DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  static constexpr Index MaxCoeffsAtCompileTime = polynomials::max_num_coeffs(
      DegreeAtCompileTime, LowDegreeAtCompileTime, MaxDegreeAtCompileTime);
  using Coeffs =
      Eigen::Map<const Eigen::Matrix<Scalar, CoeffsAtCompileTime, 1, Options,
                                     MaxCoeffsAtCompileTime, 1>>;
  using LowCoeffDegree = polynomials::LowDegreeValue<LowDegreeAtCompileTime>;

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

#endif
