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
#include "polynomials/types.hpp"

#include <Eigen/Dense>

namespace Eigen {
namespace internal {
template <typename Scalar_, Index DegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options_>
struct traits<polynomials::DensePoly<Scalar_, DegreeAtCompileTime,
                                     MaxDegreeAtCompileTime, Options_>> {
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};

template <typename Scalar_, Index DegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options_>
struct traits<Map<polynomials::DensePoly<Scalar_, DegreeAtCompileTime,
                                         MaxDegreeAtCompileTime>,
                  Options_>> {

  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};
template <typename Scalar_, Index DegreeAtCompileTime,
          Index MaxDegreeAtCompileTime, int Options_>
struct traits<Map<const polynomials::DensePoly<Scalar_, DegreeAtCompileTime,
                                               MaxDegreeAtCompileTime>,
                  Options_>> {

  static constexpr int Options = Options_;
  using Scalar = Scalar_;
};

} // namespace internal
} // namespace Eigen

namespace polynomials {

template <typename Derived_> struct DensePolyBase {
  using Derived = Derived_;
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  template <typename T>
  using OpType = typename Eigen::ScalarBinaryOpTraits<Scalar, T>::ReturnType;
  static constexpr Index DegreeAtCompileTime = Derived::DegreeAtCompileTime;
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

    return result;
  }

  template <typename T = Scalar> auto df(const T &at) const -> OpType<T> {
    using Res = OpType<T>;
    const auto &C = coeffs();

    const Index n_coeffs = total_coeffs();
    Res result(C[n_coeffs - 1] * T(n_coeffs - 1));

    for (Index i = n_coeffs - 2; i > 0; --i) {
      result *= at;
      result += C[i] * T(i);
    }

    return result;
  }

  template <template <typename> typename Algorithm = QuotientRingMulX>
  auto roots() const {
    return RootFinder<Derived, Algorithm>::complex_roots(*this);
  }

  template <template <typename> typename Algorithm = QuotientRingMulX>
  auto real_roots() const {
    return RootFinder<Derived, Algorithm>::real_roots(*this);
  }

  template <template <typename> typename Algorithm = QuotientRingMulX>
  auto positive_real_roots() const {
    return RootFinder<Derived, Algorithm>::positive_real_roots(*this);
  }

  auto operator[](const Index &i) const { return coeffs()[i]; }
  auto &operator[](const Index &i) { return coeffs()[i]; }

  const auto &coeffs() const { return derived().coeffs(); }
  auto &coeffs() { return derived().coeffs(); }

  constexpr Index degree() const {
    return DegreeAtCompileTime == Dynamic ? derived().degree()
                                          : DegreeAtCompileTime;
  }
  constexpr Index total_coeffs() const {
    return CoeffsCompileTime == Dynamic ? coeffs().size() : CoeffsCompileTime;
  }

  void resize(const Index degree) { derived().resize(degree); }

  template <typename OtherDerived>
  Derived &operator=(const DensePolyBase<OtherDerived> &other);

protected:
  auto &derived() const { return *static_cast<const Derived *>(this); }
  auto &derived() { return *static_cast<Derived *>(this); }
};

template <typename T> struct is_dense_poly : std::false_type {};

template <typename T>
struct is_dense_poly<DensePolyBase<T>> : std::true_type {};

constexpr Index max_num_coeffs(const Index MaxDegreeAtCompileTime) {
  if (MaxDegreeAtCompileTime == Dynamic)
    return Dynamic;
  return MaxDegreeAtCompileTime + 1;
}
constexpr Index num_coeffs_compile_time(const Index DegreeAtCompileTime) {

  if (DegreeAtCompileTime != Dynamic)
    return DegreeAtCompileTime + 1;
  return Dynamic;
}

constexpr Index max_or_dynamic(const Index &a, const Index &b) {
  if (a == Dynamic || b == Dynamic)
    return Dynamic;
  return std::max(a, b);
}
constexpr Index min_or_dynamic(const Index &a, const Index &b) {
  if (a == Dynamic || b == Dynamic)
    return Dynamic;
  return std::min(a, b);
}

template <typename T, Index DegreeAtCompileTime_, Index MaxDegreeAtCompileTime_,
          int Options>
struct DensePoly : DensePolyBase<DensePoly<T, DegreeAtCompileTime_,
                                           MaxDegreeAtCompileTime_, Options>> {
  using Scalar = T;
  using Base = DensePolyBase<
      DensePoly<T, DegreeAtCompileTime_, MaxDegreeAtCompileTime_>>;
  static constexpr Index DegreeAtCompileTime = DegreeAtCompileTime_;
  static constexpr Index MaxDegreeAtCompileTime = MaxDegreeAtCompileTime_;

  static constexpr Index CoeffsCompileTime =
      num_coeffs_compile_time(DegreeAtCompileTime);
  static constexpr Index MaxCoeffsCompileTime =
      max_num_coeffs(MaxDegreeAtCompileTime);
  using Coeffs = Eigen::Matrix<Scalar, CoeffsCompileTime, 1, Options,
                               MaxCoeffsCompileTime, 1>;
  using Base::operator=;

  DensePoly(const Index degree = DegreeAtCompileTime)
      : coeffs_(degree + 1, 1) {}

  template <typename OtherDerived>
  DensePoly(const DensePolyBase<OtherDerived> &other)
      : DensePoly(DegreeAtCompileTime == Dynamic ? other.degree()
                                                 : DegreeAtCompileTime) {
    using Other = DensePolyBase<OtherDerived>;
    static_assert(std::is_same_v<Scalar, typename Other::Scalar>,
                  "Incompatible scalar types for copy ctor");
    static_assert(DegreeAtCompileTime == Dynamic ||
                  Other::DegreeAtCompileTime <= DegreeAtCompileTime);
    if constexpr (MaxDegreeAtCompileTime != Dynamic &&
                  Other::DegreeAtCompileTime == Dynamic) {
      POLYNOMIALS_ASSERT(MaxDegreeAtCompileTime >= other.degree(),
                         "Not enough storage to assign "
                             << other.degree() << "-degree polynomial");
    }

    const Index head = other.total_coeffs();
    const Index tail = Base::total_coeffs() - head;
    if (head) {
      coeffs_.head(head) = other.coeffs();
    }
    if (tail) {
      coeffs_.tail(tail).setZero();
    }
  }

  DensePoly(const DensePoly &) = default;
  DensePoly(DensePoly &&) = default;

  const Coeffs &coeffs() const { return coeffs_; }
  Coeffs &coeffs() { return coeffs_; }

  constexpr Index degree() const {
    return DegreeAtCompileTime == Dynamic ? coeffs_.size() - 1
                                          : DegreeAtCompileTime;
  }

  void resize(const Index degree) { coeffs_.resize(degree + 1, 1); }

  const Scalar *data() const { return coeffs_.data(); }
  Scalar *data() { return coeffs_.data(); }

  DensePoly &operator=(const DensePoly &) = default;
  DensePoly &operator=(DensePoly &&) = default;

protected:
  Coeffs coeffs_;
};
} // namespace polynomials

namespace Eigen {

template <typename T, Index DegreeAtCompileTime_, Index MaxDegreeAtCompileTime_,
          int Options>
class Map<polynomials::DensePoly<T, DegreeAtCompileTime_,
                                 MaxDegreeAtCompileTime_, Options>>
    : public polynomials::DensePolyBase<Map<polynomials::DensePoly<
          T, DegreeAtCompileTime_, MaxDegreeAtCompileTime_, Options>>> {
public:
  using Scalar = T;
  using Mapped =
      typename polynomials::DensePoly<T, DegreeAtCompileTime_,
                                      MaxDegreeAtCompileTime_, Options>;
  using Base = polynomials::DensePolyBase<Map<polynomials::DensePoly<
      T, DegreeAtCompileTime_, MaxDegreeAtCompileTime_, Options>>>;
  static constexpr Index DegreeAtCompileTime = Mapped::DegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Mapped::MaxDegreeAtCompileTime;
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsCompileTime;
  static constexpr Index MaxCoeffsCompileTime = Mapped::MaxCoeffsCompileTime;

  using Base::operator=;
  using Coeffs = Eigen::Map<typename Mapped::Coeffs>;

  Map(T *data, const Index degree = DegreeAtCompileTime)
      : coeffs_(data, degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }
  Coeffs &coeffs() { return coeffs_; }

  constexpr Index degree() const {
    return DegreeAtCompileTime == polynomials::Dynamic ? coeffs_.size() - 1
                                                       : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }
  Scalar *data() { return coeffs_.data(); }

  void resize(const Index degree) { coeffs_.resize(degree + 1); }

  Map &operator=(const Map &map) {
    Base::template operator=<Map>(map);
    return *this;
  }

protected:
  Coeffs coeffs_;
};

template <typename T, Index DegreeAtCompileTime_, Index MaxDegreeAtCompileTime_,
          int Options>
class Map<const polynomials::DensePoly<T, DegreeAtCompileTime_,
                                       MaxDegreeAtCompileTime_, Options>>
    : public polynomials::DensePolyBase<Map<const polynomials::DensePoly<
          T, DegreeAtCompileTime_, MaxDegreeAtCompileTime_, Options>>> {
public:
  using Scalar = T;
  using Base = polynomials::DensePolyBase<Map<const polynomials::DensePoly<
      T, DegreeAtCompileTime_, MaxDegreeAtCompileTime_, Options>>>;
  using Mapped =
      typename polynomials::DensePoly<T, DegreeAtCompileTime_,
                                      MaxDegreeAtCompileTime_, Options>;
  static constexpr Index DegreeAtCompileTime = Mapped::DegreeAtCompileTime;
  static constexpr Index MaxDegreeAtCompileTime =
      Mapped::MaxDegreeAtCompileTime;
  static constexpr Index CoeffsCompileTime = Mapped::CoeffsCompileTime;
  static constexpr Index MaxCoeffsCompileTime = Mapped::MaxCoeffsCompileTime;

  using Coeffs = Eigen::Map<const typename Mapped::Coeffs>;

  Map(const T *data, const Index degree = DegreeAtCompileTime)
      : coeffs_(data, degree + 1, 1) {}

  const Coeffs &coeffs() const { return coeffs_; }

  constexpr Index degree() const {
    return DegreeAtCompileTime == polynomials::Dynamic ? coeffs_.size() - 1
                                                       : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }

protected:
  const Coeffs coeffs_;
};

} // namespace Eigen

#include "src/dense_operators.hpp"
#include "src/dense_scalar_operators.hpp"
#include "src/roots.hpp"
#endif
