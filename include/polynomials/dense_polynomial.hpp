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
    return LowDegreeAtCompileTime == Dynamic ? derived().low_degree()
                                             : LowDegreeAtCompileTime;
  }
  constexpr Index degree() const {
    return DegreeAtCompileTime == Dynamic ? derived().degree()
                                          : DegreeAtCompileTime;
  }
  constexpr Index total_coeffs() const {
    return CoeffsCompileTime == Dynamic ? coeffs().size() : CoeffsCompileTime;
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

template <> struct LowDegreeValue<Dynamic> {
  LowDegreeValue(const Index &sz) : sz(sz) {}

  Index low_degree_value() const { return sz; }
  void reset_low_degree(const Index &new_sz) { sz = new_sz; }

protected:
  Index sz;
};

constexpr Index max_num_coeffs(const Index LowDegreeAtCompileTime,
                               const Index MaxDegreeAtCompileTime) {
  if (MaxDegreeAtCompileTime == Dynamic)
    return Dynamic;
  if (LowDegreeAtCompileTime != Dynamic)
    return MaxDegreeAtCompileTime - LowDegreeAtCompileTime + 1;

  return MaxDegreeAtCompileTime + 1;
}
constexpr Index num_coeffs_compile_time(const Index DegreeAtCompileTime,
                                        const Index LowDegreeAtCompileTime) {

  if (DegreeAtCompileTime != Dynamic && LowDegreeAtCompileTime != Dynamic)
    return DegreeAtCompileTime - LowDegreeAtCompileTime + 1;
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
    return DegreeAtCompileTime == Dynamic ? low_degree() + coeffs_.size() - 1
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
    return DegreeAtCompileTime == polynomials::Dynamic
               ? low_degree() + coeffs_.size() - 1
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
    return DegreeAtCompileTime == polynomials::Dynamic
               ? low_degree() + coeffs_.size() - 1
               : DegreeAtCompileTime;
  }

  const Scalar *data() const { return coeffs_.data(); }

protected:
  const Coeffs coeffs_;
};

} // namespace Eigen

#include "src/dense_operators.hpp"
#include "src/dense_scalar_operators.hpp"
#endif
