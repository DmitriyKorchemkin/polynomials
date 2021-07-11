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

#ifndef POLYNOMIALS_DENSE_SCALAR_OPERATORS_HPP
#define POLYNOMIALS_DENSE_SCALAR_OPERATORS_HPP

namespace polynomials {

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
