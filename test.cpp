#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "polynomials/dense_polynomial.hpp"

#include <ceres/jet.h>
#include <iostream>
#include <random>

template <typename T> struct scalar_type { using type = T; };

template <typename T, int n> struct scalar_type<ceres::Jet<T, n>> {
  using type = T;
};

template <typename T> double norm_diff(const T &A, const T &B) {
  return std::abs(A - B);
}

template <typename T, int N>
double norm_diff(const ceres::Jet<T, N> &A, const ceres::Jet<T, N> &B) {
  auto diff = A - B;
  return std::sqrt(diff.a * diff.a + diff.v.squaredNorm());
}

std::mt19937 rng(1337);

template <typename T> struct Random {
  T operator()() const {
    std::uniform_real_distribution<T> runif(T(-2.), T(2.));
    return runif(rng);
  }
};
template <typename S, int N> struct Random<ceres::Jet<S, N>> {
  using T = ceres::Jet<S, N>;

  T operator()() const {
    std::uniform_real_distribution<S> runif(S(-2.), S(2.));
    T res;
    res.a = runif(rng);
    for (int i = 0; i < N; ++i)
      res.v[i] = runif(rng);
    return res;
  }
};

TEST_CASE_TEMPLATE("Static quadric", T, float, double, ceres::Jet<float, 5>,
                   ceres::Jet<double, 7>) {
  using S = typename scalar_type<T>::type;
  using quadric = polynomials::DensePoly<T, 2>;
  using qmap = Eigen::Map<const quadric>;
  using R = Random<T>;
  using RS = Random<S>;

  R r;
  RS rs;
  quadric Q(2, 0);
  qmap QM(Q.data());

  T coeffs[] = {r(), r(), r()};
  Q.coeffs() << coeffs[0], coeffs[1], coeffs[2];

  for (int i = 0; i < 10; ++i) {
    const auto x = rs();
    auto at = Q(x);
    auto mapped_at = QM(x);
    auto real_at = (x * coeffs[2] + coeffs[1]) * x + coeffs[0];
    CHECK(norm_diff(at, real_at) == doctest::Approx(0.));
    CHECK(norm_diff(mapped_at, real_at) == doctest::Approx(0.));
  }
}

TEST_CASE_TEMPLATE("Jet test", T, ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using S = typename scalar_type<T>::type;
  using quadric = polynomials::DensePoly<T, 2>;
  using qmap = Eigen::Map<const quadric>;
  using R = Random<T>;
  using RS = Random<S>;

  R r;
  quadric Q(2, 0);
  qmap QM(Q.data());

  T coeffs[] = {r(), r(), r()};
  Q.coeffs() << coeffs[0], coeffs[1], coeffs[2];

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = Q(x);
    auto mapped_at = QM(x);
    auto real_at = (x * coeffs[2] + coeffs[1]) * x + coeffs[0];
    CHECK(norm_diff(at, real_at) == doctest::Approx(0.));
    CHECK(norm_diff(mapped_at, real_at) == doctest::Approx(0.));
  }
}

TEST_CASE_TEMPLATE("Offset quartic", T, float, double, ceres::Jet<float, 5>,
                   ceres::Jet<double, 7>) {
  using S = typename scalar_type<T>::type;
  using quartic = polynomials::DensePoly<T, 4, 2>;
  using qmap = Eigen::Map<const quartic>;
  using R = Random<T>;
  using RS = Random<S>;

  R r;
  RS rs;
  quartic Q;
  qmap QM(Q.data());

  T coeffs[] = {r(), r(), r()};
  Q.coeffs() << coeffs[0], coeffs[1], coeffs[2];

  for (int i = 0; i < 10; ++i) {
    const auto x = rs();
    auto at = Q(x);
    auto mapped_at = QM(x);
    auto real_at = ((x * coeffs[2] + coeffs[1]) * x + coeffs[0]) * x * x;
    CHECK(norm_diff(at, real_at) == doctest::Approx(0.));
    CHECK(norm_diff(mapped_at, real_at) == doctest::Approx(0.));
  }
}
