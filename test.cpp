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

template <typename T> using scalar_type_t = typename scalar_type<T>::type;

// doctest::Approx is not templated... what a shame
template <typename T> bool approximately_equal(const T &a, const T &b) {
  using std::abs;
  using std::max;
  using std::sqrt;
  const T eps = sqrt(std::numeric_limits<T>::epsilon());
  return abs(a - b) < eps * (T(1.) + max(abs(a), abs(b)));
}

template <typename T, int N>
bool approximately_equal(const ceres::Jet<T, N> &a, const ceres::Jet<T, N> &b) {
  bool equal = approximately_equal(a.a, b.a);
  for (int i = 0; i < N && equal; ++i)
    equal &= approximately_equal(a.v[i], b.v[i]);
  return equal;
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
  using S = scalar_type_t<T>;
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
    CHECK(approximately_equal(at, real_at));
    CHECK(approximately_equal(mapped_at, real_at));
  }
}

TEST_CASE_TEMPLATE("Jet test", T, ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using qmap = Eigen::Map<const quadric>;
  using R = Random<T>;

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
    CHECK(approximately_equal(at, real_at));
    CHECK(approximately_equal(mapped_at, real_at));
  }
}

TEST_CASE_TEMPLATE("Offset quartic", T, float, double, ceres::Jet<float, 5>,
                   ceres::Jet<double, 7>) {
  using S = scalar_type_t<T>;
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
    CHECK(approximately_equal(at, real_at));
    CHECK(approximately_equal(mapped_at, real_at));
  }
}

TEST_CASE_TEMPLATE("Add same type, same size", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using R = Random<T>;

  R r;

  quadric q1, q2;

  T coeffs1[] = {r(), r(), r()};
  T coeffs2[] = {r(), r(), r()};

  q1.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];
  q2.coeffs() << coeffs2[0], coeffs2[1], coeffs2[2];

  quadric sum = q1 + q2;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = sum(x);
    auto real_at = q1(x) + q2(x);
    CHECK(approximately_equal(at, real_at));
  }
}

TEST_CASE_TEMPLATE("Add same type, mixed size", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using cubic = polynomials::DensePoly<T, 3>;
  using R = Random<T>;

  R r;

  quadric q;
  cubic c;

  T coeffs1[] = {r(), r(), r()};
  T coeffs2[] = {r(), r(), r(), r()};

  q.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];
  c.coeffs() << coeffs2[0], coeffs2[1], coeffs2[2], coeffs2[3];

  cubic sum = q + c;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = sum(x);
    auto real_at = q(x) + c(x);
    CHECK(approximately_equal(at, real_at));
  }
}

TEST_CASE_TEMPLATE("Mul same type, mixed size", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using cubic = polynomials::DensePoly<T, 3>;
  using R = Random<T>;

  R r;

  quadric q;
  cubic c;

  T coeffs1[] = {r(), r(), r()};
  T coeffs2[] = {r(), r(), r(), r()};

  q.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];
  c.coeffs() << coeffs2[0], coeffs2[1], coeffs2[2], coeffs2[3];

  auto mul = q * c;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = mul(x);
    auto real_at = q(x) * c(x);
    CHECK(approximately_equal(at, real_at));
  }
}

TEST_CASE_TEMPLATE("Sub same type, same size", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using R = Random<T>;

  R r;

  quadric q1, q2;

  T coeffs1[] = {r(), r(), r()};
  T coeffs2[] = {r(), r(), r()};

  q1.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];
  q2.coeffs() << coeffs2[0], coeffs2[1], coeffs2[2];

  quadric sum = q1 - q2;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = sum(x);
    auto real_at = q1(x) - q2(x);
    CHECK(approximately_equal(at, real_at));
  }
}

TEST_CASE_TEMPLATE("Sub same type, mixed size", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using cubic = polynomials::DensePoly<T, 3>;
  using R = Random<T>;

  R r;

  quadric q;
  cubic c;

  T coeffs1[] = {r(), r(), r()};
  T coeffs2[] = {r(), r(), r(), r()};

  q.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];
  c.coeffs() << coeffs2[0], coeffs2[1], coeffs2[2], coeffs2[3];

  cubic sum = q - c;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at = sum(x);
    auto real_at = q(x) - c(x);
    CHECK(approximately_equal(at, real_at));
  }
}

TEST_CASE_TEMPLATE("Scalar operations, same type", T, float, double,
                   ceres::Jet<float, 5>, ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using R = Random<T>;

  R r;

  quadric q;

  T coeffs1[] = {r(), r(), r()};
  const T mul = r();

  q.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];

  quadric prod1 = q * mul;
  quadric prod2 = mul * q;
  quadric div = q / mul;
  quadric add1 = q + mul;
  quadric add2 = mul + q;
  quadric sub1 = q - mul;
  quadric sub2 = mul - q;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at1 = prod1(x);
    auto at2 = prod2(x);
    auto at3 = div(x);
    auto at4 = add1(x);
    auto at5 = add2(x);
    auto at6 = sub1(x);
    auto at7 = sub2(x);
    auto real_at = q(x) * mul;
    auto real_at_div = q(x) / mul;
    auto real_at_add = q(x) + mul;
    auto real_at_sub1 = q(x) - mul;
    auto real_at_sub2 = mul - q(x);
    CHECK(approximately_equal(at1, real_at));
    CHECK(approximately_equal(at2, real_at));
    CHECK(approximately_equal(at3, real_at_div));
    CHECK(approximately_equal(at4, real_at_add));
    CHECK(approximately_equal(at5, real_at_add));
    CHECK(approximately_equal(at6, real_at_sub1));
    CHECK(approximately_equal(at7, real_at_sub2));
  }
}

TEST_CASE_TEMPLATE("Scalar operations, jet + scalar", T, ceres::Jet<float, 5>,
                   ceres::Jet<double, 7>) {
  using quadric = polynomials::DensePoly<T, 2>;
  using S = scalar_type_t<T>;
  using R = Random<T>;
  using RS = Random<S>;

  R r;
  RS rs;

  quadric q;

  T coeffs1[] = {r(), r(), r()};
  const S mul = rs();

  q.coeffs() << coeffs1[0], coeffs1[1], coeffs1[2];

  auto prod1 = q * mul;
  auto prod2 = mul * q;
  auto div = q / mul;
  auto add1 = q + mul;
  auto add2 = mul + q;
  auto sub1 = q - mul;
  auto sub2 = mul - q;

  for (int i = 0; i < 10; ++i) {
    const auto x = r();
    auto at1 = prod1(x);
    auto at2 = prod2(x);
    auto at3 = div(x);
    auto at4 = add1(x);
    auto at5 = add2(x);
    auto at6 = sub1(x);
    auto at7 = sub2(x);
    auto real_at = q(x) * mul;
    auto real_at_div = q(x) / mul;
    auto real_at_add = q(x) + mul;
    auto real_at_sub1 = q(x) - mul;
    auto real_at_sub2 = mul - q(x);
    CHECK(approximately_equal(at1, real_at));
    CHECK(approximately_equal(at2, real_at));
    CHECK(approximately_equal(at3, real_at_div));
    CHECK(approximately_equal(at4, real_at_add));
    CHECK(approximately_equal(at5, real_at_add));
    CHECK(approximately_equal(at6, real_at_sub1));
    CHECK(approximately_equal(at7, real_at_sub2));
  }
}

TEST_CASE("Approx") {
  CHECK(approximately_equal(0., 0.));
  CHECK(approximately_equal(0., 1e-9));
  CHECK(!approximately_equal(0., 1e-6));
  CHECK(approximately_equal(0.f, 1e-5f));
  CHECK(!approximately_equal(0.f, 1e-3f));
}

TEST_CASE("static asserts") {
  using T = float;
  using quadric = polynomials::DensePoly<T, 2>;
  using quadric_sum = decltype(quadric() + quadric());
  static_assert(quadric_sum::DegreeAtCompileTime == 2);
  static_assert(quadric_sum::MaxDegreeAtCompileTime == 2);
  static_assert(quadric_sum::LowDegreeAtCompileTime == 0);
}
