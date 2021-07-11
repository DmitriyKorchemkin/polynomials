#include "testing.hpp"

TEST_CASE("Dense scalar operator-") {
  DegreeIterator<MinusScalarTest, double, double, double, 0, 3, 5>().test();
  DegreeIterator<MinusScalarTest, double, double, ceres::Jet<double, 4>, 0, 3,
                 5>()
      .test();
  DegreeIterator<MinusScalarTest, double, ceres::Jet<double, 4>, double, 0, 3,
                 5>()
      .test();
  DegreeIterator<MinusScalarTest, double, ceres::Jet<double, 4>,
                 ceres::Jet<double, 4>, 0, 3, 5>()
      .test();
}
