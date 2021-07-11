#include "testing.hpp"

TEST_CASE("Dense scalar operator+") {
  DegreeIterator<PlusScalarTest, double, double, double, 0, 3, 5>().test();
  DegreeIterator<PlusScalarTest, double, double, ceres::Jet<double, 4>, 0, 3,
                 5>()
      .test();
  DegreeIterator<PlusScalarTest, double, ceres::Jet<double, 4>, double, 0, 3,
                 5>()
      .test();
  DegreeIterator<PlusScalarTest, double, ceres::Jet<double, 4>,
                 ceres::Jet<double, 4>, 0, 3, 5>()
      .test();
}
