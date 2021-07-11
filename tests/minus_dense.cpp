#include "testing.hpp"

TEST_CASE("Dense operator-") {
  DegreeIterator<MinusTest, double, double, double, 0, 3, 5>().test();
  DegreeIterator<MinusTest, double, double, ceres::Jet<double, 4>, 0, 3, 5>()
      .test();
  DegreeIterator<MinusTest, double, ceres::Jet<double, 4>, double, 0, 3, 5>()
      .test();
  DegreeIterator<MinusTest, double, ceres::Jet<double, 4>,
                 ceres::Jet<double, 4>, 0, 3, 5>()
      .test();
}
