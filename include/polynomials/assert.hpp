#ifndef POLYNOMIALS_ASSERT_HPP
#define POLYNOMIALS_ASSERT_HPP

#include <stdexcept>

#ifndef NDEBUG
#define POLYNOMIALS_ASSERT(what, why)                                          \
  do {                                                                         \
    if (!(what))                                                               \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + ": " + why);         \
  } while (false);
#else
#define POLYNOMIALS_ASSERT(...)
#endif

#endif
