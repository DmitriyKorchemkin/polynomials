project("polynomials" LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(polynomials INTERFACE)
target_include_directories(polynomials
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(polynomials INTERFACE Eigen3::Eigen)

add_subdirectory(thirdparty/doctest)

find_package(Ceres REQUIRED)
add_executable(test test.cpp)
target_link_libraries(test polynomials doctest::doctest Ceres::ceres)
