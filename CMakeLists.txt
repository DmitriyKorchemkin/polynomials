project("polynomials" LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(polynomials INTERFACE)
target_include_directories(polynomials
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(polynomials INTERFACE Eigen3::Eigen)
target_compile_features(polynomials INTERFACE cxx_std_17)

enable_testing()
add_subdirectory(thirdparty/doctest)
find_package(Ceres REQUIRED)
add_executable(test test.cpp)
target_link_libraries(test polynomials doctest::doctest Ceres::ceres)
target_compile_options(test PRIVATE -Wall -Wextra -Werror --pedantic)
add_test(NAME dense_test COMMAND $<TARGET_FILE:test>)
