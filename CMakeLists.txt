# ##############################################################################
# Copyright (c) 2021-2022 Dmitriy Korchemkin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ##############################################################################
cmake_minimum_required(VERSION 3.18)
project("polynomials" LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(polynomials INTERFACE)
target_include_directories(polynomials
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(polynomials INTERFACE Eigen3::Eigen)
target_compile_features(polynomials INTERFACE cxx_std_17)
target_compile_options(polynomials INTERFACE -ftemplate-backtrace-limit=0)

option(POLYNOMIALS_INITIALIZE_NAN "Propagate nan-initialization via interface" OFF)
if (${POLYNOMIALS_INITIALIZE_NAN})
  target_compile_definitions(polynomials INTERFACE -DEIGEN_INITIALIZE_MATRICES_BY_NAN)
endif()

enable_testing()
add_subdirectory(tests)
add_subdirectory(thirdparty/doctest)
