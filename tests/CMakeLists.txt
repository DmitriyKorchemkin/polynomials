# ##############################################################################
# Copyright (c) 2021 Dmitriy Korchemkin
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
find_package(Ceres REQUIRED)

macro(add_new_test name)
  set(target test_${name})
  message(STATUS "TEST TARGET: ${target}")
  add_executable(${target} ${name}.cpp)
  target_link_libraries(${target} polynomials doctest::doctest Ceres::ceres)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror --pedantic)
  add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
endmacro()

add_new_test(plus_dense)
add_new_test(minus_dense)
add_new_test(mul_dense)
add_new_test(plus_scalar_dense)
add_new_test(minus_scalar_dense)
add_new_test(mul_scalar_dense)
