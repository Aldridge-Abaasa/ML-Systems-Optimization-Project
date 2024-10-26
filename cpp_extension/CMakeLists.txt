cmake_minimum_required(VERSION 3.10)
project(optimized_ops)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)

add_library(optimized_ops SHARED optimized_ops.cpp)
target_link_libraries(optimized_ops OpenMP::OpenMP_CXX)
