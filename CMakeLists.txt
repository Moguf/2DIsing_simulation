cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

add_definitions("-std=c++11")
add_subdirectory(src)