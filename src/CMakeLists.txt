set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
add_definitions("-Wall -O3")

cuda_add_executable(run6.exe 2d_ising.cu)
