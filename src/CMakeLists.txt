set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
add_definitions(-Wall -O3)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

cuda_add_executable(run.exe optimized_2d_ising.cu Ising2D.cu)
