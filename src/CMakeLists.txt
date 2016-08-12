set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;-arch=sm_30)

add_definitions(-Wall -O3)
add_definitions(-D_MWAITXINTRIN_H_INCLUDED)

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(run.exe  Image.cpp optimized_2d_ising.cu Ising2D.cu mykernel.cu)

target_link_libraries(run.exe -lcurand ${OpenCV_LIBS})