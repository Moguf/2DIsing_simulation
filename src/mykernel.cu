#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "mykernel.hpp"

__global__ void devRandInit(int size,curandState *states,long int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    if(idx < size){
        curand_init(seed,idx,0,&states[idx]);
    }
}