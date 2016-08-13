#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

using namespace std;

#include "Ising2D.hpp"
#include "mykernel.hpp"

__global__ void devRandInit(int size,curandState *states,long int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    long long int idx = x + y*gridDim.x*blockDim.x;
    if( idx < size)
        curand_init(seed,idx,0,&states[idx]);
}

__global__ void devSpinInit(int size,curandState *states,SPIN *dS)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    if(idx < size){
        if(curand_uniform(&states[idx]) <= 0.5)
            dS[idx] = -1;
        else
            dS[idx] = 1;
    }
}

__global__ void devCalcEnergy(int J,int *S,int *E,int row,int col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int nx;
    int ny;

    if( x < row && y < col){
        E[x*col+y] = 0;
        nx = x-1;
        ny = y;
        if(nx < 0)
            nx = col-1;
        E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

        nx = x;
        ny = y-1;
        if(ny < 0)
            ny = row-1;
        E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

        nx = x+1;
        ny = y;
        if(nx >= row)
            nx = 0;
        E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

        nx = x;
        ny = y+1;
        if(ny >= col)
            ny = 0;
        E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];
     }
    return ;
}

__global__ void devSimulate(int J,float invT,int *S,int *E,int row,int col,curandState *states,int flag){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    int nx;
    int ny;
    float prob;

    if( x < row && y < col){
        if((x+y)% 2 == flag){
            S[x*col+y] = -S[x*col+y];
            E[x*col+y] = 0;
            nx = x-1;
            ny = y;
            if(nx < 0)
                nx = col-1;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x;
            ny = y-1;
            if(ny < 0)
                ny = row-1;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x+1;
            ny = y;
            if(nx >= row)
                nx = 0;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x;
            ny = y+1;
            if(ny >= col)
                ny = 0;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];
            if( E[x*col+y]>=0){
                prob = curand_uniform(&states[idx]);
                if(prob > exp(-2.0*E[x*col+y]*invT)){
                    S[x*col+y] = -S[x*col+y];
                }
            }
        }
    }
    __syncthreads();
}

