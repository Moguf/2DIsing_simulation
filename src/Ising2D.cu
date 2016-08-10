#include <iostream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Ising2D.hpp"
#include "mykernel.hpp"

void Ising2D::devInfo(){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Size: %10d\n",size);
    printf("Grid.(%3d,%3d),Block.(%3d,%3d)\n",_dim.grid[0],_dim.grid[1],_dim.block[0],_dim.block[1]);
    printf("Device %d: %s\n",dev,deviceProp.name);
    printf("CUDA Capability Major/Minor version number:   %d.%d\n",deviceProp.major,deviceProp.minor);
    
}

void Ising2D::hostInit(){
    _dim.grid[0] = 32;
    _dim.grid[1] = 32;
    _dim.block[0] = (_dim.grid[0] + ROW -1) / _dim.grid[0];
    _dim.block[1] = (_dim.grid[1] + COL -1) / _dim.grid[1];
    hS = (char *)malloc(sizeof(char)*size);
    hE = (char *)malloc(sizeof(char)*size);
}

void Ising2D::devInit(){
    dim3 _grid(_dim.grid[0],_dim.grid[1]),_block(_dim.block[0],_dim.block[1]);
    grid = _grid;
    block = _block;
    nthreads = _dim.grid[0]*_dim.grid[1]*_dim.block[0]*_dim.block[1];
    
    cudaMalloc((curandState **)&states,sizeof(curandState)*nthreads);
    cudaMalloc((char **)&dS,sizeof(char)*size);
    cudaMalloc((char **)&dE,sizeof(char)*size);

    devRandInit<<<grid,block>>>(nthreads,states,seed);
    devSpinInit<<<grid,block>>>(size,states,dS);
}

void Ising2D::hostEnd(){
    free(hS);
    free(hE);
}

void Ising2D::setDim(int xgrid,int ygrid,int xblock,int yblock){
    _dim.grid[0] = xgrid;
    _dim.grid[1] = ygrid;
    _dim.block[0] = xblock;
    _dim.block[1] = yblock;
}

void Ising2D::devEnd(){
    cudaFree(dS);
    cudaFree(dE);
    cudaFree(states);
    D_CHECK(cudaDeviceReset());
}

void Ising2D::deviceRun(){

}

void Ising2D::hostRun(){
    
}

