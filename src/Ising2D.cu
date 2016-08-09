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

    printf("Device %d: %s\n",dev,deviceProp.name);
    

}

void Ising2D::hostInit(){
    hS = (char *)malloc(sizeof(char)*size);
    hE = (char *)malloc(sizeof(char)*size);
}

void Ising2D::devInit(){
    cudaMalloc((char **)&dS,sizeof(char)*size);
    cudaMalloc((char **)&dE,sizeof(char)*size);
}

void Ising2D::hostEnd(){
    free(hS);
    free(hE);
}

void Ising2D::devEnd(){
    cudaFree(dS);
    cudaFree(dE);
}

void Ising2D::run(){

}


