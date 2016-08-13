#include <iostream>
#include <array>
#include <string>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>

using namespace std;

#include "Ising2D.hpp"
#include "mykernel.hpp"
#include "Image.hpp"

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
    hS = (SPIN *)malloc(sizeof(SPIN)*size);
    hE = (SPIN *)malloc(sizeof(SPIN)*size);
}

void Ising2D::devInit(){
    dim3 _grid(_dim.grid[0],_dim.grid[1]),_block(_dim.block[0],_dim.block[1]);
    grid = _grid;
    block = _block;
    nthreads = _dim.grid[0]*_dim.grid[1]*_dim.block[0]*_dim.block[1];
    
    cudaMalloc((curandState **)&states,sizeof(curandState)*nthreads);
    cudaMalloc((SPIN **)&dS,sizeof(SPIN)*size);
    cudaMalloc((SPIN **)&dE,sizeof(SPIN)*size);

    devRandInit<<<grid,block>>>(nthreads,states,seed);
    devSpinInit<<<grid,block>>>(size,states,dS);
    devCalcEnergy<<<grid,block>>>(1,dS,dE,ROW,COL);
}


void Ising2D::writeGraph(char *filename){
    Image tmp;
    //tmp.printCVversion();
    spinDtoH();
    tmp.draw(hS,filename);

}

void Ising2D::printSpin(){
    
}

void Ising2D::printEnergy(){
    for(int i = 0;i<20;i++){
        printf("%d\n",hE[i]);
    }
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

void Ising2D::spinDtoH(){
    D_CHECK(cudaMemcpy(hS,dS,sizeof(SPIN)*size,cudaMemcpyDeviceToHost));
}
void Ising2D::energyDtoH(){
    D_CHECK(cudaMemcpy(hE,dE,sizeof(SPIN)*size,cudaMemcpyDeviceToHost));
}


void Ising2D::devEnd(){
    cudaFree(dS);
    cudaFree(dE);
    cudaFree(states);
    D_CHECK(cudaDeviceReset());
}

void Ising2D::deviceRun(){
    int flag;
    int nstep = 500;
    char filename[255];

    for(int i=0;i<nstep;i++){
        sprintf(filename , "test%03d.png",i);
        cout << filename <<endl;
        writeGraph(filename);

        flag = 0;
        devSimulate<<<grid,block>>>(1,1,dS,dE,ROW,COL,states,flag);
        flag = 1;
        devSimulate<<<grid,block>>>(1,1,dS,dE,ROW,COL,states,flag);
        //filename = filename + itoa(i) + ".png";
        
    }
}

void Ising2D::hostRun(){
    
}
