#include <iostream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Ising2D.hpp"
#include "mykernel.hpp"

void Ising2D::devInfo(){

}

void Ising2D::hostInit(){
    hS = (char *)malloc(sizeof(char)*ROW*COL);
}

void Ising2D::devInit(){

}

void Ising2D::hostEnd(){
    free(hS);
}

void Ising2D::devEnd(){

}

void Ising2D::run(){

}


