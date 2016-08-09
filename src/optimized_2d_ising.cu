#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "Ising2D.hpp"
#include "mykernel.hpp"

using namespace std;

int main(void){
    Ising2D tmp;
    tmp.hostInit();
    tmp.devInit();
    tmp.run();
    tmp.devEnd();
    tmp.hostEnd();

    return 0;
}
    
