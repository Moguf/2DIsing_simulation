#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Ising2D.hpp"
#include "mykernel.hpp"

using namespace std;

int main(void){
    const chrono::system_clock::time_point start =
        chrono::system_clock::now();
        
    Ising2D tmp;

    tmp.hostInit();
    tmp.setDim(32,32,(ROW + 32 -1)/32 ,(COL + 32 -1 ) / 32);
    tmp.devInit();

    tmp.spinDtoH();
    
    tmp.devInfo();
    tmp.deviceRun();
    tmp.hostRun();
    
    tmp.devEnd();
    tmp.hostEnd();
    
    const chrono::system_clock::time_point end =
        chrono::system_clock::now();
    const auto ttime =
        chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("total::%10ldms\n",ttime.count());
    
    return 0;
}
    
