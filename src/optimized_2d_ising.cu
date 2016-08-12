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
    int xgrid = 32;
    int ygrid = 32;
    const chrono::system_clock::time_point start =
        chrono::system_clock::now();
        
    Ising2D tmp;

    // setup
    tmp.hostInit();
    tmp.setDim(xgrid,ygrid,(ROW + xgrid -1)/xgrid ,(COL + ygrid -1 ) / ygrid);
    tmp.devInit();
    tmp.devInfo();

    // test
    tmp.spinDtoH();
    //tmp.printSpin();
    tmp.writeGraph();
    //tmp.energyDtoH();
    //tmp.printEnergy();
    // main
    tmp.deviceRun();
    tmp.hostRun();


    // End
    tmp.devEnd();
    tmp.hostEnd();
    
    const chrono::system_clock::time_point end =
        chrono::system_clock::now();
    const auto ttime =
        chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("total::%10ldms\n",ttime.count());
    
    return 0;
}
    
