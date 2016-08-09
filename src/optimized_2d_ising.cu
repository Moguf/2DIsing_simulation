#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "mycuda.hpp"
#include "Ising2D.hpp"

using namespace std;
#define D_CHECK(err){\
    if(err !=cudaSuccess)\
        printf("error %s,at %d\n",cudaGetErrorString(err),__LINE__);    \
}

#define ROW 3072
#define COL 3072

struct Environments{
    int i;
}typedef Env;

class Ising2D{
private:
    Env env;

public:
    int i;
    void run();
    void 
};

int main(void){

    return 0;
}
    
