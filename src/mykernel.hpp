#ifndef __MYKERNEL_HPP
#define __MYKERNEL_HPP

#define D_CHECK(err){\
    if(err !=cudaSuccess)\
        printf("error %s,at %d\n",cudaGetErrorString(err),__LINE__);    \
}

__global__ void devRandInit(int size,curandState *states,long int seed);
__global__ void devRandEnergy(int J,int *S,int *E,int row,int col,curandState *states);
__global__ void devSpinInit(int size,curandState *states,SPIN *dS);
__global__ void devSimulate(int J,float invT,int *S,int *E,int row,int col,curandState *states,int flag);

#endif
