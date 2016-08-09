#ifndef __MYKERNEL_HPP
#define __MYKERNEL_HPP

#define D_CHECK(err){\
    if(err !=cudaSuccess)\
        printf("error %s,at %d\n",cudaGetErrorString(err),__LINE__);    \
}

__global__ void g_rand_init(int size,curandState *states);
__global__ void g_calc_energy(int J,int *S,int *E,int row,int col,curandState *states);
__global__ void g_S_init(int *S,int size,curandState *states);
__global__ void g_simulate(int J,float invT,int *S,int *E,int row,int col,curandState *states,int flag);


#endif
