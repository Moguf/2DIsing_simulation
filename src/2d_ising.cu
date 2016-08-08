#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

using namespace std;
#define D_CHECK(err){\
    if(err !=cudaSuccess)\
        printf("error %s,at %d\n",cudaGetErrorString(err),__LINE__);    \
}

#define ROW 2048
#define COL 2048

__global__ void g_rand_init(int size,curandState *states);
__global__ void g_calc_energy(int J,int *S,int *E,int row,int col,curandState *states);
__global__ void g_S_init(int *S,int size,curandState *states);
__global__ void g_simulate(int J,float invT,int *S,int *E,int row,int col,curandState *states,int flag);


template <typename T>
class ising{
public:
    // host values
    int row;
    int col;
    int size;
    T J;
    // 交換相互作用エネルギー
    int block_x,block_y;
    int thread_x,thread_y;
    dim3 grid,block;
    int nthreads;
    int sumE;
    float invT;
    FILE *fp;
    
    T *S;
    // スピンの情報

    T *E;
    // 各スピンのエネルギー

    T *hE;
    // エネルギー計算用
    
    // device values
    cudaError_t err;
    curandState *states;
    //乱数列生成用

    T *dS;
    // デバイス側のスピン

    T *dE;
    // デバイス側の各スピンのエネルギー


    //functions 
    ising(int row_,int col_,T J_,float temperature);
    ~ising();

    void devInit(int bx,int by,int tx,int ty);
    // デバイス側の初期化

    void toFile();
    // ファイルの書き出し

    void checkEnergy();
    // 系のエネルギーのチェック

    void run();
   //シミュレーションのスタート

    void devEnd();
    // デバイスの変数の解放
};

template <typename T>
ising<T>::ising(int row_,int col_,T J_,float temperature_){
    row = row_;
    col = col_;
    size = row_ * col_;
    J = J_;
    invT = 1.0/temperature_;
    sumE = 0;
    S = (T *) malloc(sizeof(T) * size);
    E = (T *) malloc(sizeof(T) * size);
    hE = (T *) malloc(sizeof(T) * size);
    
    if((fp = fopen("result.dat","w")) == NULL){
        printf("Error at opening file.\n");
        exit(0);
    }
    
    fprintf(fp,"#%9d%9d\n",row,col);
    for(int i=0;i<size;i++)
        S[i] = 0;
}

template <typename T>
ising<T>::~ising(){
    free(S);
    free(E);
    free(hE);
    fclose(fp);
}

template <typename T>
void ising<T>::devInit(int bx,int by,int tx,int ty){
    nthreads = bx * by * tx * ty;
    dim3 grid_(bx,by),block_(tx,ty);
    grid = grid_;
    block = block_;

    printf("grid=[%d,%d],block[%d,%d]\n",grid.x,grid.y,block.x,block.y);
    cudaMalloc((curandState **)&states,sizeof(curandState)*nthreads);
    cudaMalloc((T **)&dS,sizeof(T)*size);
    cudaMalloc((T **)&dE,sizeof(T)*size);

    g_rand_init<<<grid,block>>>(size,states);
    g_S_init<<<grid,block>>>(dS,size,states);
    D_CHECK(cudaMemcpy(S,dS,sizeof(int)*size,cudaMemcpyDeviceToHost));
}

template <typename T>
void ising<T>::devEnd(){
    cudaFree(states);
    cudaFree(dS);
    cudaFree(dE);
    cudaDeviceReset();
}

__global__ void g_rand_init(int size,curandState *states){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    if(idx < size){
        curand_init(idx,0,0,&states[idx]);
    }
}

__global__ void g_S_init(int *S,int size,curandState *states){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    if(idx < size){
        if(curand_uniform(&states[idx]) <= 0.5)
            S[idx] = -1;
        else
            S[idx] = 1;
    }
}


template <typename T>
void ising<T>::run(){
    int flag;
    flag = 0;
    g_simulate<<<grid,block>>>(J,invT,dS,dE,row,col,states,flag);
    flag = 1;
    g_simulate<<<grid,block>>>(J,invT,dS,dE,row,col,states,flag);
}

template <typename T>
void ising<T>::toFile(){
    for(int i=0;i<size;i++)
        fprintf(fp,"%3d",S[i]);
    return;
}

__global__ void
g_simulate(int J,float invT,int *S,int *E,int row,int col,curandState *states,int flag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = x + y*gridDim.x*blockDim.x;
    int nx;
    int ny;
    float prob;

    if( x < row && y < col){
        if((x+y)% 2 == flag){
            S[x*col+y] = -S[x*col+y];
            E[x*col+y] = 0;
            nx = x-1;
            ny = y;
            if(nx < 0)
                nx = col-1;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x;
            ny = y-1;
            if(ny < 0)
                ny = row-1;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x+1;
            ny = y;
            if(nx >= row)
                nx = 0;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];

            nx = x;
            ny = y+1;
            if(ny >= col)
                ny = 0;
            E[x*col+y] += -J*S[x*col+y]*S[nx*col+ny];
            if( E[x*col+y]>=0){
                prob = curand_uniform(&states[idx]);
                if(prob > exp(-2.0*E[x*col+y]*invT)){
                    S[x*col+y] = -S[x*col+y];
                }
            }
        }
    }
    __syncthreads();
}

int main(void){
    int row = ROW;
    int col = COL;
    int J = 1;

    ising<int> ising2d(row,col,J,0.5);
    ising2d.devInit(64,64,32,32);

    ising2d.toFile();

    for(int i=0;i<100;i++){
        ising2d.run();
        //ising2d.checkEnergy();                                                                             //ising2d.toFile();
    }
    ising2d.devEnd();
    return 0;
}

