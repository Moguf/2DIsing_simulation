#include <stdio.h>
#include <stdlib.h>
#include "MT.h"
#include <omp.h>
#include <math.h>

float calcE(int *m,int row,int col){
    int i,j,nx,ny;
    float J = 1;
    float E = 0;
    #pragma omp parallel num_threads(12)
    #pragma omp for private(i,j),reduction(+:E)
    for(i=0;i<row;i++){
        for(j=0;j<col;j++){
            nx = i;
            ny = (j+1)%col;
            E += -J * m[i*row+j]*m[nx*row+ny];

            nx = (i+1)%row;
            ny = j;
            E += -J * m[i*row+j]*m[nx*row+ny];

            nx = i;
            ny = ((j-1)%col != -1) ? j-1 : col-1;
            E += -J * m[i*row+j]*m[nx*row+ny];
            
            nx = ((i-1)%row != -1) ? i-1 : row-1;
            ny = j;
            E += -J * m[i*row+j]*m[nx*row+ny];
        }
    }
    return E;
}

float calcDeltaE(int *m,int row,int col,int mp){
    int nx,ny,x,y;
    float J = 1;
    float dE = 0;
    
    x = mp/row;
    y = mp%row;
    
    nx = x;
    ny = (y+1)%col;
    dE += -J * m[x*row+y]*m[nx*row+ny];

    nx = (x+1)%row;
    ny = y;
    dE += -J * m[x*row+y]*m[nx*row+ny];

    nx = x;
    ny = ((y-1)%col != -1) ? y-1 : col-1;
    dE += -J * m[x*row+y]*m[nx*row+ny];
            
    nx = ((x-1)%row != -1) ? x-1 : row-1;
    ny = y;
    dE += -J * m[x*row+y]*m[nx*row+ny];
    
    return -4*dE;
}

int init(int *m,int size){
    int i;
    int sign[] = {-1,1};
    for(i=0;i<size;i++)
        m[i] = sign[genrand_int32()%2];
    return(0);
}


int metropolis(int *m,int row,int col,int block_row,int block_col,int block_num){
    int i,j,point;
    float T = 3.0;
    float invT = 1/T;
    float dE,dEtot=0;
    float prob;

    int row_size = row/block_row;
    int col_size = col/block_col;
    int rowIdx = block_num/block_col;
    int colIdx = block_num%block_col;
    int stride = col_size*block_col;
    
    int start = rowIdx*col*row_size+colIdx*col_size;

    for(i=0;i<row_size;i++){
        for(j=0;j<col_size;j++){
            point = start+stride*i+j;
            dE=calcDeltaE(m,row,col,point);

            if (dE<=0){
                m[point]=-m[point];
                dEtot+=dE;
            }else{
                prob = genrand_real1();
                if(prob < exp(-dE*invT)){
                    m[point]=-m[point];
                    dEtot+=dE;
                }
            }
        }
    }
    return(dEtot);
}


void simulate(int *m,int row,int col,int nstep){
    int istep,i;
    float E;
    E=calcE(m,row,col);
    for(istep=0;istep<nstep;istep++){
        #pragma omp parallel num_threads(12)
        {
        #pragma omp for 
        for(i=0;i<24;i+=2)
            E+=metropolis(m,row,col,8,3,i);
        #pragma omp for 
        for(i=1;i<24;i+=2)
            E+=metropolis(m,row,col,8,3,i);
        }
    }
}

int main(void){
    int bsize = 11;
    int row = 3<<bsize;
    int col = 3<<bsize;
    int size = row * col;
    int nstep = 100;
    int seed = 1;
    int *mat;    

    init_genrand(seed);

    mat = (int *) malloc(sizeof(int)*size);
    init(mat,size);

    simulate(mat,row,col,nstep);

    printf("row=%5d,col=%5d\n",row,col);
    free(mat);

    return(0);
}
