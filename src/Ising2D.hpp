#ifndef __ISING2D_HPP
#define __ISING2D_HPP

#define ROW 3072
#define COL 3072

struct Environments{
    int i;
}typedef Env;

struct GridBlcok{
    int grid[2];
    int block[2];
}typedef gridblock;

class Ising2D
{
private:
    int size = ROW*COL;
    long int seed = 0;
    Env env;
    gridblock _dim;
    char *hS;
    char *hE;
    char *dS;
    char *dE;
    curandState *states;
    dim3 grid,block;
    int nthreads;
public:
    void devInfo();
    void deviceRun();
    void hostRun();
    void devInit();
    void hostInit();
    void devEnd();
    void hostEnd();
    void setDim(int xgrid,int ygrid,int xblock,int yblock);
};

#endif
