#ifndef __ISING2D_HPP
#define __ISING2D_HPP

//#define ROW 3072
#define ROW 2024
//#define ROW 4048
//#define COL 3072
#define COL 2024
//#define COL 4048

#define SPIN int

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
    SPIN *hS;
    SPIN *hE;
    SPIN *dS;
    SPIN *dE;
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
    void spinDtoH();
    void showGraph();
    void printSpin();
    void setDim(int xgrid,int ygrid,int xblock,int yblock);
};

#endif
