#ifndef __ISING2D_HPP
#define __ISING2D_HPP

#define ROW 3072
#define COL 3072

struct Environments{
    int i;
}typedef Env;

class Ising2D
{
private:
    int size = ROW*COL;
    Env env;
    char *hS;
    char *hE;
    char *dS;
    char *dE;
public:
    void devInfo();
    void run();
    void devInit();
    void hostInit();
    void devEnd();
    void hostEnd();
};

#endif
