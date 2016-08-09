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
    Env env;
    char *hS;
    char *dS;
public:
    void devInfo();
    void run();
    void devInit();
    void hostInit();
    void devEnd();
    void hostEnd();
};

#endif
