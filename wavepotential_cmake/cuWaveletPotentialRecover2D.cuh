#ifndef __CUWAVELETPOTENTIALRECOVER2D_CUH__
#define __CUWAVELETPOTENTIALRECOVER2D_CUH__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"
#include "CuDivLocalProject.cuh"

class CuWaveletPotentialRecover2D
{
public:
    CuWaveletPotentialRecover2D() = delete;
    CuWaveletPotentialRecover2D(int2 dim, DTYPE2 dx, char bc, WaveletType type_w_curl, WaveletType type_w_div);
    ~CuWaveletPotentialRecover2D();

    void Solve(DTYPE* qz, DTYPE* xv, DTYPE* yv);

private:
    DTYPE2 dx_;
    int2 dim_ext_;
    int2 dim_;
    int size_;
    int size_ext_;
    char bc_;
    char bc_div_;
    int2 levels_;

    WaveletType type_w_curl_;
    WaveletType type_w_div_;
    CuDivLocalProject cudlp_;
};

#endif