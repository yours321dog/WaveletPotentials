#ifndef __CUWAVELETPOTENTIALRECOVER3D_CUH__
#define __CUWAVELETPOTENTIALRECOVER3D_CUH__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"
#include "CuDivLocalProject3D.cuh"

class CuWaveletPotentialRecover3D
{
public:
    CuWaveletPotentialRecover3D() = delete;
    CuWaveletPotentialRecover3D(int3 dim, DTYPE3 dx, char bc, WaveletType type_w_curl, WaveletType type_w_div);
    ~CuWaveletPotentialRecover3D();

    void Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv);
    void Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv);
    void CutCoefficients(DTYPE* qx_out, DTYPE* qy_out, DTYPE* qz_out, DTYPE* qx, DTYPE* qy, DTYPE* qz, int level);

private:
    DTYPE3 dx_;
    int3 dim_ext_;
    int3 dim_;
    int size_;
    int size_ext_;
    char bc_;
    char bc_div_;
    int3 levels_;

    WaveletType type_w_curl_;
    WaveletType type_w_div_;
    CuDivLocalProject3D cudlp_;
};

#endif