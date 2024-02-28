#ifndef __CUDST3D_CUH__
#define __CUDST3D_CUH__

#include "cuDst.cuh"

class CuDst3D
{
public:
    CuDst3D() = delete;
    CuDst3D(int3 dim, DTYPE3 dx);
    ~CuDst3D();

    void Solve(DTYPE* dst, DTYPE* f);
    static void PirntComplex(DTYPE* val, int3 dim);

private:
    DTYPE3 dx_;
    DTYPE3 len_;
    int3 dim_;
    int3 dim_fft_;
    int3 dim_fft_x_;
    int3 dim_fft_y_;
    int3 dim_fft_z_;
    int size_fft_x_;
    int size_fft_y_;
    int size_fft_z_;

    dim3 block_x_;
    dim3 grid_x_;
    int sm_x_;

    cufftHandle plan_x_;
    cufftHandle plan_y_;
    cufftHandle plan_z_;

    cufftDtypeComplex* buf_;
};

#endif