#ifndef __CUDST_CUH__
#define __CUDST_CUH__

#include "cudaGlobal.cuh"

#include <complex>
#include <cufft.h>

#ifdef USE_DOUBLE
typedef cufftDoubleComplex cufftDtypeComplex;
#else
typedef cufftComplex cufftDtypeComplex;
#endif // USE_DOUBLE


class CuDst
{
public:
    CuDst() = delete;
    CuDst(int2 dim, DTYPE2 dx);
    ~CuDst();

    void Solve(DTYPE* dst, DTYPE* f);
    static void PirntComplex(DTYPE* val, int2 dim);

    static cufftResult CufftC2CExec1D(DTYPE* dst, DTYPE* src, const cufftHandle& plan, int direction);
    static cufftResult CufftC2CExec1D(cufftDtypeComplex* dst, cufftDtypeComplex* src, const cufftHandle& plan, int direction);

private:

    DTYPE2 dx_;
    DTYPE2 len_;
    int2 dim_;
    int2 dim_fft_;
    int size_fft_x_;
    int size_fft_y_;
    cufftHandle plan_x_;
    cufftHandle plan_y_;
    cufftDtypeComplex* buf_;
};

#endif