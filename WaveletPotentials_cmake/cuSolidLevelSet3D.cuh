#ifndef __CUSOLIDLEVELSET3D_CUH__
#define __CUSOLIDLEVELSET3D_CUH__

#include "cudaGlobal.cuh"
#include "cuMultigridFrac3D.cuh"
#include "cuParallelSweep3D.cuh"

class CuSolidLevelSet3D
{
public:
    CuSolidLevelSet3D() = delete;
    CuSolidLevelSet3D(int3 dim, DTYPE3 dx);
    CuSolidLevelSet3D(int3 dim, DTYPE3 dx, DTYPE dx_c);
    ~CuSolidLevelSet3D();

    void InitLsSphere(DTYPE3 center, DTYPE radius);
    void GetFrac(DTYPE* ax, DTYPE* ay, DTYPE* az);

    static void InitLsSphere(DTYPE* ls, DTYPE3 center, DTYPE radius, int3 dim, DTYPE3 dx);
    static void GetFrac(DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, int3 dim, DTYPE3 dx);

private:

    DTYPE* ls_;
    int3 dim_;
    int3 dim_ax_;
    int3 dim_ay_;
    int3 dim_az_;
    DTYPE3 dx_;
    int size_;
    int size_ax_;
    int size_ay_;
    int size_az_;
};

#endif