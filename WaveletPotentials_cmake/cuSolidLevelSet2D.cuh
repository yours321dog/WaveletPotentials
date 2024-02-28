#ifndef __CUSOLIDLEVELSET2D_CUH__
#define __CUSOLIDLEVELSET2D_CUH__

#include "cudaGlobal.cuh"

class CuSolidLevelSet2D
{
public:
    CuSolidLevelSet2D() = delete;
    CuSolidLevelSet2D(int2 dim, DTYPE2 dx);
    ~CuSolidLevelSet2D();

    void InitLsSphere(DTYPE2 center, DTYPE radius);
    void GetFrac(DTYPE* ax, DTYPE* ay);
    void GetFracHost(DTYPE* ax, DTYPE* ay);

private:
    DTYPE* ls_;
    int2 dim_;
    int2 dim_ax_;
    int2 dim_ay_;
    DTYPE2 dx_;
    int size_;
    int size_ax_;
    int size_ay_;
};

#endif