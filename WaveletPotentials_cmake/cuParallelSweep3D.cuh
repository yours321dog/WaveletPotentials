#ifndef __CUPARALLELSWEEP3D_CUH__
#define __CUPARALLELSWEEP3D_CUH__

#include "cudaGlobal.cuh"
#include "cuDst3D.cuh"
#include "cuMultigridFrac3D.cuh"
//#include "cuLinearSolver3D.cuh"

class CuParallelSweep3D
{
public:
    CuParallelSweep3D(int2 dim, DTYPE2 dx, char bc);
    CuParallelSweep3D(int3 dim, DTYPE3 dx, char bc);
    ~CuParallelSweep3D();

    void Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv);
    void Solve(DTYPE* qz, DTYPE* xv, DTYPE* yv);
    void SolveFromMid(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 mid_3d);

    void Project(DTYPE* qx_curl, DTYPE* qy_curl, DTYPE* qz_curl, DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol = 1e-6f);
    void ProjectZero(DTYPE* qx_curl, DTYPE* qy_curl, DTYPE* qz_curl, DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol = 1e-6f);
    void ProjectZero(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol = 1e-6f);
    void ProjectD(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol = 1e-6f);

    void ProjectFrac(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE tol = 1e-6f);
    void ProjectFrac(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, DTYPE tol = 1e-6f);

    void ProjectZero(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* qp_o,
        int3 dim_o, DTYPE dx_inv_o, DTYPE3 off_o, DTYPE3 c_min);

    void Project(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol = 1e-6f);

    void GradientQxyz(DTYPE * qx, DTYPE * qy, DTYPE * qz, DTYPE * qp);
    void static ComputeDivergence(DTYPE* div, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx);

    void static Rescale(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ls, int3 dim, DTYPE3 dx);

private:
    void SetQxQyBoundary(DTYPE * qx, DTYPE * qy, DTYPE * xv, DTYPE * yv, DTYPE * zv, char type = 'n');

    void ComputeDivergence(DTYPE * qb, DTYPE * qx, DTYPE * qy, DTYPE * qz);

    int3 dim_;
    int3 dim_qx_;
    int3 dim_qy_;
    int3 dim_qz_;
    int3 dim_qp_;

    int slice_qx_;
    int slice_qy_;
    int slice_qz_;

    int size_;
    int size_qp_;
    int size_qx_;
    int size_qy_;
    int size_qz_;

    char bc_;
    DTYPE3 dx_;
    DTYPE3 dx_inv_;

    DTYPE* qp_;

    DTYPE* qx_frac_;
    DTYPE* qy_frac_;
    DTYPE* qz_frac_;

    CuDst3D cudst3d_;
    //CuLinearSolver3D culs3d_;
    CuMultigridFrac3D* cumg3d_;

    void UpdateGridBlockDir(dim3& grid_q, dim3& block_q, int& sm_q, int3 dim, char dir);
};

#endif