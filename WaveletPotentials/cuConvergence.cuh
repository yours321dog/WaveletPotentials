#ifndef __CUCONVERGENCE_CUH__
#define __CUCONVERGENCE_CUH__

#include "cudaGlobal.cuh"

class CuConvergence
{
public:
    CuConvergence() = default;
    ~CuConvergence() = default;
    static CuConvergence* GetInstance();

    void GetB_2d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx, char bc);

    void GetB_3d_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx);
    void GetB_3d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx);
    void GetB_3d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, char bc);
    void GetB_3d_q_zeros(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx);
    void GetB_3d_q_zeros_interp(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx, DTYPE* qp_o,
        int3 dim_o, DTYPE dx_inv_o, DTYPE3 off_o, DTYPE3 c_min);
    void GetB_3d_q_zeros_interp(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx, DTYPE* qx_o, DTYPE* qy_o, DTYPE* qz_o,
        int3 dim_o, DTYPE dx_inv_o, DTYPE3 c_min);
    void GetB_3d_q_d(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx);

private:
    static std::auto_ptr<CuConvergence> instance_;
};

#endif