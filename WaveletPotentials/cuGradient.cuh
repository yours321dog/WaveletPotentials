#ifndef __CUGRADIENT_CUH__
#define __CUGRADIENT_CUH__

#include "cudaGlobal.cuh"

class CuGradient
{
public:
    CuGradient() = default;
    ~CuGradient() = default;
    static CuGradient* GetInstance();

    void GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc);
    void GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx, char bc);

    void GradientMinusLwtQ(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, DTYPE2 dx, char bc);
    void GradientMinusLwtQ(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx, char bc);

    void GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc);
    void GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx, char bc);

    void GradientMinusLwtQ(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, DTYPE2 dx, char bc);
    void GradientMinusLwtQ(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx, char bc);

    //void GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int2 dim, DTYPE2 dx, char bc);
    void GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc);
    void GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc);
    void GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, 
        DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc);

    void Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc = 'n');
    void Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx, char bc = 'n');

    void Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, char bc = 'n');
    void Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc = 'n');

    void Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc = 'n');
    void Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc = 'n');

    void Gradient3D_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, DTYPE* dx, DTYPE* dy,
        DTYPE* dz, char bc = 'n');
    void Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az,
        int3 dim, DTYPE3 dx, char bc = 'n');
    void Gradient3D_minus_Frac_P(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv, 
        DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, char bc = 'n');
    void Gradient3D_minus_Frac_P_select(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
        DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ax_select, DTYPE* ay_select, DTYPE* az_select, int3 dim, DTYPE3 dx, char bc = 'n');
    void Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, 
        int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc = 'n');
    void Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az,
        int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc = 'n');

    void Gradient3D_minus_Frac_P_bound(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
        DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* is_bound, int3 dim, DTYPE3 dx, char bc = 'n');

    void Gradient3D_LastPhi_X(DTYPE* qx, DTYPE* phi, int3 dim_qx, DTYPE dx_inv);

    void GradientMinus3D_zero_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx);
    void GradientMinus3D_d_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx);

    void GetVelocityFromQ(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, DTYPE dx);

private:
    static std::auto_ptr<CuGradient> instance_;
};

#endif