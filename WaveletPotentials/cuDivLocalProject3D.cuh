#ifndef __CUDIVLOCALPROJECT3D_CUH__
#define __CUDIVLOCALPROJECT3D_CUH__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"

class CuDivLocalProject3D
{
public:
    CuDivLocalProject3D(int3 dim, int3 levels, DTYPE3 dx, std::string type);
    CuDivLocalProject3D(int3 dim, int3 levels, DTYPE3 dx, WaveletType type, char bc);
    ~CuDivLocalProject3D() = default;

    void ProjectLocal(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);
    void ProjectLocal(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);
    void ProjectLocal_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);
    void ProjectLocal_q_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);

    void ProjectLocal_q_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt_out, DTYPE* yv_lwt_out, DTYPE* zv_lwt_out,
        DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);

    void ProjectLocal_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels);
    void ProjectLocal_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 levels);
    void ProjectLocal_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 levels);

private:
    DTYPE* ip_base_x_;
    DTYPE* ip_base_y_;
    DTYPE* ip_base_z_;
    DTYPE* ip_proj_x_;
    DTYPE* ip_proj_y_;
    DTYPE* ip_proj_z_;

    DTYPE* proj_coef_xv_;
    DTYPE* proj_coef_yv_;
    DTYPE* proj_coef_zv_;
    int proj_len_;

    WaveletType wt_base_;
    WaveletType wt_proj_;
    char bc_base_;
    char bc_proj_;
    int3 dim_;
    int3 levels_;
    DTYPE3 dx_inv_;

};

#endif