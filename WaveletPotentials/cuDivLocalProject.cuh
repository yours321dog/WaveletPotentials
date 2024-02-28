#ifndef __CUDIVLOCALPROJECT_CUH__
#define __CUDIVLOCALPROJECT_CUH__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"

class CuDivLocalProject
{
public:
    CuDivLocalProject(int2 dim, int2 levels, DTYPE2 dx, std::string type);
    CuDivLocalProject(int2 dim, int2 levels, DTYPE2 dx, WaveletType type, char bc);
    ~CuDivLocalProject() = default;

    void ProjectSingle(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt);
    void ProjectLocal(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt);
    void ProjectLocal(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, int2 levels);
    void ProjectLocal(DTYPE* xv, DTYPE* yv, DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt);

private:
    DTYPE* ip_base_x_;
    DTYPE* ip_base_y_;
    DTYPE* ip_proj_x_;
    DTYPE* ip_proj_y_;
    WaveletType wt_base_;
    WaveletType wt_proj_;
    char bc_base_;
    char bc_proj_;
    int2 dim_;
    int2 levels_;
    DTYPE2 dx_inv_;

};

#endif