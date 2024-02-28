#include "cuDivLocalProject.cuh"
#include "cuGetInnerProduct.cuh"

#include "DivL0Matrix.h"

__constant__ DTYPE div_l0_mat[16];

CuDivLocalProject::CuDivLocalProject(int2 dim, int2 levels, DTYPE2 dx, std::string type) : dim_(dim), levels_(levels)
{
    bc_base_ = type[0];
    bc_proj_ = proj_bc(type[0]);
    wt_base_ = type_from_string(type);
    wt_proj_ = proj_WaveletType(wt_base_);
    ip_base_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.x);
    ip_base_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.y);
    ip_proj_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.x - 1);
    ip_proj_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.y - 1);

    dx_inv_ = { DTYPE(1.) / dx.x, DTYPE(1.) / dx.y };

    DTYPE tmp[16];
    GetDivL0Matrix(tmp, levels);
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat, &tmp, 16 * sizeof(DTYPE)));
}

CuDivLocalProject::CuDivLocalProject(int2 dim, int2 levels, DTYPE2 dx, WaveletType type, char bc) : dim_(dim), levels_(levels)
{
    bc_base_ = bc;
    bc_proj_ = proj_bc(bc);
    wt_base_ = type;
    wt_proj_ = proj_WaveletType(wt_base_);
    ip_base_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.x);
    ip_base_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.y);
    ip_proj_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.x - 1);
    ip_proj_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.y - 1);

    dx_inv_ = { DTYPE(1.) / dx.x, DTYPE(1.) / dx.y };

    DTYPE tmp[16];
    GetDivL0Matrix(tmp, levels);
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat, &tmp, 16 * sizeof(DTYPE)));
}

__device__ __forceinline__ DTYPE LocalProjFromNei_q(const DTYPE& xv_in, const DTYPE& yv_in, const DTYPE& wjBiBj_x,
    const DTYPE& wjBiBj_y, const DTYPE& wi_x, const DTYPE& wi_y, const DTYPE& proj_i)
{
    return (xv_in * wjBiBj_y - yv_in * wjBiBj_x) - proj_i * (wi_x * wjBiBj_x + wi_y * wjBiBj_y);
}

__global__ void cuProjectLocal_q_cross(DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 dim, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    int2 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1 };
    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    int slice_xy = dim_dst.x * dim_dst.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_dst_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * dim_dst.y;
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
        //printf("idx_sm: (%d, %d), idx_gm: (%d, %d), val: %f\n", threadIdx.x, threadIdx.y, data_idx3.y, threadIdx.y - 5, sm_ip_1_y[idx_sm]);
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_dst_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim_dst.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;

    __syncthreads();

    //int idx_x_0 = data_idx3.x + 1;
    //int idx_y_0 = data_idx3.y + 1;
    int idx_xv = (data_idx3.y - 1) + (data_idx3.x * dim_dst.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_dst_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x && data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
    int jx = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //            data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_mid_y, ip_1_mid_y, ip_0_x, ip_1_x, proj_i);
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_m2_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_m1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_p1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_p2_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;


        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
        //q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3.y > 1)
        //{
        //    q_lwt_y = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;
        //    //q_lwt_y = proj_i;
        //    //p_lwt_y = data_idx3.x > 0 ? proj_i *0.5  : yv_in / scale_i_y;
        //    //p_lwt_y = data_idx3.x > 0 ? 0.f : yv_in / scale_i_y;
        //}
    }

    __syncthreads();


    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //                data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2);

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);


        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        //if (data_idx3_x < dim_ext.x && data_idx3_y < dim_ext.y)
        //{
        //    //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f, proj_i: %f\n",
        //    //    data_idx3_x, data_idx3_y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        //    printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3_x, data_idx3_y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        //}

        //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : xv_in / scale_i_x;
        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
        //sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3_x > 1)
        //{
        //    sm_xv[idx_sm] = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;

        //    //sm_xv[idx_sm] = proj_i * 1.f ;
        //    //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f : xv_in  / scale_i_x;
        //}
        //else
        //{
        //    sm_xv[idx_sm] = 0.f;
        //}
    }
    else
    {
        sm_xv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int idx = data_idx3.y + data_idx3.x * dim_dst_ext.y;
        {
            idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
            q_lwt[idx] = sm_xv[idx_sm] + q_lwt_y;
            //q_lwt[idx] =  q_lwt_y;
        }
        //p_lwt[data_idx3.y + data_idx3.x * dim.y] = p_lwt_y;
        //printf("idx: (%d, %d), proj_i: %f\n", data_idx3.x, data_idx3.y, proj_i);
    }
}

__global__ void cuProjectLocal_q_cross_n(DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 dim, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    int2 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1 };

    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    //int slice_xy = dim.x * dim.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_dst_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * (dim_dst.y);
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
        //printf("idx_sm: (%d, %d), idx_gm: (%d, %d), val: %f\n", threadIdx.x, threadIdx.y, data_idx3.y, threadIdx.y - 5, sm_ip_1_y[idx_sm]);
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_dst_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim_dst.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;

    __syncthreads();

    //int idx_x_0 = data_idx3.x + 1;
    //int idx_y_0 = data_idx3.y + 1;
    int idx_xv = (data_idx3.y - 1) + (data_idx3.x * dim_dst.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_dst_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x&& data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(max(data_idx3.y - 1, 0))) + 1;
    int jx = levels.x - (32 - __clz(max(data_idx3.x - 1, 0))) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //            data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_mid_y, ip_1_mid_y, ip_0_x, ip_1_x, proj_i);
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_m2_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_m1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_p1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_p2_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;


        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
        //if (data_idx3.y > 1)
        //{
        //    q_lwt_y = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;
        //    //q_lwt_y = proj_i;
        //    //p_lwt_y = data_idx3.x > 0 ? proj_i *0.5  : yv_in / scale_i_y;
        //    //p_lwt_y = data_idx3.x > 0 ? 0.f : yv_in / scale_i_y;
        //}
    }
    else if (data_idx3.x == 1 || data_idx3.x == 0)
    {

        //DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

        DTYPE xv_in_0 = sm_xv[idx_sm - data_idx3.x * (blockDim.x + 1)];
        DTYPE xv_in_1 = sm_xv[idx_sm + (1 - data_idx3.x) * (blockDim.x + 1)];

        yv_in = sm_yv[threadIdx.x + (l_halo + 1) * (blockDim.x + 1)];

        DTYPE ip_1_x_00 = ip_base_x[0];
        DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];

        DTYPE ip_0_x = ip_proj_x[0];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        scale_i_x = 1.f * dx_inv.x / dim_dst.x;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_mat[1] = -scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x_01 * ip_0_mid_y;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), xv_in_0: %f, xv_in_1: %f, yv_in: %f, ip_1_x_00: %f, ip_1_x_01: %f, ip_0_x: %f, ip_0_mid_y: %f, ip_1_mid_y: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, xv_in_0, xv_in_1, yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_0_mid_y, ip_1_mid_y, proj_mat[0], proj_mat[1]);
        //}

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3.x;
        int idx1 = data_idx3.x;

        DTYPE proj_f1 = +yv_in * scale_i_x * ip_0_x * ip_1_mid_y;
        proj_f1 += xv_in_0 * scale_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_f1 += xv_in_1 * scale_i_y * ip_1_x_01 * ip_0_mid_y;

        DTYPE proj_f2 = -yv_in * scale_i_x * ip_0_x * ip_1_mid_y;
        proj_f2 += xv_in_0 * scale_i_y * ip_1_x_01 * ip_0_mid_y;
        proj_f2 += xv_in_1 * scale_i_y * ip_1_x_00 * ip_0_mid_y;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), f1_y: %f, f1_x1: %f, f1_x2: %f, f2_y: %f, f2_x1: %f, f2_x2: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, -scale_i_x * ip_0_x * ip_1_mid_y, scale_i_y* ip_1_x_00* ip_0_mid_y, 
        //        scale_i_y* ip_1_x_01* ip_0_mid_y, scale_i_x* ip_0_x* ip_1_mid_y, scale_i_y* ip_1_x_01* ip_0_mid_y,
        //        scale_i_y* ip_1_x_00* ip_0_mid_y, proj_mat[0], proj_mat[1]);
        //}
        if (data_idx3.y > 1)
        {
            q_lwt_y = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        }
        //else
        //{
        //    ip_0_mid_y = __shfl_sync(0xF, ip_0_mid_y, 3);
        //    q_lwt_y = data_idx3.y == 0 ? ip_1_x_00 * ip_0_mid_y : ip_1_x_01 * ip_0_mid_y;
        //}

        //printf("idx: (%d, %d), threadIdx.x: %d, xv_in_0: %f, xv_in_1: %f, yv_in: %f, scale_i_x: %f, scale_i_y: %f, proj_mat_0: %f, proj_mat_1: %f, q_lwt_y: %f\n",
        //    data_idx3.x, data_idx3.y, threadIdx.x, xv_in_0, xv_in_1, yv_in, scale_i_x, scale_i_y, proj_mat[0], proj_mat[1], q_lwt_y);
    }

    DTYPE xv_in_remain = data_idx3_y == 1 || data_idx3_y == 0 ? sm_xv[threadIdx.x * (blockDim.x + 1) + (l_halo + 1)] : 0;

    __syncthreads();

    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //                data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2);

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);


        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        //if (data_idx3_x < dim_ext.x && data_idx3_y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f, proj_i: %f\n",
        //        data_idx3_x, data_idx3_y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        //    //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3_x, data_idx3_y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        //}

        //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : xv_in / scale_i_x;
        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
        //if (data_idx3_x > 1)
        //{
        //    sm_xv[idx_sm] = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;

        //    //sm_xv[idx_sm] = proj_i * 1.f ;
        //    //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f : xv_in  / scale_i_x;
        //}
        //else
        //{
        //    sm_xv[idx_sm] = 0.f;
        //}
    }
    else if (data_idx3_y == 1 || data_idx3_y == 0)
    {

        //DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

        DTYPE yv_in_0 = sm_yv[idx_sm - data_idx3_y];
        DTYPE yv_in_1 = sm_yv[idx_sm + (1 - data_idx3_y)];

        xv_in = xv_in_remain;

        DTYPE ip_1_y_00 = ip_base_y[0];
        DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];

        DTYPE ip_0_y = ip_proj_y[0];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        scale_i_y = 1.f * dx_inv.y / dim_dst.y;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_00 + scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_mat[1] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_01 - scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;

        //if (data_idx3_x < dim_ext.x)
        //{
        //    printf("idx: (%d, %d), yv_in_0: %f, yv_in_1: %f, xv_in: %f, ip_1_y_00: %f, ip_1_y_01: %f, ip_0_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3_x, data_idx3_y, yv_in_0, yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_0_mid_x, ip_1_mid_x, proj_mat[0], proj_mat[1]);
        //}

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3_y;
        int idx1 = data_idx3_y;

        DTYPE ip_0_x_ip_1_00 = ip_0_mid_x * ip_1_y_00;
        DTYPE ip_0_x_ip_1_01 = ip_0_mid_x * ip_1_y_01;
        DTYPE proj_f1 = -xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f1 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_00;
        proj_f1 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_01;

        DTYPE proj_f2 = xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f2 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_01;
        proj_f2 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_00;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), f1_x: %f, f1_y1: %f, f1_y2: %f, f2_x: %f, f2_y1: %f, f2_y2: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, -scale_i_y * ip_1_mid_x * ip_0_y, scale_i_x* ip_0_x_ip_1_00,
        //        scale_i_x* ip_0_x_ip_1_01, scale_i_y* ip_1_mid_x* ip_0_y, scale_i_x* ip_0_x_ip_1_01,
        //        scale_i_x* ip_0_x_ip_1_00, proj_mat[0], proj_mat[1]);
        //}

        //sm_xv[idx_sm] = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        if (data_idx3_x > 1)
        {
            sm_xv[idx_sm] = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        }
        else
        {
         
            yv_in_0 = __shfl_sync(0xF, yv_in_0, 3);
            yv_in_1 = __shfl_sync(0xF, yv_in_1, 3);

            DTYPE xv_in_0 = __shfl_sync(0xF, xv_in, 2);
            DTYPE xv_in_1 = __shfl_sync(0xF, xv_in, 3);

            __syncwarp(0xF);

            int idx_inv = max((data_idx3_y * 2 + data_idx3_x) * 4, 0);
            DTYPE a00_inv = div_l0_mat[idx_inv + 0];
            DTYPE a01_inv = div_l0_mat[idx_inv + 1];
            DTYPE a02_inv = div_l0_mat[idx_inv + 2];
            DTYPE a03_inv = div_l0_mat[idx_inv + 3];

            sm_xv[idx_sm] = a02_inv * xv_in_0 + a03_inv * xv_in_1 - a00_inv * yv_in_0 - a01_inv * yv_in_1;
            q_lwt_y = 0.f;

            //printf("idx: (%d, %d), yv_in_0: %f, yv_in_1: %f, xv_in_0: %f, xv_in_1: %f, a00_inv: %f, a01_inv: %f, a02_inv: %f, a03_inv: %f, val: %f\n",
            //        data_idx3_x, data_idx3_y, yv_in_0, yv_in_1, xv_in_0, xv_in_1, a00_inv, a01_inv, a02_inv, a03_inv, sm_xv[idx_sm]);
            //printf("idx: (%d, %d), threadIdx.x: %d, a00: %f, a00_inv: %f, a01_inv: %f, a02_inv: %f, a03_inv: %f\n",
            //    data_idx3_x, data_idx3_y, threadIdx.x, a00_inv, a01_inv, a02_inv, a03_inv);
        }
    }
    else
    {
        sm_yv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int idx = data_idx3.y + data_idx3.x * dim_dst_ext.y;
        idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
        {
            q_lwt[idx] = sm_xv[idx_sm] + q_lwt_y;
            //q_lwt[idx] =  q_lwt_y;
        }
        //p_lwt[data_idx3.y + data_idx3.x * dim.y] = p_lwt_y;
        //printf("idx: (%d, %d), proj_i: %f\n", data_idx3.x, data_idx3.y, proj_i);
    }
}

__global__ void cuProjectLocal_q_single(DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 dim, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    int2 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1 };
    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    int slice_xy = dim_dst.x * dim_dst.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_dst_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * dim_dst.y;
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
        //printf("idx_sm: (%d, %d), idx_gm: (%d, %d), val: %f\n", threadIdx.x, threadIdx.y, data_idx3.y, threadIdx.y - 5, sm_ip_1_y[idx_sm]);
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_dst_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim_dst.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;

    __syncthreads();

    //int idx_x_0 = data_idx3.x + 1;
    //int idx_y_0 = data_idx3.y + 1;
    int idx_xv = (data_idx3.y - 1) + (data_idx3.x * dim_dst.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_dst_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x&& data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
    int jx = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //            data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_mid_y, ip_1_mid_y, ip_0_x, ip_1_x, proj_i);
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_m2_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_m1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_p1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_p2_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;


        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f : 0.f;
        //q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3.y > 1)
        //{
        //    q_lwt_y = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;
        //    //q_lwt_y = proj_i;
        //    //p_lwt_y = data_idx3.x > 0 ? proj_i *0.5  : yv_in / scale_i_y;
        //    //p_lwt_y = data_idx3.x > 0 ? 0.f : yv_in / scale_i_y;
        //}
    }

    __syncthreads();


    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //                data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2);

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);


        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        //if (data_idx3_x < dim_ext.x && data_idx3_y < dim_ext.y)
        //{
        //    //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f, proj_i: %f\n",
        //    //    data_idx3_x, data_idx3_y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        //    printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3_x, data_idx3_y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        //}

        //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : xv_in / scale_i_x;
        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f : 0.f;
        //sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3_x > 1)
        //{
        //    sm_xv[idx_sm] = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;

        //    //sm_xv[idx_sm] = proj_i * 1.f ;
        //    //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f : xv_in  / scale_i_x;
        //}
        //else
        //{
        //    sm_xv[idx_sm] = 0.f;
        //}
    }
    else
    {
        sm_xv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int idx = data_idx3.y + data_idx3.x * dim_dst_ext.y;
        {
            idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
            q_lwt[idx] = sm_xv[idx_sm] + q_lwt_y;
            //q_lwt[idx] =  q_lwt_y;
        }
        //p_lwt[data_idx3.y + data_idx3.x * dim.y] = p_lwt_y;
        //printf("idx: (%d, %d), proj_i: %f\n", data_idx3.x, data_idx3.y, proj_i);
    }
}

__global__ void cuProjectLocal_q_single_n(DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 dim, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    int2 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1 };

    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    //int slice_xy = dim.x * dim.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_dst_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * (dim_dst.y);
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
        //printf("idx_sm: (%d, %d), idx_gm: (%d, %d), val: %f\n", threadIdx.x, threadIdx.y, data_idx3.y, threadIdx.y - 5, sm_ip_1_y[idx_sm]);
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_dst_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim_dst.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;

    __syncthreads();

    //int idx_x_0 = data_idx3.x + 1;
    //int idx_y_0 = data_idx3.y + 1;
    int idx_xv = (data_idx3.y - 1) + (data_idx3.x * dim_dst.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_dst_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x&& data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(max(data_idx3.y - 1, 0))) + 1;
    int jx = levels.x - (32 - __clz(max(data_idx3.x - 1, 0))) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //            data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_mid_y, ip_1_mid_y, ip_0_x, ip_1_x, proj_i);
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_m2_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_m2_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_m1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_m1_y, scale_i_x, scale_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_x * ip_1_p1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_i_y, 1) * ip_1_x * ip_0_p1_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_x * ip_1_p2_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_i_y, 2) * ip_1_x * ip_0_p2_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;


        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3.y > 1)
        //{
        //    q_lwt_y = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;
        //    //q_lwt_y = proj_i;
        //    //p_lwt_y = data_idx3.x > 0 ? proj_i *0.5  : yv_in / scale_i_y;
        //    //p_lwt_y = data_idx3.x > 0 ? 0.f : yv_in / scale_i_y;
        //}
    }
    else if (data_idx3.x == 1 || data_idx3.x == 0)
    {

        //DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

        DTYPE xv_in_0 = sm_xv[idx_sm - data_idx3.x * (blockDim.x + 1)];
        DTYPE xv_in_1 = sm_xv[idx_sm + (1 - data_idx3.x) * (blockDim.x + 1)];

        yv_in = sm_yv[threadIdx.x + (l_halo + 1) * (blockDim.x + 1)];

        DTYPE ip_1_x_00 = ip_base_x[0];
        DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];

        DTYPE ip_0_x = ip_proj_x[0];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        scale_i_x = 1.f * dx_inv.x / dim_dst.x;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_mat[1] = -scale_i_x * scale_i_x * ip_0_x * ip_1_mid_y + scale_i_y * scale_i_y * ip_1_x_01 * ip_0_mid_y;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), xv_in_0: %f, xv_in_1: %f, yv_in: %f, ip_1_x_00: %f, ip_1_x_01: %f, ip_0_x: %f, ip_0_mid_y: %f, ip_1_mid_y: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, xv_in_0, xv_in_1, yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_0_mid_y, ip_1_mid_y, proj_mat[0], proj_mat[1]);
        //}

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3.x;
        int idx1 = data_idx3.x;

        DTYPE proj_f1 = +yv_in * scale_i_x * ip_0_x * ip_1_mid_y;
        proj_f1 += xv_in_0 * scale_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_f1 += xv_in_1 * scale_i_y * ip_1_x_01 * ip_0_mid_y;

        DTYPE proj_f2 = -yv_in * scale_i_x * ip_0_x * ip_1_mid_y;
        proj_f2 += xv_in_0 * scale_i_y * ip_1_x_01 * ip_0_mid_y;
        proj_f2 += xv_in_1 * scale_i_y * ip_1_x_00 * ip_0_mid_y;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), f1_y: %f, f1_x1: %f, f1_x2: %f, f2_y: %f, f2_x1: %f, f2_x2: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, -scale_i_x * ip_0_x * ip_1_mid_y, scale_i_y* ip_1_x_00* ip_0_mid_y, 
        //        scale_i_y* ip_1_x_01* ip_0_mid_y, scale_i_x* ip_0_x* ip_1_mid_y, scale_i_y* ip_1_x_01* ip_0_mid_y,
        //        scale_i_y* ip_1_x_00* ip_0_mid_y, proj_mat[0], proj_mat[1]);
        //}
        if (data_idx3.y > 1)
        {
            q_lwt_y = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        }
        //else
        //{
        //    ip_0_mid_y = __shfl_sync(0xF, ip_0_mid_y, 3);
        //    q_lwt_y = data_idx3.y == 0 ? ip_1_x_00 * ip_0_mid_y : ip_1_x_01 * ip_0_mid_y;
        //}

        //printf("idx: (%d, %d), threadIdx.x: %d, xv_in_0: %f, xv_in_1: %f, yv_in: %f, scale_i_x: %f, scale_i_y: %f, proj_mat_0: %f, proj_mat_1: %f, q_lwt_y: %f\n",
        //    data_idx3.x, data_idx3.y, threadIdx.x, xv_in_0, xv_in_1, yv_in, scale_i_x, scale_i_y, proj_mat[0], proj_mat[1], q_lwt_y);
    }

    DTYPE xv_in_remain = data_idx3_y == 1 || data_idx3_y == 0 ? sm_xv[threadIdx.x * (blockDim.x + 1) + (l_halo + 1)] : 0;

    __syncthreads();

    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;
        //if (data_idx3.y < dim_ext.y && data_idx3.x < dim_ext.x)
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_x: %f, ip_1_x: %f, proj_i: %f\n",
        //                data_idx3.x, data_idx3.y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];
        //DTYPE proj_m2 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 2) * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f\n", data_idx3.x, data_idx3.y, proj_i, proj_m2);

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        //DTYPE proj_m1 = LocalProjFromNei(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
        //    __shfl_up_sync(-1, scale_i_y, 1) * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);


        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        //DTYPE proj_p1 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 1) * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        //DTYPE proj_p2 = LocalProjFromNei(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
        //    __shfl_down_sync(-1, scale_i_y, 2) * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);

        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        //if (data_idx3_x < dim_ext.x && data_idx3_y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d): scale_i_y: %f, scale_i_x: %f, ip_0_y: %f, ip_1_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f, proj_i: %f\n",
        //        data_idx3_x, data_idx3_y, scale_i_y, scale_i_x, ip_0_y, ip_1_y, ip_0_mid_x, ip_1_mid_x, proj_i);
        //    //printf("idx: (%d, %d), proj_i: %f, proj_m2: %f, proj_m1: %f, proj_p1: %f, proj_p2: %f\n", data_idx3_x, data_idx3_y, proj_i, proj_m2, proj_m1, proj_p1, proj_p2);

        //}

        //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : xv_in / scale_i_x;
        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f : 0.f;
        //if (data_idx3_x > 1)
        //{
        //    sm_xv[idx_sm] = proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2;

        //    //sm_xv[idx_sm] = proj_i * 1.f ;
        //    //sm_xv[idx_sm] = data_idx3_y > 0 ? proj_i * 0.5f : xv_in  / scale_i_x;
        //}
        //else
        //{
        //    sm_xv[idx_sm] = 0.f;
        //}
    }
    else if (data_idx3_y == 1 || data_idx3_y == 0)
    {

        //DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

        DTYPE yv_in_0 = sm_yv[idx_sm - data_idx3_y];
        DTYPE yv_in_1 = sm_yv[idx_sm + (1 - data_idx3_y)];

        xv_in = xv_in_remain;

        DTYPE ip_1_y_00 = ip_base_y[0];
        DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];

        DTYPE ip_0_y = ip_proj_y[0];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        scale_i_y = 1.f * dx_inv.y / dim_dst.y;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_00 + scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_mat[1] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_01 - scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;

        //if (data_idx3_x < dim_ext.x)
        //{
        //    printf("idx: (%d, %d), yv_in_0: %f, yv_in_1: %f, xv_in: %f, ip_1_y_00: %f, ip_1_y_01: %f, ip_0_y: %f, ip_0_mid_x: %f, ip_1_mid_x: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3_x, data_idx3_y, yv_in_0, yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_0_mid_x, ip_1_mid_x, proj_mat[0], proj_mat[1]);
        //}

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3_y;
        int idx1 = data_idx3_y;

        DTYPE ip_0_x_ip_1_00 = ip_0_mid_x * ip_1_y_00;
        DTYPE ip_0_x_ip_1_01 = ip_0_mid_x * ip_1_y_01;
        DTYPE proj_f1 = -xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f1 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_00;
        proj_f1 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_01;

        DTYPE proj_f2 = xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f2 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_01;
        proj_f2 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_00;

        //if (data_idx3.y < dim_ext.y)
        //{
        //    printf("idx: (%d, %d), f1_x: %f, f1_y1: %f, f1_y2: %f, f2_x: %f, f2_y1: %f, f2_y2: %f,  proj_mat_0: %f, proj_mat_1: %f\n",
        //        data_idx3.x, data_idx3.y, -scale_i_y * ip_1_mid_x * ip_0_y, scale_i_x* ip_0_x_ip_1_00,
        //        scale_i_x* ip_0_x_ip_1_01, scale_i_y* ip_1_mid_x* ip_0_y, scale_i_x* ip_0_x_ip_1_01,
        //        scale_i_x* ip_0_x_ip_1_00, proj_mat[0], proj_mat[1]);
        //}

        //sm_xv[idx_sm] = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        if (data_idx3_x > 1)
        {
            sm_xv[idx_sm] = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        }
        else
        {

            yv_in_0 = __shfl_sync(0xF, yv_in_0, 3);
            yv_in_1 = __shfl_sync(0xF, yv_in_1, 3);

            DTYPE xv_in_0 = __shfl_sync(0xF, xv_in, 2);
            DTYPE xv_in_1 = __shfl_sync(0xF, xv_in, 3);

            __syncwarp(0xF);

            int idx_inv = max((data_idx3_y * 2 + data_idx3_x) * 4, 0);
            DTYPE a00_inv = div_l0_mat[idx_inv + 0];
            DTYPE a01_inv = div_l0_mat[idx_inv + 1];
            DTYPE a02_inv = div_l0_mat[idx_inv + 2];
            DTYPE a03_inv = div_l0_mat[idx_inv + 3];

            sm_xv[idx_sm] = a02_inv * xv_in_0 + a03_inv * xv_in_1 - a00_inv * yv_in_0 - a01_inv * yv_in_1;
            q_lwt_y = 0.f;

            //printf("idx: (%d, %d), yv_in_0: %f, yv_in_1: %f, xv_in_0: %f, xv_in_1: %f, a00_inv: %f, a01_inv: %f, a02_inv: %f, a03_inv: %f, val: %f\n",
            //        data_idx3_x, data_idx3_y, yv_in_0, yv_in_1, xv_in_0, xv_in_1, a00_inv, a01_inv, a02_inv, a03_inv, sm_xv[idx_sm]);
            //printf("idx: (%d, %d), threadIdx.x: %d, a00: %f, a00_inv: %f, a01_inv: %f, a02_inv: %f, a03_inv: %f\n",
            //    data_idx3_x, data_idx3_y, threadIdx.x, a00_inv, a01_inv, a02_inv, a03_inv);
        }
    }
    else
    {
        sm_yv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int idx = data_idx3.y + data_idx3.x * dim_dst_ext.y;
        idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
        {
            q_lwt[idx] = sm_xv[idx_sm] + q_lwt_y;
            //q_lwt[idx] =  q_lwt_y;
        }
        //p_lwt[data_idx3.y + data_idx3.x * dim.y] = p_lwt_y;
        //printf("idx: (%d, %d), proj_i: %f\n", data_idx3.x, data_idx3.y, proj_i);
    }
}

__global__ void cuProjectLocal_q_cross_vel_d(DTYPE* xv, DTYPE* yv, DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim = { 1 << levels.x, 1 << levels.y };
    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    int slice_xy = dim.x * dim.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * (dim.y);
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
        //printf("idx_sm: (%d, %d), idx_gm: (%d, %d), val: %f\n", threadIdx.x, threadIdx.y, data_idx3.y, threadIdx.y - 5, sm_ip_1_y[idx_sm]);
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;

    __syncthreads();

    int idx_xv = (data_idx3.y - 1) + (data_idx3.x << levels.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x&& data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
    int jx = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;

    DTYPE scale_nnn_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_nnn_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_nnn_i_x * scale_nnn_i_x * ip_0_x * ip_1_mid_y + scale_nnn_i_y * scale_nnn_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_nnn_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_nnn_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_nnn_i_y, 2) * ip_1_x * ip_0_m2_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_nnn_i_y, 1) * ip_1_x * ip_0_m1_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_nnn_i_y, 1) * ip_1_x * ip_0_p1_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_nnn_i_y, 2) * ip_1_x * ip_0_p2_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;
        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : -yv_in / scale_nnn_i_x;
    }

    __syncthreads();


    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;
        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;
        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : xv_in / scale_i_y;
    }
    else
    {
        sm_xv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int idx = data_idx3.y + data_idx3.x * dim_ext.y;
        idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
        q_lwt_y += sm_xv[idx_sm];

        int idx_xv = data_idx3.y - 1 + (data_idx3.x << levels.y);
        int idx_yv = data_idx3.y + (data_idx3.x - 1) * (dim.y + 1);

        if (data_idx3.y > 1 && data_idx3.x > 1)
        {
            xv[idx_xv] -= q_lwt_y * scale_nnn_i_y;
            yv[idx_yv] += q_lwt_y * scale_nnn_i_x;
        }

        q_lwt[idx] += q_lwt_y;
        //printf("idx: (%d, %d), q_lwt_y: %f\n", data_idx3.x, data_idx3.y, q_lwt_y);
    }
}

__global__ void cuProjectLocal_q_cross_vel_n(DTYPE* xv, DTYPE* yv, DTYPE* q_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_proj_x, DTYPE* ip_proj_y, int2 levels, DTYPE2 dx_inv, char bc, int l_halo, int r_halo)
{
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.x * write_len, blockIdx.y * write_len,
        blockDim.z * blockIdx.z);
    int2 dim = { 1 << levels.x, 1 << levels.y };
    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    int slice_xy = dim.x * dim.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y - l_halo, thread_start_idx3.x + threadIdx.x - l_halo,
        thread_start_idx3.z);

    extern __shared__ DTYPE sm_ip_1_y[];
    DTYPE* sm_ip_0_y = &sm_ip_1_y[blockDim.x * 5];
    DTYPE* sm_ip_1_x = &sm_ip_1_y[blockDim.x * 10];
    DTYPE* sm_ip_0_x = &sm_ip_1_y[blockDim.x * 15];

    DTYPE* sm_xv = &sm_ip_1_y[blockDim.x * 20];
    DTYPE* sm_yv = &sm_xv[(blockDim.x + 1) * (blockDim.y)];

    int data_idx3_x = thread_start_idx3.y + threadIdx.x - l_halo;
    int data_idx3_y = thread_start_idx3.x + threadIdx.y - l_halo;

    int idx_sm = blockDim.x * threadIdx.y + threadIdx.x;

    if (threadIdx.y < 5)
    {
        if (data_idx3.y >= 0 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_base_y[data_idx3.y + threadIdx.y * dim_ext.y];
        else
            sm_ip_1_y[idx_sm] = 0.f;

    }
    else if (threadIdx.y < 10)
    {
        int idx_gm = data_idx3.y - 1 + (threadIdx.y - 5) * (dim.y);
        if (data_idx3.y >= 1 && data_idx3.y < dim_ext.y)
            sm_ip_1_y[idx_sm] = ip_proj_y[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 15)
    {
        int idx_gm = data_idx3_x + (threadIdx.y - 10) * dim_ext.x;
        if (data_idx3_x >= 0 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_base_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }
    else if (threadIdx.y < 20)
    {
        int idx_gm = data_idx3_x - 1 + (threadIdx.y - 15) * dim.x;
        if (data_idx3_x >= 1 && data_idx3_x < dim_ext.x)
            sm_ip_1_y[idx_sm] = ip_proj_x[idx_gm];
        else
            sm_ip_1_y[idx_sm] = 0.f;
    }


    DTYPE q_lwt_y = 0.f;
    DTYPE q_lwt_1m0 = 0.f;

    __syncthreads();

    int idx_xv = (data_idx3.y - 1) + (data_idx3.x << levels.y);
    int idx_yv = data_idx3.y + (data_idx3.x - 1) * dim_ext.y;

    idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    bool range_p = data_idx3.x >= 0 && data_idx3.x < dim_ext.x&& data_idx3.y >= 0 && data_idx3.y < dim_ext.y;
    if (range_p && data_idx3.y > 0)
    {
        sm_xv[idx_sm] = xv_lwt[idx_xv];
    }
    else
    {
        sm_xv[idx_sm] = 0;
    }
    if (range_p && data_idx3.x > 0)
    {
        sm_yv[idx_sm] = yv_lwt[idx_yv];
    }
    else
    {
        sm_yv[idx_sm] = 0;
    }

    __syncthreads();

    int jy = levels.y - (32 - __clz(max(data_idx3.y - 1, 0))) + 1;
    int jx = levels.x - (32 - __clz(max(data_idx3.x - 1, 0))) + 1;

    DTYPE scale_nnn_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_nnn_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    // y direction
    DTYPE xv_in = sm_xv[idx_sm];
    DTYPE yv_in = sm_yv[idx_sm];

    if (data_idx3.x > 1 && data_idx3.x < dim_ext.x)
    {
        DTYPE ip_0_x = ip_proj_x[data_idx3.x - 1];
        DTYPE ip_1_x = ip_base_x[data_idx3.x];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE ip_mid = scale_nnn_i_x * scale_nnn_i_x * ip_0_x * ip_1_mid_y + scale_nnn_i_y * scale_nnn_i_y * ip_1_x * ip_0_mid_y + eps<DTYPE>;
        DTYPE proj_i = data_idx3.y > 1 ? (xv_in * scale_nnn_i_y * ip_1_x * ip_0_mid_y - yv_in * scale_nnn_i_x * ip_0_x * ip_1_mid_y) / ip_mid : 0.f;
        DTYPE ip_0_m2_y = sm_ip_0_y[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_y = sm_ip_1_y[threadIdx.x + blockDim.x];
        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_m2_y,
            __shfl_up_sync(-1, scale_nnn_i_y, 2) * ip_1_x * ip_0_m2_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_m2 = (jy == __shfl_down_sync(-1, jy, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;


        DTYPE ip_0_m1_y = sm_ip_0_y[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_y = sm_ip_1_y[threadIdx.x + 2 * blockDim.x];
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_m1_y,
            __shfl_up_sync(-1, scale_nnn_i_y, 1) * ip_1_x * ip_0_m1_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_m1 = (jy == __shfl_down_sync(-1, jy, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_y = __shfl_down_sync(-1, ip_0_m1_y, 1);
        DTYPE ip_1_p1_y = __shfl_down_sync(-1, ip_1_m1_y, 1);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_p1_y,
            __shfl_down_sync(-1, scale_nnn_i_y, 1) * ip_1_x * ip_0_p1_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_p1 = (jy == __shfl_up_sync(-1, jy, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_y = __shfl_down_sync(-1, ip_0_m2_y, 2);
        DTYPE ip_1_p2_y = __shfl_down_sync(-1, ip_1_m2_y, 2);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, scale_nnn_i_x * ip_0_x * ip_1_p2_y,
            __shfl_down_sync(-1, scale_nnn_i_y, 2) * ip_1_x * ip_0_p2_y, scale_nnn_i_x, scale_nnn_i_y, proj_i);

        proj_p2 = (jy == __shfl_up_sync(-1, jy, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        q_lwt_y = data_idx3.y > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
    }
    else if (data_idx3.x == 1 || data_idx3.x == 0)
    {
        DTYPE xv_in_0 = sm_xv[idx_sm - data_idx3.x * (blockDim.x + 1)];
        DTYPE xv_in_1 = sm_xv[idx_sm + (1 - data_idx3.x) * (blockDim.x + 1)];

        yv_in = sm_yv[threadIdx.x + (l_halo + 1) * (blockDim.x + 1)];

        DTYPE ip_1_x_00 = ip_base_x[0];
        DTYPE ip_1_x_01 = ip_base_x[(dim.x + 1) * 3];

        DTYPE ip_0_x = ip_proj_x[0];

        DTYPE ip_0_mid_y = sm_ip_0_y[threadIdx.x];
        DTYPE ip_1_mid_y = sm_ip_1_y[threadIdx.x];

        DTYPE scale_i_x_00 = 1.f * dx_inv.x / dim.x;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x_00 * scale_i_x_00 * ip_0_x * ip_1_mid_y + scale_nnn_i_y * scale_nnn_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_mat[1] = -scale_i_x_00 * scale_i_x_00 * ip_0_x * ip_1_mid_y + scale_nnn_i_y * scale_nnn_i_y * ip_1_x_01 * ip_0_mid_y;

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3.x;
        int idx1 = data_idx3.x;

        DTYPE proj_f1 = +yv_in * scale_i_x_00 * ip_0_x * ip_1_mid_y;
        proj_f1 += xv_in_0 * scale_nnn_i_y * ip_1_x_00 * ip_0_mid_y;
        proj_f1 += xv_in_1 * scale_nnn_i_y * ip_1_x_01 * ip_0_mid_y;

        DTYPE proj_f2 = -yv_in * scale_i_x_00 * ip_0_x * ip_1_mid_y;
        proj_f2 += xv_in_0 * scale_nnn_i_y * ip_1_x_01 * ip_0_mid_y;
        proj_f2 += xv_in_1 * scale_nnn_i_y * ip_1_x_00 * ip_0_mid_y;

        //if (data_idx3.y > 1)
        //{
        //    q_lwt_y = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        //    //q_lwt_1m0 = q_lwt_y - (proj_mat[idx2] * proj_f1 + proj_mat[idx1] * proj_f2);
        //}

        q_lwt_y = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        q_lwt_1m0 = q_lwt_y - (proj_mat[idx2] * proj_f1 + proj_mat[idx1] * proj_f2);
        //if (data_idx3.y < 2)
        //{
        //    q_lwt_y -= (proj_mat[idx2] * proj_f1 + proj_mat[idx1] * proj_f2);
        //}

    }

    DTYPE xv_in_remain = data_idx3_y == 1 || data_idx3_y == 0 ? sm_xv[threadIdx.x * (blockDim.x + 1) + (l_halo + 1)] : 0;

    __syncthreads();

    // x direction
    idx_sm = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

    jy = levels.y - (32 - __clz(data_idx3_y - 1)) + 1;
    jx = levels.x - (32 - __clz(data_idx3_x - 1)) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

    xv_in = sm_xv[idx_sm];
    yv_in = sm_yv[idx_sm];

    if (data_idx3_y > 1 && data_idx3_y < dim_ext.y)
    {
        DTYPE ip_0_y = ip_proj_y[data_idx3_y - 1];
        DTYPE ip_1_y = ip_base_y[data_idx3_y];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        DTYPE ip_mid = scale_i_y * scale_i_y * ip_0_y * ip_1_mid_x + scale_i_x * scale_i_x * ip_1_y * ip_0_mid_x + eps<DTYPE>;
        DTYPE proj_i = data_idx3_x > 1 ? (xv_in * scale_i_y * ip_0_y * ip_1_mid_x - yv_in * scale_i_x * ip_1_y * ip_0_mid_x) / ip_mid : 0.f;

        DTYPE ip_0_m2_x = sm_ip_0_x[threadIdx.x + blockDim.x];
        DTYPE ip_1_m2_x = sm_ip_1_x[threadIdx.x + blockDim.x];

        DTYPE proj_m2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 2) * ip_0_m2_x * ip_1_y,
            scale_i_y * ip_1_m2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_m2 = (jx == __shfl_down_sync(-1, jx, 2)) * __shfl_down_sync(-1, proj_m2, 2) / ip_mid;

        DTYPE ip_0_m1_x = sm_ip_0_x[threadIdx.x + 2 * blockDim.x];
        DTYPE ip_1_m1_x = sm_ip_1_x[threadIdx.x + 2 * blockDim.x];
        DTYPE proj_m1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_up_sync(-1, scale_i_x, 1) * ip_0_m1_x * ip_1_y,
            scale_i_y * ip_1_m1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_m1 = (jx == __shfl_down_sync(-1, jx, 1)) * __shfl_down_sync(-1, proj_m1, 1) / ip_mid;


        DTYPE ip_0_p1_x = __shfl_down_sync(-1, ip_0_m1_x, 1);
        DTYPE ip_1_p1_x = __shfl_down_sync(-1, ip_1_m1_x, 1);
        DTYPE proj_p1 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 1) * ip_0_p1_x * ip_1_y,
            scale_i_y * ip_1_p1_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_p1 = (jx == __shfl_up_sync(-1, jx, 1)) * __shfl_up_sync(-1, proj_p1, 1) / ip_mid;


        DTYPE ip_0_p2_x = __shfl_down_sync(-1, ip_0_m2_x, 2);
        DTYPE ip_1_p2_x = __shfl_down_sync(-1, ip_1_m2_x, 2);
        DTYPE proj_p2 = LocalProjFromNei_q(xv_in, yv_in, __shfl_down_sync(-1, scale_i_x, 2) * ip_0_p2_x * ip_1_y,
            scale_i_y * ip_1_p2_x * ip_0_y, scale_i_x, scale_i_y, proj_i);
        proj_p2 = (jx == __shfl_up_sync(-1, jx, 2)) * __shfl_up_sync(-1, proj_p2, 2) / ip_mid;

        sm_xv[idx_sm] = data_idx3_x > 1 ? proj_i * 0.5f + proj_m2 + proj_m1 + proj_p1 + proj_p2 : 0.f;
    }
    else if (data_idx3_y == 1 || data_idx3_y == 0)
    {
        DTYPE yv_in_0 = sm_yv[idx_sm - data_idx3_y];
        DTYPE yv_in_1 = sm_yv[idx_sm + (1 - data_idx3_y)];

        xv_in = xv_in_remain;

        DTYPE ip_1_y_00 = ip_base_y[0];
        DTYPE ip_1_y_01 = ip_base_y[(dim.y + 1) * 3];

        DTYPE ip_0_y = ip_proj_y[0];

        DTYPE ip_0_mid_x = sm_ip_0_x[threadIdx.x];
        DTYPE ip_1_mid_x = sm_ip_1_x[threadIdx.x];

        scale_i_y = 1.f * dx_inv.y / dim.y;

        DTYPE proj_mat[2];
        proj_mat[0] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_00 + scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_mat[1] = scale_i_x * scale_i_x * ip_0_mid_x * ip_1_y_01 - scale_i_y * scale_i_y * ip_1_mid_x * ip_0_y;

        DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
        proj_mat[0] /= proj_inv_div;
        proj_mat[1] /= -proj_inv_div;

        int idx2 = 1 - data_idx3_y;
        int idx1 = data_idx3_y;

        DTYPE ip_0_x_ip_1_00 = ip_0_mid_x * ip_1_y_00;
        DTYPE ip_0_x_ip_1_01 = ip_0_mid_x * ip_1_y_01;
        DTYPE proj_f1 = -xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f1 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_00;
        proj_f1 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_01;

        DTYPE proj_f2 = xv_in * scale_i_y * ip_1_mid_x * ip_0_y;
        proj_f2 -= yv_in_0 * scale_i_x * ip_0_x_ip_1_01;
        proj_f2 -= yv_in_1 * scale_i_x * ip_0_x_ip_1_00;

        if (data_idx3_x > 1)
        {
            sm_xv[idx_sm] = proj_mat[idx1] * proj_f1 + proj_mat[idx2] * proj_f2;
        }
        else
        {
            yv_in_0 = __shfl_sync(0xF, yv_in_0, 3);
            yv_in_1 = __shfl_sync(0xF, yv_in_1, 3);

            DTYPE xv_in_0 = __shfl_sync(0xF, xv_in, 2);
            DTYPE xv_in_1 = __shfl_sync(0xF, xv_in, 3);

            __syncwarp(0xF);

            int idx_inv = max((data_idx3_y * 2 + data_idx3_x) * 4, 0);
            DTYPE a00_inv = div_l0_mat[idx_inv + 0];
            DTYPE a01_inv = div_l0_mat[idx_inv + 1];
            DTYPE a02_inv = div_l0_mat[idx_inv + 2];
            DTYPE a03_inv = div_l0_mat[idx_inv + 3];

            q_lwt_y = a02_inv * xv_in_0 + a03_inv * xv_in_1 - a00_inv * yv_in_0 - a01_inv * yv_in_1;
            //sm_xv[idx_sm] = a02_inv * xv_in_0 + a03_inv * xv_in_1 - a00_inv * yv_in_0 - a01_inv * yv_in_1;
            sm_xv[idx_sm] = q_lwt_y;
            //q_lwt_y = 0.f;
        }
    }
    else
    {
        sm_yv[idx_sm] = 0.f;
    }

    __syncthreads();

    int2 threadIdxOff = { threadIdx.x - l_halo , threadIdx.y - l_halo };
    if (threadIdxOff.x >= 0 && threadIdxOff.x < write_len && threadIdxOff.y >= 0 && threadIdxOff.y < write_len && data_idx3.x < dim_ext.x
        && data_idx3.y < dim_ext.y)
    {
        int mask = __activemask();
        
        DTYPE2 L_inv = { dx_inv.x / dim.x, dx_inv.y / dim.y };

        int idx = data_idx3.y + data_idx3.x * dim_ext.y;
        idx_sm = threadIdx.x + threadIdx.y * (blockDim.x + 1);
        DTYPE q_lwt_x = sm_xv[idx_sm];
        DTYPE qq_lwt = q_lwt_x + ((data_idx3.y < 2) ? 0.f : q_lwt_y);
        q_lwt[idx] += qq_lwt;

        int idx_xv = max(data_idx3.y - 1 + (data_idx3.x << levels.y), 0);
        int idx_yv = max(data_idx3.y + (data_idx3.x - 1) * (dim.y + 1), 0);

        DTYPE xv_res = 0.f;
        DTYPE yv_res = 0.f;

        DTYPE q_lwt_1 = __shfl_up_sync(mask, q_lwt_x, 1);
        DTYPE q_lwt_y_1 = __shfl_up_sync(mask, q_lwt_y, 1);

        // calculate L0L0
        if (data_idx3.y > 1 && data_idx3.x > 1)
        {
            xv_res -= qq_lwt * scale_nnn_i_y;
            yv_res += qq_lwt * scale_nnn_i_x;
        }
        else if (data_idx3.x > 1)  // calculate L01
        {
            if (data_idx3.y == 1)
                xv_res -= (qq_lwt - q_lwt_1) * L_inv.y;
            yv_res += qq_lwt * scale_nnn_i_x;
        }
        else if (data_idx3.y > 1) // calculate 1L0
        {
            xv_res -= scale_nnn_i_y * qq_lwt;

            if (data_idx3.x == 1)
                yv_res += q_lwt_1m0 * L_inv.x;
        }
        else if (data_idx3.x == 1 && data_idx3.y == 1)
        {
            xv_res -= (qq_lwt - q_lwt_1) * L_inv.y;
            yv_res += (qq_lwt - q_lwt_y_1) * L_inv.x;
        }
        else if (data_idx3.x == 0 && data_idx3.y == 1)
        {
            xv_res -= (qq_lwt - q_lwt_y_1) * L_inv.y;
        }
        else if (data_idx3.x == 1 && data_idx3.y == 0)
        {
            yv_res += (qq_lwt - sm_xv[idx_sm - (blockDim.x + 1)]) * L_inv.x;
        }

        if (data_idx3.y > 0)
        {
            xv[idx_xv] += xv_res;
        }
        if (data_idx3.x > 0)
        {
            yv[idx_yv] += yv_res;
        }

    }
}

void CuDivLocalProject::ProjectLocal(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt)
{
//    const int l_halo = 2;
//    const int r_halo = 2;
//
//    const int threads_y = 32;
//    const int threads_x = 32;
//
//    const int efficient_threads_x = threads_x - l_halo - r_halo;
//    const int efficient_threads_y = threads_y - l_halo - r_halo;
//
//    int blocks_y = std::ceil(double(dim_.y) / efficient_threads_y);
//    int blocks_x = std::ceil(double(dim_.x) / efficient_threads_x);
//
//    dim3 grid(blocks_y, blocks_x, 1);
//    dim3 block(threads_y, threads_x, 1);
//    int sm_size = (threads_y * 20 + (threads_x + 1) * threads_y * 2) * sizeof(DTYPE);
//
//#ifdef PRINTF_THREADS
//    printf("threads: %d, %d, %d, sm_size: %d\n", block.x, block.y, block.z, sm_size);
//    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
//#endif
//    if (bc_base_ == 'n')
//    {
//        cuProjectLocal_q_cross_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_, levels_, dx_inv_, bc_base_, l_halo, r_halo);
//
//    }
//    else
//    {
//        cuProjectLocal_q_cross << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_, levels_, dx_inv_, bc_base_, l_halo, r_halo);
//    }
    ProjectLocal(p_lwt, xv_lwt, yv_lwt, levels_);
}

void CuDivLocalProject::ProjectLocal(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt, int2 levels)
{
    const int2 dim = { 1 << levels.x, 1 << levels.y };
    const int2 dim_ext = { dim.x + 1, dim.y + 1 };

    const int l_halo = 2;
    const int r_halo = 2;

    const int threads_y = 32;
    const int threads_x = 32;

    const int efficient_threads_x = threads_x - l_halo - r_halo;
    const int efficient_threads_y = threads_y - l_halo - r_halo;

    int blocks_y = std::ceil(double(dim_ext.y) / efficient_threads_y);
    int blocks_x = std::ceil(double(dim_ext.x) / efficient_threads_x);

    dim3 grid(blocks_y, blocks_x, 1);
    dim3 block(threads_y, threads_x, 1);
    int sm_size = (threads_y * 20 + (threads_x + 1) * threads_y * 2) * sizeof(DTYPE);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm_size: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif
    if (bc_base_ == 'd')
    {
        cuProjectLocal_q_cross << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_,
            dim, levels_, dx_inv_, bc_base_, l_halo, r_halo);
    }
    else
    {
        cuProjectLocal_q_cross_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_,
            dim, levels_, dx_inv_, bc_base_, l_halo, r_halo);
    }
}

void CuDivLocalProject::ProjectSingle(DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt)
{
    int2 levels = levels_;
    const int2 dim = { 1 << levels.x, 1 << levels.y };
    const int2 dim_ext = { dim.x + 1, dim.y + 1 };

    const int l_halo = 2;
    const int r_halo = 2;

    const int threads_y = 32;
    const int threads_x = 32;

    const int efficient_threads_x = threads_x - l_halo - r_halo;
    const int efficient_threads_y = threads_y - l_halo - r_halo;

    int blocks_y = std::ceil(double(dim_ext.y) / efficient_threads_y);
    int blocks_x = std::ceil(double(dim_ext.x) / efficient_threads_x);

    dim3 grid(blocks_y, blocks_x, 1);
    dim3 block(threads_y, threads_x, 1);
    int sm_size = (threads_y * 20 + (threads_x + 1) * threads_y * 2) * sizeof(DTYPE);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm_size: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif
    if (bc_base_ == 'd')
    {
        cuProjectLocal_q_single << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_,
            dim, levels_, dx_inv_, bc_base_, l_halo, r_halo);
    }
    else
    {
        cuProjectLocal_q_single_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_,
            dim, levels_, dx_inv_, bc_base_, l_halo, r_halo);
    }
}

void CuDivLocalProject::ProjectLocal(DTYPE* xv, DTYPE* yv, DTYPE* p_lwt, DTYPE* xv_lwt, DTYPE* yv_lwt)
{
    const int l_halo = 2;
    const int r_halo = 2;

    const int threads_y = 32;
    const int threads_x = 32;

    const int efficient_threads_x = threads_x - l_halo - r_halo;
    const int efficient_threads_y = threads_y - l_halo - r_halo;

    int blocks_y = std::ceil(double(dim_.y) / efficient_threads_y);
    int blocks_x = std::ceil(double(dim_.x) / efficient_threads_x);

    dim3 grid(blocks_y, blocks_x, 1);
    dim3 block(threads_y, threads_x, 1);
    int sm_size = (threads_y * 20 + (threads_x + 1) * threads_y * 2) * sizeof(DTYPE);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm_size: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'n')
    {
        cuProjectLocal_q_cross_vel_n << <grid, block, sm_size >> > (xv, yv, p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_, levels_, dx_inv_, bc_base_, l_halo, r_halo);

    }
    else
    {
        cuProjectLocal_q_cross_vel_d << <grid, block, sm_size >> > (xv, yv, p_lwt, xv_lwt, yv_lwt, ip_base_x_, ip_base_y_, ip_proj_x_, ip_proj_y_, levels_, dx_inv_, bc_base_, l_halo, r_halo);
    }
}