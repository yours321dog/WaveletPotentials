#include "cuDivLocalProject3D.cuh"
#include "cuGetInnerProduct.cuh"
#include "DivL0Matrix.h"
#include "cudaMath.cuh"

__constant__ DTYPE div_l0_mat_zx[16];
__constant__ DTYPE div_l0_mat_xy[16];
__constant__ DTYPE div_l0_mat_yz[16];

CuDivLocalProject3D::CuDivLocalProject3D(int3 dim, int3 levels, DTYPE3 dx, std::string type) : dim_(dim), levels_(levels)
{
    bc_base_ = type[0];
    bc_proj_ = proj_bc(type[0]);
    wt_base_ = type_from_string(type);
    wt_proj_ = proj_WaveletType(wt_base_);
    ip_base_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.x);
    ip_base_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.y);
    ip_base_z_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.z);
    ip_proj_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.x - 1);
    ip_proj_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.y - 1);
    ip_proj_z_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.z - 1);

    proj_coef_xv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'x');
    proj_coef_yv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'y');
    proj_coef_zv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'z');

    proj_len_ = std::pow(proj_len_ + 1., 1. / 3.);
    dx_inv_ = { DTYPE(1.) / dx.x, DTYPE(1.) / dx.y, DTYPE(1.) / dx.z };

    printf("levels: %d, %d, %d, proj_len_: %d\n", levels.x, levels.y, levels.z, proj_len_);

    DTYPE tmp[16];
    GetDivL0Matrix(tmp, { levels.x, levels.y });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_xy, &tmp, 16 * sizeof(DTYPE)));

    GetDivL0Matrix(tmp, { levels.z, levels.x });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_zx, &tmp, 16 * sizeof(DTYPE)));

    GetDivL0Matrix(tmp, { levels.y, levels.z });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_yz, &tmp, 16 * sizeof(DTYPE)));
}

CuDivLocalProject3D::CuDivLocalProject3D(int3 dim, int3 levels, DTYPE3 dx, WaveletType type, char bc) : dim_(dim), levels_(levels)
{
    bc_base_ = bc;
    bc_proj_ = proj_bc(bc);
    wt_base_ = type;
    wt_proj_ = proj_WaveletType(wt_base_);
    ip_base_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.x);
    ip_base_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.y);
    ip_base_z_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_base_, bc_base_, dim.z);
    ip_proj_x_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.x - 1);
    ip_proj_y_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.y - 1);
    ip_proj_z_ = CuGetInnerProduct::GetInstance()->GenerateIp(wt_proj_, bc_proj_, dim.z - 1);

    proj_coef_xv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'x');
    proj_coef_yv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'y');
    proj_coef_zv_ = CuGetInnerProduct::GetInstance()->GenerateProjQCoef(proj_len_, 'z');

    proj_len_ = std::pow(proj_len_ + 1., 1. / 3.);
    dx_inv_ = { DTYPE(1.) / dx.x, DTYPE(1.) / dx.y, DTYPE(1.) / dx.z };

    DTYPE tmp[16];
    GetDivL0Matrix(tmp, { levels.x, levels.y });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_xy, &tmp, 16 * sizeof(DTYPE)));

    GetDivL0Matrix(tmp, { levels.z, levels.x });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_zx, &tmp, 16 * sizeof(DTYPE)));

    GetDivL0Matrix(tmp, { levels.y, levels.z });
    checkCudaErrors(cudaMemcpyToSymbol(div_l0_mat_yz, &tmp, 16 * sizeof(DTYPE)));
}

__global__ void cuProjectLocal_q_3d_d(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, 
    DTYPE* ip_base_x, DTYPE* ip_base_y, DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
    int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
    int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
    int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
    int slice_xy = dim_dst.x * dim_dst.y;

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    int idx_x_0 = data_idx3.x + 1;
    int idx_y_0 = data_idx3.y + 1;
    int idx_z_0 = data_idx3.z + 1;

    int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
    int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
    int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

    int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
    int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
    int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;


    int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
    int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
    int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

    DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
    DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
    DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

    DTYPE ip_0_x = ip_proj_x[data_idx3.x];
    DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

    DTYPE ip_0_y = ip_proj_y[data_idx3.y];
    DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

    DTYPE ip_0_z = ip_proj_z[data_idx3.z];
    DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

    DTYPE xv_in = xv_lwt[idx_xv];
    DTYPE yv_in = yv_lwt[idx_yv];
    DTYPE zv_in = zv_lwt[idx_zv];

    DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y;
    DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y;
    DTYPE sxBxBz = scale_i_x * ip_0_x * ip_1_z;
    DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_z;
    DTYPE syByBz = scale_i_y * ip_0_y * ip_1_z;
    DTYPE szByBz = scale_i_z * ip_1_y * ip_0_z;

    DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
    DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
    DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

    DTYPE proj_i_x = yv_in * syByBz - zv_in * szByBz;
    DTYPE proj_i_y = zv_in * szBxBz - xv_in * sxBxBz;
    DTYPE proj_i_z = xv_in * sxBxBy - yv_in * syBxBy;

    DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;
    DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;
    DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;

    // y direction
    //if (data_idx3.x > 0  && data_idx3.z > 0)
    {
        q_lwt_y[idx_qy] = (data_idx3.x > 0 && data_idx3.z > 0) * lwt_s_y;
    }
    //if (data_idx3.y > 0 && data_idx3.z > 0)
    {
        q_lwt_x[idx_qx] = (data_idx3.y > 0 && data_idx3.z > 0) * lwt_s_x;
    }
    //if (data_idx3.x > 0 && data_idx3.y > 0)
    {
        q_lwt_z[idx_qz] = (data_idx3.x > 0 && data_idx3.y > 0) * lwt_s_z;
    }
}

__device__ __forceinline__ DTYPE2 cuCalculateP0Val_single(const DTYPE& xv_in_0, const DTYPE& xv_in_1, const DTYPE& yv_in,
    const DTYPE& ip_1_x_00, const DTYPE& ip_1_x_01, const DTYPE& ip_0_x, const DTYPE& ip_1_y,
    const DTYPE& ip_0_y, const DTYPE& scale_i_x, const DTYPE& scale_i_y)
{
    //DTYPE proj_mat[2];
    //proj_mat[0] = scale_i_x * scale_i_x * ip_0_x * ip_1_y + scale_i_y * scale_i_y * ip_1_x_00 * ip_0_y;
    //proj_mat[1] = -scale_i_x * scale_i_x * ip_0_x * ip_1_y + scale_i_y * scale_i_y * ip_1_x_01 * ip_0_y;

    //DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
    //proj_mat[0] /= proj_inv_div;
    //proj_mat[1] /= -proj_inv_div;

    //DTYPE proj_f1 = -yv_in * scale_i_x * ip_0_x * ip_1_y;
    //proj_f1 += xv_in_0 * scale_i_y * ip_1_x_00 * ip_0_y;
    //proj_f1 += xv_in_1 * scale_i_y * ip_1_x_01 * ip_0_y;

    //DTYPE proj_f2 = yv_in * scale_i_x * ip_0_x * ip_1_y;
    //proj_f2 += xv_in_0 * scale_i_y * ip_1_x_01 * ip_0_y;
    //proj_f2 += xv_in_1 * scale_i_y * ip_1_x_00 * ip_0_y;

    //return { proj_mat[0] * proj_f1 + proj_mat[1] * proj_f2, proj_mat[1] * proj_f1 + proj_mat[0] * proj_f2 };
    return { xv_in_0 / scale_i_y, xv_in_1 / scale_i_y };
}

__device__ __forceinline__ DTYPE2 cuCalculateP0Val(const DTYPE& xv_in_0, const DTYPE& xv_in_1, const DTYPE& yv_in, 
    const DTYPE& ip_1_x_00, const DTYPE& ip_1_x_01, const DTYPE& ip_0_x, const DTYPE& ip_1_y, 
    const DTYPE& ip_0_y, const DTYPE& scale_i_x, const DTYPE& scale_i_y)
{
    DTYPE proj_mat[2];
    proj_mat[0] = scale_i_x * scale_i_x * ip_0_x * ip_1_y + scale_i_y * scale_i_y * ip_1_x_00 * ip_0_y;
    proj_mat[1] = -scale_i_x * scale_i_x * ip_0_x * ip_1_y + scale_i_y * scale_i_y * ip_1_x_01 * ip_0_y;

    DTYPE proj_inv_div = proj_mat[0] * proj_mat[0] - proj_mat[1] * proj_mat[1] + eps<DTYPE>;
    proj_mat[0] /= proj_inv_div;
    proj_mat[1] /= -proj_inv_div;

    DTYPE proj_f1 = -yv_in * scale_i_x * ip_0_x * ip_1_y;
    proj_f1 += xv_in_0 * scale_i_y * ip_1_x_00 * ip_0_y;
    proj_f1 += xv_in_1 * scale_i_y * ip_1_x_01 * ip_0_y;

    DTYPE proj_f2 = yv_in * scale_i_x * ip_0_x * ip_1_y;
    proj_f2 += xv_in_0 * scale_i_y * ip_1_x_01 * ip_0_y;
    proj_f2 += xv_in_1 * scale_i_y * ip_1_x_00 * ip_0_y;

    return { proj_mat[0] * proj_f1 + proj_mat[1] * proj_f2, proj_mat[1] * proj_f1 + proj_mat[0] * proj_f2 };
    //return { xv_in_0 / scale_i_y, xv_in_1 / scale_i_y };
}

__global__ void cuProjectLocal_q_3d_n(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y, 
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        //int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        //int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        //int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;
        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };
        DTYPE2 L_inv_x = { L_inv.y / max(L_inv.y, L_inv.z), L_inv.z / max(L_inv.y, L_inv.z), };
        DTYPE2 L_inv_y = { L_inv.x / max(L_inv.x, L_inv.z), L_inv.z / max(L_inv.x, L_inv.z), };
        DTYPE2 L_inv_z = { L_inv.x / max(L_inv.x, L_inv.y), L_inv.y / max(L_inv.x, L_inv.y), };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        //DTYPE sxBxBy = scale_i_x;
        //DTYPE syBxBy = scale_i_y;
        //DTYPE sxBxBz = scale_i_x;
        //DTYPE szBxBz = scale_i_z;
        //DTYPE syByBz = scale_i_y;
        //DTYPE szByBz = scale_i_z;

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        /*printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f\n", data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in);*/

        //if (data_idx3.z == 1)
        //printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f, lwt_s_x: %f, scale_i_x: %f, scale_i_y: %f, scale_i_z: %f\n", 
        //    data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in, lwt_s_x, scale_i_x, scale_i_y, scale_i_z);
        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;

            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            //DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y_fast
            //DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;
            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;

            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;
            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            //xv_lwt[idx_xv - dim_dst.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv.z;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv.z;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv.x;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv.y;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;


            //printf("xv: %f, %f, zv: %f, %f, yv: %f, %f, L_inv: %f, %f, %f\n", xv_in_0, yv_in_0, zv_in_0, zv_in_1, yv_in_0, yv_in_1, L_inv.x, L_inv.y, L_inv.z);

            // x fast
            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            //printf("lwt_s_x_0_4: %f, %f, %f, %f, zv: %f, %f, yv: %f, %f, \n", lwt_s_x_0, lwt_s_x_1, lwt_s_x_2, lwt_s_x_3, zv_in_0, zv_in_1, yv_in_0, yv_in_1);

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            //printf("lwt_s_y_0_4: %f, %f, %f, %f, zv: %f, %f, xv: %f, %f\n", lwt_s_y_0, lwt_s_y_1, lwt_s_y_2, lwt_s_y_3, zv_in_0, zv_in_1, xv_in_0, xv_in_1);

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            //printf("lwt_s_z_0_4: %f, %f, %f, %f, yv: %f, %f, xv: %f, %f\n", lwt_s_z_0, lwt_s_z_1, lwt_s_z_2, lwt_s_z_3, yv_in_0, yv_in_1, xv_in_0, xv_in_1);
            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        //int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        //int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        //int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;


        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        /*printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f\n", data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in);*/

        //DTYPE sxBxBy = scale_i_x;
        //DTYPE syBxBy = scale_i_y;
        //DTYPE sxBxBz = scale_i_x;
        //DTYPE szBxBz = scale_i_z;
        //DTYPE syByBz = scale_i_y;
        //DTYPE szByBz = scale_i_z;

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
        DTYPE lwt_s_x = (data_idx3.y > 0 && data_idx3.z > 0) * proj_i_x * ip_mid_x_inv;
        //if (data_idx3.z == 1)
        //printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f, lwt_s_x: %f, scale_i_x: %f, scale_i_y: %f, scale_i_z: %f\n", 
        //    data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in, lwt_s_x, scale_i_x, scale_i_y, scale_i_z);
        {
            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;
        }

        DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE lwt_s_y = (data_idx3.x > 0 && data_idx3.z > 0) * proj_i_y * ip_mid_y_inv;
        {
            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;
        }

        DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
        DTYPE lwt_s_z = (data_idx3.x > 0 && data_idx3.y > 0) * proj_i_z * ip_mid_z_inv;
        if (data_idx3.x > 0 && data_idx3.y > 0)
        {
            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;
        }

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* proj_coef_xv, DTYPE* proj_coef_yv, 
    DTYPE* proj_coef_zv, int3 dim, int3 levels, DTYPE3 dx_inv, int pc_len)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    DTYPE scale_base_inv = 0.5f / dx_inv.x;
    int pc_size = pc_len * pc_len * pc_len;

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        int idx_jy = data_idx3.y > 0 ? pc_len - jy : 0;
        int idx_jx = data_idx3.x > 0 ? pc_len - jx : 0;
        int idx_jz = data_idx3.z > 0 ? pc_len - jz : 0;
        int idx_ll = idx_jy + idx_jx * pc_len + idx_jz * pc_len * pc_len;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE pc_xv = proj_coef_xv[idx_ll];
        DTYPE pc_yv = proj_coef_yv[idx_ll];
        DTYPE pc_zv = proj_coef_zv[idx_ll];
        DTYPE lwt_s_x = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        pc_xv = proj_coef_xv[idx_ll + pc_size];
        pc_yv = proj_coef_yv[idx_ll + pc_size];
        pc_zv = proj_coef_zv[idx_ll + pc_size];
        DTYPE lwt_s_y = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        int jy_l = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
        int jx_l = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;
        int jz_l = levels.z - (32 - __clz(data_idx3.z - 1)) + 1;

        DTYPE xv_nei = 0.f;
        DTYPE yv_nei = 0.f;
        DTYPE zv_nei = 0.f;

        int jy_r = levels.y - (32 - __clz(data_idx3.y + 1)) + 1;
        int jx_r = levels.x - (32 - __clz(data_idx3.x + 1)) + 1;
        int jz_r = levels.z - (32 - __clz(data_idx3.z + 1)) + 1;

        if (jy_l == jy)
        {
            xv_nei = xv_lwt[idx_xv - 1];
            yv_nei = yv_lwt[idx_yv - 1];
            zv_nei = zv_lwt[idx_zv - 1];

            pc_xv = proj_coef_xv[idx_ll + 2 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 2 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 2 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 3 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 3 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 3 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jy_r == jy)
        {
            xv_nei = xv_lwt[idx_xv + 1];
            yv_nei = yv_lwt[idx_yv + 1];
            zv_nei = zv_lwt[idx_zv + 1];

            pc_xv = proj_coef_xv[idx_ll + 4 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 4 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 4 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 5 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 5 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 5 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_l == jx)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv - dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 6 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 6 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 6 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 7 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 7 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 7 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_r == jx)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv + dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 8 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 8 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 8 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 9 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 9 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 9 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_l == jz)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv - slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 10 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 10 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 10 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 11 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 11 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 11 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_r == jz)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv + slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 12 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 12 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 12 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 13 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 13 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 13 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }


        pc_xv = proj_coef_xv[idx_ll + 14 * pc_size];
        pc_yv = proj_coef_yv[idx_ll + 14 * pc_size];
        pc_zv = proj_coef_zv[idx_ll + 14 * pc_size];
        DTYPE lwt_s_z = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ttt(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* proj_coef_xv, DTYPE* proj_coef_yv,
    DTYPE* proj_coef_zv, int3 dim, int3 levels, DTYPE3 dx_inv, int pc_len)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    DTYPE scale_base_inv = 0.5f / dx_inv.x;
    int pc_size = pc_len * pc_len * pc_len;

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        int idx_jy = data_idx3.y > 0 ? pc_len - jy : 0;
        int idx_jx = data_idx3.x > 0 ? pc_len - jx : 0;
        int idx_jz = data_idx3.z > 0 ? pc_len - jz : 0;
        int idx_ll = idx_jy + idx_jx * pc_len + idx_jz * pc_len * pc_len;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE pc_xv = proj_coef_xv[idx_ll];
        DTYPE pc_yv = proj_coef_yv[idx_ll];
        DTYPE pc_zv = proj_coef_zv[idx_ll];
        DTYPE lwt_s_x = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        pc_xv = proj_coef_xv[idx_ll + pc_size];
        pc_yv = proj_coef_yv[idx_ll + pc_size];
        pc_zv = proj_coef_zv[idx_ll + pc_size];
        DTYPE lwt_s_y = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        int jy_l = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
        int jx_l = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;
        int jz_l = levels.z - (32 - __clz(data_idx3.z - 1)) + 1;

        DTYPE xv_nei = 0.f;
        DTYPE yv_nei = 0.f;
        DTYPE zv_nei = 0.f;

        int jy_r = levels.y - (32 - __clz(data_idx3.y + 1)) + 1;
        int jx_r = levels.x - (32 - __clz(data_idx3.x + 1)) + 1;
        int jz_r = levels.z - (32 - __clz(data_idx3.z + 1)) + 1;

        if (jy_l == jy)
        {
            xv_nei = xv_lwt[idx_xv - 1];
            yv_nei = yv_lwt[idx_yv - 1];
            zv_nei = zv_lwt[idx_zv - 1];

            pc_xv = proj_coef_xv[idx_ll + 2 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 2 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 2 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 3 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 3 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 3 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jy_r == jy)
        {
            xv_nei = xv_lwt[idx_xv + 1];
            yv_nei = yv_lwt[idx_yv + 1];
            zv_nei = zv_lwt[idx_zv + 1];

            pc_xv = proj_coef_xv[idx_ll + 4 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 4 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 4 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 5 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 5 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 5 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_l == jx)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv - dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 6 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 6 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 6 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 7 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 7 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 7 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_r == jx)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv + dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 8 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 8 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 8 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 9 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 9 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 9 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_l == jz)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv - slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 10 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 10 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 10 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 11 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 11 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 11 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_r == jz)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv + slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 12 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 12 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 12 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 13 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 13 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 13 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }


        pc_xv = proj_coef_xv[idx_ll + 14 * pc_size];
        pc_yv = proj_coef_yv[idx_ll + 14 * pc_size];
        pc_zv = proj_coef_zv[idx_ll + 14 * pc_size];
        DTYPE lwt_s_z = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        yv[idx_yv] -= scale_i_z * lwt_s_x;
        zv[idx_zv] += scale_i_y * lwt_s_x;

        xv[idx_xv] += scale_i_z * lwt_s_y;
        zv[idx_zv] -= scale_i_x * lwt_s_y;

        xv[idx_xv] -= scale_i_y * lwt_s_z;
        yv[idx_yv] += scale_i_x * lwt_s_z;

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* proj_coef_xv, DTYPE* proj_coef_yv,
    DTYPE* proj_coef_zv, DTYPE* ip_base_x, DTYPE* ip_base_y, DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv, int pc_len)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    DTYPE scale_base_inv = 0.5f / dx_inv.x;
    int pc_size = pc_len * pc_len * pc_len;

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        int idx_jy = data_idx3.y > 0 ? pc_len - jy : 0;
        int idx_jx = data_idx3.x > 0 ? pc_len - jx : 0;
        int idx_jz = data_idx3.z > 0 ? pc_len - jz : 0;
        int idx_ll = idx_jy + idx_jx * pc_len + idx_jz * pc_len * pc_len;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE pc_xv = proj_coef_xv[idx_ll];
        DTYPE pc_yv = proj_coef_yv[idx_ll];
        DTYPE pc_zv = proj_coef_zv[idx_ll];
        DTYPE lwt_s_x = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        pc_xv = proj_coef_xv[idx_ll + pc_size];
        pc_yv = proj_coef_yv[idx_ll + pc_size];
        pc_zv = proj_coef_zv[idx_ll + pc_size];
        DTYPE lwt_s_y = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        int jy_l = levels.y - (32 - __clz(data_idx3.y - 1)) + 1;
        int jx_l = levels.x - (32 - __clz(data_idx3.x - 1)) + 1;
        int jz_l = levels.z - (32 - __clz(data_idx3.z - 1)) + 1;

        DTYPE xv_nei = 0.f;
        DTYPE yv_nei = 0.f;
        DTYPE zv_nei = 0.f;

        int jy_r = levels.y - (32 - __clz(data_idx3.y + 1)) + 1;
        int jx_r = levels.x - (32 - __clz(data_idx3.x + 1)) + 1;
        int jz_r = levels.z - (32 - __clz(data_idx3.z + 1)) + 1;

        if (jy_l == jy)
        {
            xv_nei = xv_lwt[idx_xv - 1];
            yv_nei = yv_lwt[idx_yv - 1];
            zv_nei = zv_lwt[idx_zv - 1];

            pc_xv = proj_coef_xv[idx_ll + 2 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 2 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 2 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 3 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 3 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 3 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jy_r == jy)
        {
            xv_nei = xv_lwt[idx_xv + 1];
            yv_nei = yv_lwt[idx_yv + 1];
            zv_nei = zv_lwt[idx_zv + 1];

            pc_xv = proj_coef_xv[idx_ll + 4 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 4 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 4 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 5 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 5 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 5 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_l == jx)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv - dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 6 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 6 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 6 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 7 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 7 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 7 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jx_r == jx)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y];
            zv_nei = zv_lwt[idx_zv + dim_dst.y];

            pc_xv = proj_coef_xv[idx_ll + 8 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 8 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 8 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 9 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 9 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 9 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_l == jz)
        {
            xv_nei = xv_lwt[idx_xv - dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv - dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv - slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 10 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 10 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 10 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 11 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 11 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 11 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }

        if (jz_r == jz)
        {
            xv_nei = xv_lwt[idx_xv + dim_dst.y * dim_dst_ext.x];
            yv_nei = yv_lwt[idx_yv + dim_dst_ext.y * dim_dst.x];
            zv_nei = zv_lwt[idx_zv + slice_xy];

            pc_xv = proj_coef_xv[idx_ll + 12 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 12 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 12 * pc_size];

            lwt_s_x += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;

            pc_xv = proj_coef_xv[idx_ll + 13 * pc_size];
            pc_yv = proj_coef_yv[idx_ll + 13 * pc_size];
            pc_zv = proj_coef_zv[idx_ll + 13 * pc_size];

            lwt_s_y += (xv_nei * pc_xv + yv_nei * pc_yv + zv_nei * pc_zv) * scale_base_inv;
        }


        pc_xv = proj_coef_xv[idx_ll + 14 * pc_size];
        pc_yv = proj_coef_yv[idx_ll + 14 * pc_size];
        pc_zv = proj_coef_zv[idx_ll + 14 * pc_size];
        DTYPE lwt_s_z = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ttt_old(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* proj_coef_xv, DTYPE* proj_coef_yv,
    DTYPE* proj_coef_zv, int3 dim, int3 levels, DTYPE3 dx_inv, int pc_len)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    DTYPE scale_base_inv = 0.5f / dx_inv.x;
    int pc_size = pc_len * pc_len * pc_len;

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        int idx_jy = data_idx3.y > 0 ? pc_len - jy : 0;
        int idx_jx = data_idx3.x > 0 ? pc_len - jx : 0;
        int idx_jz = data_idx3.z > 0 ? pc_len - jz : 0;
        int idx_ll = idx_jy + idx_jx * pc_len + idx_jz * pc_len * pc_len;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE pc_xv = proj_coef_xv[idx_ll];
        DTYPE pc_yv = proj_coef_yv[idx_ll];
        DTYPE pc_zv = proj_coef_zv[idx_ll];
        DTYPE lwt_s_x = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        pc_xv = proj_coef_xv[idx_ll + pc_size];
        pc_yv = proj_coef_yv[idx_ll + pc_size];
        pc_zv = proj_coef_zv[idx_ll + pc_size];
        DTYPE lwt_s_y = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        pc_xv = proj_coef_xv[idx_ll + 2 * pc_size];
        pc_yv = proj_coef_yv[idx_ll + 2 * pc_size];
        pc_zv = proj_coef_zv[idx_ll + 2 * pc_size];
        DTYPE lwt_s_z = (xv_in * pc_xv + yv_in * pc_yv + zv_in * pc_zv) * scale_base_inv;

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * (dim_dst.y + 1) + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        //int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        //int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        //int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;


        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];


        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;

        DTYPE fx = yv_in * szByBz - zv_in * syByBz;
        DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE fz = xv_in * syBxBy - yv_in * sxBxBy;

        if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        {
            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            proj_mat[2] = -scale_i_y * sxBxBz;
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;
        }
        else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        {
            //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x = fx * ip_mid_x_inv;

        }
        else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        {
            //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y = fy * ip_mid_y_inv;

        }
        else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        {
            //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z = fz * ip_mid_z_inv;

        }

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_n_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        //int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        //int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        //int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;
        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };
        DTYPE2 L_inv_x = { L_inv.y / max(L_inv.y, L_inv.z), L_inv.z / max(L_inv.y, L_inv.z), };
        DTYPE2 L_inv_y = { L_inv.x / max(L_inv.x, L_inv.z), L_inv.z / max(L_inv.x, L_inv.z), };
        DTYPE2 L_inv_z = { L_inv.x / max(L_inv.x, L_inv.y), L_inv.y / max(L_inv.x, L_inv.y), };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        //DTYPE sxBxBy = scale_i_x;
        //DTYPE syBxBy = scale_i_y;
        //DTYPE sxBxBz = scale_i_x;
        //DTYPE szBxBz = scale_i_z;
        //DTYPE syByBz = scale_i_y;
        //DTYPE szByBz = scale_i_z;

        //DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        //DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        //DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        //DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        //DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        //DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        //DTYPE ip_mid_x_inv = 1.f / (scale_i_y * scale_i_y + scale_i_z * scale_i_z + eps<DTYPE>);
        //DTYPE ip_mid_y_inv = 1.f / (scale_i_x * scale_i_x + scale_i_z * scale_i_z + eps<DTYPE>);
        //DTYPE ip_mid_z_inv = 1.f / (scale_i_x * scale_i_x + scale_i_y * scale_i_y + eps<DTYPE>);

        /*printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f\n", data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in);*/

        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            proj_mat[2] = proj_mat[1];
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            fx = yv_in * szByBz - zv_in * syByBz;
            fy = zv_in * sxBxBz - xv_in * szBxBz;

            lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            ////DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //zv_in_0 += scale_i_y * lwt_s_x.x;
            //zv_in_1 += scale_i_y * lwt_s_x.y;
            //yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y_fast
            //DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;
            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;


            //lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //zv_in_0 += scale_i_y * lwt_s_x.x;
            //zv_in_1 += scale_i_y * lwt_s_x.y;
            //yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            //// y_fast
            ////DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            //lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            //zv_in_0 -= scale_i_x * lwt_s_y.x;
            //zv_in_1 -= scale_i_x * lwt_s_y.y;
            //xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            //// z_fast
            //proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            //lwt_s_z = proj_i_z * ip_mid_z_inv;
            //xv_in -= scale_i_y * lwt_s_z;
            //yv_in += scale_i_x * lwt_s_z;



            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            //// z fast
            //DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            //yv_in_0 += scale_i_x * lwt_s_z.x;
            //yv_in_1 += scale_i_x * lwt_s_z.y;
            //xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            //DTYPE2 lwt_s_y = cuCalculateP0Val_single(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            //xv_in_0 += scale_i_z * lwt_s_y.x;
            //xv_in_1 += scale_i_z * lwt_s_y.y;
            //zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;
            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            //DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            ////zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
            ////zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
            ////yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
            ////yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            //lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            //zv_in_0 -= scale_i_x * lwt_s_y.x;
            //zv_in_1 -= scale_i_x * lwt_s_y.y;
            //xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            //// z fast
            //lwt_s_z = cuCalculateP0Val_single(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            //yv_in_0 += scale_i_x * lwt_s_z.x;
            //yv_in_1 += scale_i_x * lwt_s_z.y;
            //xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            //lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            ////zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
            ////zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
            ////yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
            ////yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            //// y fast
            //DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            ////xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv.z;
            ////xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv.z;
            ////zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv.x;
            ////zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv.x;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;



            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            //// z fast
            //DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            ////yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv.x;
            ////yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv.x;
            ////xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv.y;
            ////xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv.y;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE b = -((zv_in_1 - zv_in_0) / L_inv.z + (yv_in_1 - yv_in_0) / L_inv.y + (xv_in_1 - xv_in_0) / L_inv.x);

            DTYPE p = b / 12.f;
            zv_in_1 += 2.f * p;
            zv_in_0 -= 2.f * p;
            xv_in_1 += 2.f * p;
            xv_in_0 -= 2.f * p;
            yv_in_1 += 2.f * p;
            yv_in_0 -= 2.f * p;

            DTYPE lwt_s_x_000 = 0.f;
            DTYPE lwt_s_x_010 = lwt_s_x_000 - zv_in_0 / L_inv.y;
            DTYPE lwt_s_x_001 = lwt_s_x_000 + yv_in_0 / L_inv.z;
            DTYPE lwt_s_x_011 = lwt_s_x_010 + yv_in_1 / L_inv.z;

            DTYPE lwt_s_y_000 = 0.f;
            DTYPE lwt_s_y_100 = 0.f;
            DTYPE lwt_s_y_001 = lwt_s_y_000 - xv_in_0 / L_inv.z;
            DTYPE lwt_s_y_101 = lwt_s_y_100 - xv_in_1 / L_inv.z;

            zv_in_0 += (lwt_s_x_010 - lwt_s_x_000) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_011 - lwt_s_x_001) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_001 - lwt_s_x_000) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_011 - lwt_s_x_010) * L_inv_x.y;

            xv_in_0 += (lwt_s_y_001 - lwt_s_y_000) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_101 - lwt_s_y_100) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_100 - lwt_s_y_000) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_101 - lwt_s_y_001) * L_inv_y.x;

            //// x fast
            //DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            ////printf("lwt_s_x_0_4: %f, %f, %f, %f, zv: %f, %f, yv: %f, %f, \n", lwt_s_x_0, lwt_s_x_1, lwt_s_x_2, lwt_s_x_3, zv_in_0, zv_in_1, yv_in_0, yv_in_1);

            //// y fast
            //DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            ////printf("lwt_s_y_0_4: %f, %f, %f, %f, zv: %f, %f, xv: %f, %f\n", lwt_s_y_0, lwt_s_y_1, lwt_s_y_2, lwt_s_y_3, zv_in_0, zv_in_1, xv_in_0, xv_in_1);

            //// z fast
            //DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            //// x fast
            //lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            //// y fast
            //lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            //// z fast
            //lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            ////// x fast
            ////lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            ////lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            ////lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            ////lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            ////zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            ////zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            ////yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            ////yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            ////// y fast
            ////lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            ////lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            ////lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            ////lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            ////xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            ////xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            ////zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            ////zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            ////// z fast
            ////lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            ////lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            ////lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            ////lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            ////yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            ////yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            ////xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            ////xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_n_ccc_old(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        //int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        //int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        //int idx_qz = idx_y_0 + data_idx3.x * dim_dst_ext.y + data_idx3.z * slice_xy_ext;
        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };
        DTYPE2 L_inv_x = { L_inv.y / max(L_inv.y, L_inv.z), L_inv.z / max(L_inv.y, L_inv.z), };
        DTYPE2 L_inv_y = { L_inv.x / max(L_inv.x, L_inv.z), L_inv.z / max(L_inv.x, L_inv.z), };
        DTYPE2 L_inv_z = { L_inv.x / max(L_inv.x, L_inv.y), L_inv.y / max(L_inv.x, L_inv.y), };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        //DTYPE sxBxBy = scale_i_x;
        //DTYPE syBxBy = scale_i_y;
        //DTYPE sxBxBz = scale_i_x;
        //DTYPE szBxBz = scale_i_z;
        //DTYPE syByBz = scale_i_y;
        //DTYPE szByBz = scale_i_z;

        //DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        //DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        //DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        //DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        //DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        //DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        //DTYPE ip_mid_x_inv = 1.f / (scale_i_y * scale_i_y + scale_i_z * scale_i_z + eps<DTYPE>);
        //DTYPE ip_mid_y_inv = 1.f / (scale_i_x * scale_i_x + scale_i_z * scale_i_z + eps<DTYPE>);
        //DTYPE ip_mid_z_inv = 1.f / (scale_i_x * scale_i_x + scale_i_y * scale_i_y + eps<DTYPE>);

        /*printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f\n", data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in);*/

        //if (data_idx3.z == 1)
        //printf("idx: (%d, %d, %d), xv: %f, yv: %f, zv: %f, lwt_s_x: %f, scale_i_x: %f, scale_i_y: %f, scale_i_z: %f\n", 
        //    data_idx3.x, data_idx3.y, data_idx3.z, xv_in, yv_in, zv_in, lwt_s_x, scale_i_x, scale_i_y, scale_i_z);
        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            //DTYPE szByBz = scale_i_z * B0B1B0;
            //DTYPE syByBz = scale_i_y * B0B0B1;

            //DTYPE sxBxBz = scale_i_x * B0B0B1;
            //DTYPE szBxBz = scale_i_z * B1B0B0;

            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            proj_mat[2] = proj_mat[1];
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;


        }
        //else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0   ** x-z plane
        //{

        //    //DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);
        //    //DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

        //    //DTYPE den_j2j3 = num_j2 * num_j2 + num_j3 * num_j3;

        //    //int idx_yv_1 = idx_yv - 1;
        //    //DTYPE yv_in_1 = yv_lwt[idx_yv_1];

        //    //// x_fast
        //    //DTYPE lwt_s_x_1 = 1.f / num_j3 * yv_in_1;
        //    //DTYPE lwt_s_x_end = 1.f / num_j3 * yv_in;
        //    //yv_in = 0.;
        //    //yv_lwt[idx_yv_1] = 0.;
        //    //zv_in += (lwt_s_x_end - lwt_s_x_1) * L_inv.y;
        //    //lwt_s_x = lwt_s_x_end;

        //    //// y_fast
        //    //lwt_s_y = num_j2 / den_j2j3 * zv_in - num_j3 / den_j2j3 * xv_in;
        //    //xv_in += num_j3 * lwt_s_y;
        //    //zv_in -= num_j2 * lwt_s_y;

        //    DTYPE szByBz = scale_i_z * B0B1B0;
        //    DTYPE syByBz = scale_i_y * B0B0B1;

        //    DTYPE sxBxBz = scale_i_x * B0B0B1;
        //    DTYPE szBxBz = scale_i_z * B1B0B0;

        //    DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);

        //    // x fast
        //    DTYPE& ip_1_y_00 = ip_1_y;
        //    DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
        //    DTYPE scale_i_y_0 = L_inv.y;
        //    DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    DTYPE& yv_in_1 = yv_in;

        //    DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
        //    yv_in_0 -= scale_i_z * lwt_s_x.x;
        //    yv_in_1 -= scale_i_z * lwt_s_x.y;
        //    zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

        //    // y fast
        //    DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
        //    DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

        //    xv_in += scale_i_z * lwt_s_y;
        //    zv_in -= scale_i_x * lwt_s_y;

        //    // z fast
        //    DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
        //    yv_in_0 += scale_i_x * lwt_s_z.x;
        //    yv_in_1 += scale_i_x * lwt_s_z.y;
        //    xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

        //    yv_lwt[idx_yv - 1] = yv_in_0;

        //    // // x fast
        //    //DTYPE& ip_1_y_00 = ip_1_y;
        //    //DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
        //    //DTYPE scale_i_y_0 = -2.f * L_inv.y;
        //    //DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    //DTYPE& yv_in_1 = yv_in;

        //    //DTYPE B0B1_01B0 = ip_0_x * ip_1_y_01 * ip_0_z;

        //    //DTYPE fx = (yv_in_0 - yv_in_1) * scale_i_z * (B0B1B0 - B0B1_01B0) - zv_in * scale_i_y_0 * B0B0B1;
        //    //DTYPE fy = -xv_in * scale_i_z * B1B0B0 + zv_in * scale_i_x * B0B0B1;

        //    //DTYPE proj_mat[4];
        //    //proj_mat[0] = scale_i_z * scale_i_z * 2.f * (B0B1B0 - B0B1_01B0) + scale_i_y_0 * scale_i_y_0 * B0B0B1;
        //    //proj_mat[1] = -scale_i_x * scale_i_y_0 * B0B0B1;
        //    //proj_mat[2] = proj_mat[1];
        //    //proj_mat[3] = scale_i_z * scale_i_z * B1B0B0 + scale_i_x * scale_i_x * B0B0B1;

        //    //cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

        //    //DTYPE lwt_s_x_0 = proj_mat[0] * fx + proj_mat[1] * fy;
        //    //DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

        //    //yv_in_0 -= scale_i_z * lwt_s_x_0;
        //    //yv_in_1 -= -scale_i_z * lwt_s_x_0;

        //    //zv_in += -2.f * lwt_s_x_0 * L_inv.y - scale_i_x * lwt_s_y;
        //    //xv_in += scale_i_z * lwt_s_y;

        //    //yv_lwt[idx_yv - 1] = yv_in_0;


        //    //// z_fast
        //    //DTYPE lwt_s_z_1 = -1.f / num_j2 * yv_in;
        //    //DTYPE lwt_s_z_end = -1.f / num_j2 * yv_in_end;
        //    //yv_in = 0.;
        //    //yv_in_end = 0.;
        //    //xv_in -= (lwt_s_z_end - lwt_s_z_1) / Ly;

        //}
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            //DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y_fast
            //DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;
            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;

            zv_lwt[idx_zv - slice_xy] = zv_in_0;

            //DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            //DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);

            //DTYPE den_j1j2 = num_j1 * num_j1 + num_j2 * num_j2;

            //int idx_zv_1 = idx_zv - slice_xy;
            //DTYPE zv_in_1 = zv_lwt[idx_zv_1];


            //// x_fast
            //DTYPE lwt_s_x_1 = -1.f / num_j1 * zv_in_1;
            //DTYPE lwt_s_x_end = -1.f / num_j1 * zv_in;
            ////DTYPE lwt_s_x_1 = -zv_in_1 / scale_i_y;
            ////DTYPE lwt_s_x_end = -zv_in / scale_i_y;
            //zv_in = 0.f;
            //zv_lwt[idx_zv_1] = 0.f;
            //yv_in -= (lwt_s_x_end - lwt_s_x_1) * L_inv.z;
            //// y_fast


            //// z_fast
            //DTYPE lwt_s_z = num_j1 / den_j1j2 * xv_in - num_j2 / den_j1j2 * yv_in;
            //xv_in -= num_j1 * lwt_s_z;
            //yv_in += num_j2 * lwt_s_z;


            //int idx_zv_1 = idx_zv - slice_xy;
            //DTYPE zv_in_1 = zv_lwt[idx_zv_1];


            ////// x_fast
            //DTYPE lwt_s_x_1 = -zv_in_1 / scale_i_y;
            //DTYPE lwt_s_x_end = -zv_in / scale_i_y;
            //zv_in = 0.f;
            //zv_lwt[idx_zv_1] = 0.f;
            //yv_in -= (lwt_s_x_end - lwt_s_x_1) * L_inv.z;
            ////// y_fast

            ////// z_fast
            ////DTYPE lwt_s_z = (scale_i_y * xv_in - scale_i_x * yv_in) * ip_mid_z_inv;
            //DTYPE lwt_s_z = (syBxBy * xv_in - sxBxBy * yv_in) * ip_mid_z_inv;
            //xv_in -= scale_i_y * lwt_s_z;
            //yv_in += scale_i_x * lwt_s_z;

            //DTYPE& ip_1_z_00 = ip_1_z;
            //DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            //DTYPE scale_i_z_0 = -2.f * L_inv.z;
            //DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            //DTYPE& zv_in_1 = zv_in;

            //DTYPE B0B0B1_01 = ip_0_x * ip_0_y * ip_1_z_01;

            //DTYPE fx = yv_in * scale_i_z_0 * B0B1B0 + (zv_in_0 - zv_in_1) * scale_i_y * (B0B0B1_01 - B0B0B1);
            //DTYPE fy = xv_in * -scale_i_z_0 * B1B0B0 + (zv_in_0 - zv_in_1) * scale_i_x * (B0B0B1 - B0B0B1_01);

            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z_0 * scale_i_z_0 * B0B1B0 + scale_i_y * scale_i_y * 2.f * (B0B0B1 - B0B0B1_01);
            //proj_mat[1] = -2.f * scale_i_x * scale_i_y * (B0B0B1 - B0B0B1_01);
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z_0 * scale_i_z_0 * B1B0B0 + scale_i_x * scale_i_x * 2.f * (B0B0B1 - B0B0B1_01);

            //cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

            //DTYPE lwt_s_x_0 = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y_0 = proj_mat[2] * fx + proj_mat[3] * fy;

            //zv_in_0 += scale_i_y * lwt_s_x_0 - scale_i_x * lwt_s_y_0;
            //zv_in_1 += -scale_i_y * lwt_s_x_0 + scale_i_x * lwt_s_y_0;

            //yv_in -= -2.f * lwt_s_x_0 * L_inv.z;
            //xv_in += -2.f * lwt_s_y_0 * L_inv.z;

            //zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_lwt[idx_yv - 1] = yv_in_0;

            //DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);
            //DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            //DTYPE den_j2j3 = num_j2 * num_j2 + num_j3 * num_j3;

            //int idx_yv_1 = idx_yv - 1;
            //DTYPE yv_in_1 = yv_lwt[idx_yv_1];

            //// x_fast
            //DTYPE lwt_s_x_1 = 1.f / num_j3 * yv_in_1;
            //DTYPE lwt_s_x_end = 1.f / num_j3 * yv_in;
            //yv_in = 0.;
            //yv_lwt[idx_yv_1] = 0.;
            //zv_in += (lwt_s_x_end - lwt_s_x_1) * L_inv.y;

            //// y_fast
            //lwt_s_y = num_j2 / den_j2j3 * zv_in - num_j3 / den_j2j3 * xv_in;
            //xv_in += num_j3 * lwt_s_y;
            //zv_in -= num_j2 * lwt_s_y;


            //int idx_yv_1 = idx_yv - 1;
            //DTYPE yv_in_1 = yv_lwt[idx_yv_1];

            //// x_fast
            //DTYPE lwt_s_x_1 = yv_in_1 / scale_i_z;
            //DTYPE lwt_s_x_end = yv_in / scale_i_z;
            //yv_in = 0.;
            //yv_lwt[idx_yv_1] = 0.;
            //zv_in += (lwt_s_x_end - lwt_s_x_1) * L_inv.y;

            //// y_fast
            ////lwt_s_y = (scale_i_x * zv_in - scale_i_z * xv_in) * ip_mid_y_inv;
            //lwt_s_y = (sxBxBz * zv_in - szBxBz * xv_in) * ip_mid_y_inv;
            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //    // x fast
            //DTYPE& ip_1_y_00 = ip_1_y;
            //DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            //DTYPE scale_i_y_0 = -2.f * L_inv.y;
            //DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            //DTYPE& yv_in_1 = yv_in;

            //DTYPE B0B1_01B0 = ip_0_x * ip_1_y_01 * ip_0_z;

            //DTYPE fx = (yv_in_0 - yv_in_1) * scale_i_z * (B0B1B0 - B0B1_01B0) - zv_in * scale_i_y_0 * B0B0B1;
            //DTYPE fy = -xv_in * scale_i_z * B1B0B0 + zv_in * scale_i_x * B0B0B1;

            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * scale_i_z * 2.f * (B0B1B0 - B0B1_01B0) + scale_i_y_0 * scale_i_y_0 * B0B0B1;
            //proj_mat[1] = -scale_i_x * scale_i_y_0 * B0B0B1;
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z * scale_i_z * B1B0B0 + scale_i_x * scale_i_x * B0B0B1;

            //cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

            //DTYPE lwt_s_x_0 = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in_0 -= scale_i_z * lwt_s_x_0;
            //yv_in_1 -= -scale_i_z * lwt_s_x_0;

            //zv_in += -2.f * lwt_s_x_0 * L_inv.y - scale_i_x * lwt_s_y;
            //xv_in += scale_i_z * lwt_s_y;

            //yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;
            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;

            //DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            //DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            //DTYPE den_j1j3 = num_j1 * num_j1 + num_j3 * num_j3;

            //int idx_xv_1 = idx_xv - dim_dst.y;
            //DTYPE xv_in_1 = xv_lwt[idx_xv_1];

            //// x_fast
            //lwt_s_x = num_j3 / den_j1j3 * yv_in - num_j1 / den_j1j3 * zv_in;
            //yv_in -= num_j3 * lwt_s_x;
            //zv_in += num_j1 * lwt_s_x;

            //// y_fast
            //DTYPE lwt_s_y_1 = -1.f / num_j3 * xv_in_1;
            //DTYPE lwt_s_y_end = -1.f / num_j3 * xv_in;
            ////q_lwt_y[idx_qy] += lwt_s_y_end;
            //xv_in = 0.f;
            //xv_lwt[idx_xv_1] = 0.f;
            //zv_in -= (lwt_s_y_end - lwt_s_y_1) * L_inv.x;

            //xv_lwt[idx_xv - dim_dst.y] = 0.f;

            //DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            //DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            //DTYPE den_j1j3 = num_j1 * num_j1 + num_j3 * num_j3;

            //int idx_xv_1 = idx_xv - dim_dst.y;
            //DTYPE xv_in_1 = xv_lwt[idx_xv_1];

            ////DTYPE proj_i = yv_in * scale_i_z - zv_in * scale_i_y;
            //DTYPE proj_i = yv_in * szByBz - zv_in * syByBz;

            //// x_fast
            //lwt_s_x = proj_i * ip_mid_x_inv;
            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //// y_fast
            //DTYPE lwt_s_y_1 = -xv_in_1 / scale_i_z;
            //DTYPE lwt_s_y_end = -xv_in / scale_i_z;
            ////q_lwt_y[idx_qy] += lwt_s_y_end;
            //xv_in = 0.f;
            //xv_lwt[idx_xv_1] = 0.f;
            //zv_in -= (lwt_s_y_end - lwt_s_y_1) * L_inv.x;

            //// y fast
            //DTYPE& ip_1_x_00 = ip_1_x;
            //DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            //DTYPE scale_i_x_0 = -2.f * L_inv.x;
            //DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            //DTYPE& xv_in_1 = xv_in;

            ////DTYPE B1_01B0B0 = ip_1_x_01 * ip_0_x * ip_0_z;
            //DTYPE szB1_0m1_B0B0 = scale_i_z * (ip_1_x - ip_1_x_01) * ip_0_y * ip_0_z;

            //DTYPE fx = yv_in * scale_i_z * B0B1B0 - zv_in * scale_i_y * B0B0B1;
            //DTYPE fy = -(xv_in_0 - xv_in_1) * szB1_0m1_B0B0  + zv_in * scale_i_x_0 * B0B0B1;

            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * scale_i_z * B0B1B0 + scale_i_y * scale_i_y * B0B0B1;
            //proj_mat[1] = -scale_i_x_0 * scale_i_y * B0B0B1;
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z * szB1_0m1_B0B0 + scale_i_x_0 * scale_i_x_0 * B0B0B1;

            //cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

            //DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y_0 = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x - lwt_s_y_0 * scale_i_x_0;

            //xv_in_0 += scale_i_z * lwt_s_y_0;
            //xv_in_1 += -scale_i_z * lwt_s_y_0;

            //xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv.z;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv.z;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv.x;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv.y;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;


            //printf("xv: %f, %f, zv: %f, %f, yv: %f, %f, L_inv: %f, %f, %f\n", xv_in_0, yv_in_0, zv_in_0, zv_in_1, yv_in_0, yv_in_1, L_inv.x, L_inv.y, L_inv.z);

            // x fast
            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            //printf("lwt_s_x_0_4: %f, %f, %f, %f, zv: %f, %f, yv: %f, %f, \n", lwt_s_x_0, lwt_s_x_1, lwt_s_x_2, lwt_s_x_3, zv_in_0, zv_in_1, yv_in_0, yv_in_1);

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            //printf("lwt_s_y_0_4: %f, %f, %f, %f, zv: %f, %f, xv: %f, %f\n", lwt_s_y_0, lwt_s_y_1, lwt_s_y_2, lwt_s_y_3, zv_in_0, zv_in_1, xv_in_0, xv_in_1);

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            //printf("lwt_s_z_0_4: %f, %f, %f, %f, yv: %f, %f, xv: %f, %f\n", lwt_s_z_0, lwt_s_z_1, lwt_s_z_2, lwt_s_z_3, yv_in_0, yv_in_1, xv_in_0, xv_in_1);
            xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
            yv_lwt[idx_yv - 1] = yv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_in_0;
        }


        //else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        //{
        //    DTYPE& ip_1_z_00 = ip_1_z;
        //    DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
        //    DTYPE scale_i_z_0 = -2.f * L_inv.z;
        //    DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
        //    DTYPE& zv_in_1 = zv_in;

        //    DTYPE B0B0B1_01 = ip_0_x * ip_0_y * ip_1_z_01;

        //    DTYPE fx = yv_in * scale_i_z_0 * B0B1B0 + (zv_in_0 - zv_in_1) * scale_i_y * (B0B0B1_01 - B0B0B1);
        //    DTYPE fy = xv_in * -scale_i_z_0 * B1B0B0 + (zv_in_0 - zv_in_1) * scale_i_x * (B0B0B1 - B0B0B1_01);

        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z_0 * scale_i_z_0 * B0B1B0 + scale_i_y * scale_i_y * 2.f * (B0B0B1 - B0B0B1_01);
        //    proj_mat[1] = -2.f * scale_i_x * scale_i_y * (B0B0B1 - B0B0B1_01);
        //    proj_mat[2] = proj_mat[1];
        //    proj_mat[3] = scale_i_z_0 * scale_i_z_0 * B1B0B0 + scale_i_x * scale_i_x * 2.f * (B0B0B1 - B0B0B1_01);

        //    cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

        //    DTYPE lwt_s_x_0 = proj_mat[0] * fx + proj_mat[1] * fy;
        //    DTYPE lwt_s_y_0 = proj_mat[2] * fx + proj_mat[3] * fy;

        //    zv_in_0 += scale_i_y * lwt_s_x_0 - scale_i_x * lwt_s_y_0;
        //    zv_in_1 += -scale_i_y * lwt_s_x_0 + scale_i_x * lwt_s_y_0;

        //    yv_in -= -2.f * lwt_s_x_0 * L_inv.z;
        //    xv_in += -2.f * lwt_s_y_0 * L_inv.z;

        //    zv_lwt[idx_zv - slice_xy] = zv_in_0;
        //}
        //else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        //{
        //    // x fast
        //    DTYPE& ip_1_y_00 = ip_1_y;
        //    DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
        //    DTYPE scale_i_y_0 = -2.f * L_inv.y;
        //    DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    DTYPE& yv_in_1 = yv_in;

        //    DTYPE B0B1_01B0 = ip_0_x * ip_1_y_01 * ip_0_z;

        //    DTYPE fx = (yv_in_0 - yv_in_1) * scale_i_z * (B0B1B0 - B0B1_01B0) - zv_in * scale_i_y_0 * B0B0B1;
        //    DTYPE fy = -xv_in * scale_i_z * B1B0B0 + zv_in * scale_i_x * B0B0B1;

        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z * scale_i_z * 2.f * (B0B1B0 - B0B1_01B0) + scale_i_y_0 * scale_i_y_0 * B0B0B1;
        //    proj_mat[1] = -scale_i_x * scale_i_y_0 * B0B0B1;
        //    proj_mat[2] = proj_mat[1];
        //    proj_mat[3] = scale_i_z * scale_i_z * B1B0B0 + scale_i_x * scale_i_x * B0B0B1;

        //    cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

        //    DTYPE lwt_s_x_0 = proj_mat[0] * fx + proj_mat[1] * fy;
        //    DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

        //    yv_in_0 -= scale_i_z * lwt_s_x_0;
        //    yv_in_1 -= -scale_i_z * lwt_s_x_0;

        //    zv_in += -2.f * lwt_s_x_0 * L_inv.y - scale_i_x * lwt_s_y;
        //    xv_in += scale_i_z * lwt_s_y;

        //    yv_lwt[idx_yv - 1] = yv_in_0;
        //}
        //else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        //{



        //    // y fast
        //    DTYPE& ip_1_x_00 = ip_1_x;
        //    DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
        //    DTYPE scale_i_x_0 = -2.f * L_inv.x;
        //    DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
        //    DTYPE& xv_in_1 = xv_in;

        //    //DTYPE B1_01B0B0 = ip_1_x_01 * ip_0_x * ip_0_z;
        //    DTYPE szB1_0m1_B0B0 = scale_i_z * (ip_1_x - ip_1_x_01) * ip_0_y * ip_0_z;

        //    DTYPE fx = yv_in * scale_i_z * B0B1B0 - zv_in * scale_i_y * B0B0B1;
        //    DTYPE fy = -(xv_in_0 - xv_in_1) * szB1_0m1_B0B0  + zv_in * scale_i_x_0 * B0B0B1;

        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z * scale_i_z * B0B1B0 + scale_i_y * scale_i_y * B0B0B1;
        //    proj_mat[1] = -scale_i_x_0 * scale_i_y * B0B0B1;
        //    proj_mat[2] = proj_mat[1];
        //    proj_mat[3] = scale_i_z * szB1_0m1_B0B0 + scale_i_x_0 * scale_i_x_0 * B0B0B1;

        //    cuInverse2x2Matrix(proj_mat[0], proj_mat[1], proj_mat[2], proj_mat[3]);

        //    DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
        //    DTYPE lwt_s_y_0 = proj_mat[2] * fx + proj_mat[3] * fy;

        //    yv_in -= scale_i_z * lwt_s_x;
        //    zv_in += scale_i_y * lwt_s_x - lwt_s_y_0 * scale_i_x_0;

        //    xv_in_0 += scale_i_z * lwt_s_y_0;
        //    xv_in_1 += -scale_i_z * lwt_s_y_0;

        //    xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        //}
        //else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        //{
        //    // x fast
        //    DTYPE& ip_1_y_00 = ip_1_y;
        //    DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
        //    DTYPE scale_i_y_0 = L_inv.y;
        //    DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    DTYPE& yv_in_1 = yv_in;

        //    DTYPE& ip_1_z_00 = ip_1_z;
        //    DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
        //    DTYPE scale_i_z_0 = L_inv.z;
        //    DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
        //    DTYPE& zv_in_1 = zv_in;

        //    DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
        //    DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
        //    DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
        //    DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
        //    //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
        //    //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
        //    //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
        //    //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
        //    zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
        //    zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
        //    yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
        //    yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

        //    // y fast
        //    DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
        //    zv_in_0 -= scale_i_x * lwt_s_y.x;
        //    zv_in_1 -= scale_i_x * lwt_s_y.y;
        //    xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

        //    // z fast
        //    DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
        //    yv_in_0 += scale_i_x * lwt_s_z.x;
        //    yv_in_1 += scale_i_x * lwt_s_z.y;
        //    xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

        //    yv_lwt[idx_yv - 1] = yv_in_0;
        //    zv_lwt[idx_zv - slice_xy] = zv_in_0;
        //}
        //else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        //{
        //    // x fast
        //    DTYPE& ip_1_z_00 = ip_1_z;
        //    DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
        //    DTYPE scale_i_z_0 = L_inv.z;
        //    DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
        //    DTYPE& zv_in_1 = zv_in;

        //    DTYPE& ip_1_x_00 = ip_1_x;
        //    DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
        //    DTYPE scale_i_x_0 = L_inv.x;
        //    DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
        //    DTYPE& xv_in_1 = xv_in;

        //    DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
        //    zv_in_0 += scale_i_y * lwt_s_x.x;
        //    zv_in_1 += scale_i_y * lwt_s_x.y;
        //    yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

        //    // y fast
        //    DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
        //    DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
        //    DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
        //    DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
        //    //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv.z;
        //    //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv.z;
        //    //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv.x;
        //    //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv.x;
        //    xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
        //    xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
        //    zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
        //    zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

        //    // z fast
        //    DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
        //    xv_in_0 -= scale_i_y * lwt_s_z.x;
        //    xv_in_1 -= scale_i_y * lwt_s_z.y;
        //    yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

        //    xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        //    zv_lwt[idx_zv - slice_xy] = zv_in_0;
        //}
        //else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        //{
        //    // x fast
        //    DTYPE& ip_1_y_00 = ip_1_y;
        //    DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
        //    DTYPE scale_i_y_0 = L_inv.y;
        //    DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    DTYPE& yv_in_1 = yv_in;

        //    DTYPE& ip_1_x_00 = ip_1_x;
        //    DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
        //    DTYPE scale_i_x_0 = L_inv.x;
        //    DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
        //    DTYPE& xv_in_1 = xv_in;

        //    DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
        //    yv_in_0 -= scale_i_z * lwt_s_x.x;
        //    yv_in_1 -= scale_i_z * lwt_s_x.y;
        //    zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

        //    // y fast
        //    DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
        //    xv_in_0 += scale_i_z * lwt_s_y.x;
        //    xv_in_1 += scale_i_z * lwt_s_y.y;
        //    zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

        //    // z fast
        //    DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
        //    DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
        //    DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
        //    DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
        //    //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv.x;
        //    //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv.x;
        //    //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv.y;
        //    //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv.y;
        //    yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
        //    yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
        //    xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
        //    xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

        //    xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        //    yv_lwt[idx_yv - 1] = yv_in_0;
        //}
        //else
        //{
        //    DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
        //    DTYPE& yv_in_1 = yv_in;

        //    DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
        //    DTYPE& xv_in_1 = xv_in;

        //    DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
        //    DTYPE& zv_in_1 = zv_in;


        //    //printf("xv: %f, %f, zv: %f, %f, yv: %f, %f, L_inv: %f, %f, %f\n", xv_in_0, yv_in_0, zv_in_0, zv_in_1, yv_in_0, yv_in_1, L_inv.x, L_inv.y, L_inv.z);

        //    // x fast
        //    DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
        //    DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
        //    DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
        //    DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
        //    zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
        //    zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
        //    yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
        //    yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

        //    //printf("lwt_s_x_0_4: %f, %f, %f, %f, zv: %f, %f, yv: %f, %f, \n", lwt_s_x_0, lwt_s_x_1, lwt_s_x_2, lwt_s_x_3, zv_in_0, zv_in_1, yv_in_0, yv_in_1);

        //    // y fast
        //    DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
        //    DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
        //    DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
        //    DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
        //    xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
        //    xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
        //    zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
        //    zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

        //    //printf("lwt_s_y_0_4: %f, %f, %f, %f, zv: %f, %f, xv: %f, %f\n", lwt_s_y_0, lwt_s_y_1, lwt_s_y_2, lwt_s_y_3, zv_in_0, zv_in_1, xv_in_0, xv_in_1);

        //    // z fast
        //    DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
        //    DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
        //    DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
        //    DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
        //    yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
        //    yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
        //    xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
        //    xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

        //    //printf("lwt_s_z_0_4: %f, %f, %f, %f, yv: %f, %f, xv: %f, %f\n", lwt_s_z_0, lwt_s_z_1, lwt_s_z_2, lwt_s_z_3, yv_in_0, yv_in_1, xv_in_0, xv_in_1);
        //    xv_lwt[idx_xv - dim_dst.y] = xv_in_0;
        //    yv_lwt[idx_yv - 1] = yv_in_0;
        //    zv_lwt[idx_zv - slice_xy] = zv_in_0;
        //}


        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}

__global__ void cuProjectLocal_q_3d_d_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * (dim_dst.y + 1) + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv[idx_xv];
        DTYPE yv_in = yv[idx_yv];
        DTYPE zv_in = zv[idx_zv];

        DTYPE xv_res = xv_lwt[idx_xv];
        DTYPE yv_res = yv_lwt[idx_yv];
        DTYPE zv_res = zv_lwt[idx_zv];

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;

        DTYPE fx = yv_in * szByBz - zv_in * syByBz;
        DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE fz = xv_in * syBxBy - yv_in * sxBxBy;

        if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        {
            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            proj_mat[2] = proj_mat[1];
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;
        }
        else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        {
            //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x = fx * ip_mid_x_inv;

        }
        else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        {
            //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y = fy * ip_mid_y_inv;

        }
        else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        {
            //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z = fz * ip_mid_z_inv;

        }

        yv_in -= scale_i_z * lwt_s_x;
        zv_in += scale_i_y * lwt_s_x;

        yv_res -= scale_i_z * lwt_s_x;
        zv_res += scale_i_y * lwt_s_x;

        xv_in += scale_i_z * lwt_s_y;
        zv_in -= scale_i_x * lwt_s_y;

        xv_res += scale_i_z * lwt_s_y;
        zv_res -= scale_i_x * lwt_s_y;

        xv_in -= scale_i_y * lwt_s_z;
        yv_in += scale_i_x * lwt_s_z;

        xv_res -= scale_i_y * lwt_s_z;
        yv_res += scale_i_x * lwt_s_z;

        xv[idx_xv] = xv_in;
        yv[idx_yv] = yv_in;
        zv[idx_zv] = zv_in;

        xv_lwt[idx_xv] = xv_res;
        yv_lwt[idx_yv] = yv_res;
        zv_lwt[idx_zv] = zv_res;
    }
}

__global__ void cuProjectLocal_q_3d_n_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };
        DTYPE2 L_inv_x = { L_inv.y / max(L_inv.y, L_inv.z), L_inv.z / max(L_inv.y, L_inv.z), };
        DTYPE2 L_inv_y = { L_inv.x / max(L_inv.x, L_inv.z), L_inv.z / max(L_inv.x, L_inv.z), };
        DTYPE2 L_inv_z = { L_inv.x / max(L_inv.x, L_inv.y), L_inv.y / max(L_inv.x, L_inv.y), };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv[idx_xv];
        DTYPE yv_in = yv[idx_yv];
        DTYPE zv_in = zv[idx_zv];

        DTYPE xv_res = xv_lwt[idx_xv];
        DTYPE yv_res = yv_lwt[idx_yv];
        DTYPE zv_res = zv_lwt[idx_zv];

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            proj_mat[2] = proj_mat[1];
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            yv_res -= scale_i_z * lwt_s_x;
            zv_res += scale_i_y * lwt_s_x;

            xv_res += scale_i_z * lwt_s_y;
            zv_res -= scale_i_x * lwt_s_y;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;
            DTYPE zv_res_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_res_1 = zv_res;

            //DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            zv_res_0 += scale_i_y * lwt_s_x.x;
            zv_res_1 += scale_i_y * lwt_s_x.y;
            yv_res -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y_fast
            //DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            zv_res_0 -= scale_i_x * lwt_s_y.x;
            zv_res_1 -= scale_i_x * lwt_s_y.y;
            xv_res += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            DTYPE lwt_s_z = proj_i_z * ip_mid_z_inv;
            xv_in -= scale_i_y * lwt_s_z;
            yv_in += scale_i_x * lwt_s_z;

            xv_res -= scale_i_y * lwt_s_z;
            yv_res += scale_i_x * lwt_s_z;

            zv[idx_zv - slice_xy] = zv_in_0;
            zv_lwt[idx_zv - slice_xy] = zv_res_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;
            DTYPE yv_res_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_res_1 = yv_res;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            yv_res_0 -= scale_i_z * lwt_s_x.x;
            yv_res_1 -= scale_i_z * lwt_s_x.y;
            zv_res += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            DTYPE lwt_s_y = proj_i_y * ip_mid_y_inv;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            xv_res += scale_i_z * lwt_s_y;
            zv_res -= scale_i_x * lwt_s_y;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_res_0 += scale_i_x * lwt_s_z.x;
            yv_res_1 += scale_i_x * lwt_s_z.y;
            xv_res -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv[idx_yv - 1] = yv_in_0;
            yv_lwt[idx_yv - 1] = yv_res_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            DTYPE lwt_s_x = proj_i_x * ip_mid_x_inv;
            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            yv_res -= scale_i_z * lwt_s_x;
            zv_res += scale_i_y * lwt_s_x;

            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;
            DTYPE xv_res_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_res_1 = xv_res;

            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            xv_res_0 += scale_i_z * lwt_s_y.x;
            xv_res_1 += scale_i_z * lwt_s_y.y;
            zv_res -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_res_0 -= scale_i_y * lwt_s_z.x;
            xv_res_1 -= scale_i_y * lwt_s_z.y;
            yv_res += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv[idx_xv - dim_dst.y] = xv_in_0;
            xv_lwt[idx_xv - dim_dst.y] = xv_res_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;
            DTYPE yv_res_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_res_1 = yv_res;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;
            DTYPE zv_res_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_res_1 = zv_res;

            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            //zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv.y;
            //zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv.y;
            //yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv.z;
            //yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv.z;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            zv_res_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_res_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_res_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_res_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            zv_res_0 -= scale_i_x * lwt_s_y.x;
            zv_res_1 -= scale_i_x * lwt_s_y.y;
            xv_res += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv_res_0 += scale_i_x * lwt_s_z.x;
            yv_res_1 += scale_i_x * lwt_s_z.y;
            xv_res -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;

            yv[idx_yv - 1] = yv_in_0;
            zv[idx_zv - slice_xy] = zv_in_0;
            yv_lwt[idx_yv - 1] = yv_res_0;
            zv_lwt[idx_zv - slice_xy] = zv_res_0;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;
            DTYPE zv_res_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_res_1 = zv_res;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;
            DTYPE xv_res_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_res_1 = xv_res;

            DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            zv_res_0 += scale_i_y * lwt_s_x.x;
            zv_res_1 += scale_i_y * lwt_s_x.y;
            yv_res -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            //xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv.z;
            //xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv.z;
            //zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv.x;
            //zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv.x;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            xv_res_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_res_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_res_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_res_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv_res_0 -= scale_i_y * lwt_s_z.x;
            xv_res_1 -= scale_i_y * lwt_s_z.y;
            yv_res += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            xv[idx_xv - dim_dst.y] = xv_in_0;
            zv[idx_zv - slice_xy] = zv_in_0;
            xv_lwt[idx_xv - dim_dst.y] = xv_res_0;
            zv_lwt[idx_zv - slice_xy] = zv_res_0;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;
            DTYPE yv_res_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_res_1 = yv_res;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;
            DTYPE xv_res_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_res_1 = xv_res;

            DTYPE2 lwt_s_x = cuCalculateP0Val(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            yv_res_0 -= scale_i_z * lwt_s_x.x;
            yv_res_1 -= scale_i_z * lwt_s_x.y;
            zv_res += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            xv_res_0 += scale_i_z * lwt_s_y.x;
            xv_res_1 += scale_i_z * lwt_s_y.y;
            zv_res -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            //yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv.x;
            //yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv.x;
            //xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv.y;
            //xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv.y;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            yv_res_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_res_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_res_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_res_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            xv[idx_xv - dim_dst.y] = xv_in_0;
            yv[idx_yv - 1] = yv_in_0;

            xv_lwt[idx_xv - dim_dst.y] = xv_res_0;
            yv_lwt[idx_yv - 1] = yv_res_0;
        }
        else
        {
            DTYPE yv_in_0 = yv[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;
            DTYPE yv_res_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_res_1 = yv_res;

            DTYPE xv_in_0 = xv[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;
            DTYPE xv_res_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_res_1 = xv_res;

            DTYPE zv_in_0 = zv[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;
            DTYPE zv_res_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_res_1 = zv_res;

            //printf("xv: %f, %f, zv: %f, %f, yv: %f, %f, L_inv: %f, %f, %f\n", xv_in_0, yv_in_0, zv_in_0, zv_in_1, yv_in_0, yv_in_1, L_inv.x, L_inv.y, L_inv.z);

            // x fast
            DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;
            zv_in_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_in_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_in_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_in_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            zv_res_0 += (lwt_s_x_1 - lwt_s_x_0) * L_inv_x.x;
            zv_res_1 += (lwt_s_x_3 - lwt_s_x_2) * L_inv_x.x;
            yv_res_0 -= (lwt_s_x_2 - lwt_s_x_0) * L_inv_x.y;
            yv_res_1 -= (lwt_s_x_3 - lwt_s_x_1) * L_inv_x.y;

            //printf("lwt_s_x_0_4: %f, %f, %f, %f, zv: %f, %f, yv: %f, %f, \n", lwt_s_x_0, lwt_s_x_1, lwt_s_x_2, lwt_s_x_3, zv_in_0, zv_in_1, yv_in_0, yv_in_1);

            // y fast
            DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            DTYPE lwt_s_y_3 = div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1;
            xv_in_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_in_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_in_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_in_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;

            xv_res_0 += (lwt_s_y_1 - lwt_s_y_0) * L_inv_y.y;
            xv_res_1 += (lwt_s_y_3 - lwt_s_y_2) * L_inv_y.y;
            zv_res_0 -= (lwt_s_y_2 - lwt_s_y_0) * L_inv_y.x;
            zv_res_1 -= (lwt_s_y_3 - lwt_s_y_1) * L_inv_y.x;
            //printf("lwt_s_y_0_4: %f, %f, %f, %f, zv: %f, %f, xv: %f, %f\n", lwt_s_y_0, lwt_s_y_1, lwt_s_y_2, lwt_s_y_3, zv_in_0, zv_in_1, xv_in_0, xv_in_1);

            // z fast
            DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            DTYPE lwt_s_z_3 = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;
            yv_in_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_in_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_in_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_in_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            yv_res_0 += (lwt_s_z_1 - lwt_s_z_0) * L_inv_z.x;
            yv_res_1 += (lwt_s_z_3 - lwt_s_z_2) * L_inv_z.x;
            xv_res_0 -= (lwt_s_z_2 - lwt_s_z_0) * L_inv_z.y;
            xv_res_1 -= (lwt_s_z_3 - lwt_s_z_1) * L_inv_z.y;

            //printf("lwt_s_z_0_4: %f, %f, %f, %f, yv: %f, %f, xv: %f, %f\n", lwt_s_z_0, lwt_s_z_1, lwt_s_z_2, lwt_s_z_3, yv_in_0, yv_in_1, xv_in_0, xv_in_1);
            xv[idx_xv - dim_dst.y] = xv_in_0;
            yv[idx_yv - 1] = yv_in_0;
            zv[idx_zv - slice_xy] = zv_in_0;

            xv_lwt[idx_xv - dim_dst.y] = xv_res_0;
            yv_lwt[idx_yv - 1] = yv_res_0;
            zv_lwt[idx_zv - slice_xy] = zv_res_0;
        }

        xv[idx_xv] = xv_in;
        yv[idx_yv] = yv_in;
        zv[idx_zv] = zv_in;

        xv_lwt[idx_xv] = xv_res;
        yv_lwt[idx_yv] = yv_res;
        zv_lwt[idx_zv] = zv_res;
    }
}

__global__ void cuProjectLocal_q_3d_d_single(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* proj_coef_xv, DTYPE* proj_coef_yv,
    DTYPE* proj_coef_zv, int3 dim, int3 levels, DTYPE3 dx_inv, int pc_len)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);
    int& idx_x = data_idx3.x;
    int& idx_y = data_idx3.y;
    int& idx_z = data_idx3.z;

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        // calculate L0L0L0
        if (idx_y > 0 && idx_x > 0 && idx_z > 0)
        {
            int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
            int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
            int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

            //DTYPE numJ1 = g_dPow2[jy + 2] * dx_inv.y / g_dPow2[dim_level.y];
            //DTYPE numJ2 = g_dPow2[jx + 2] * dx_inv.x / g_dPow2[dim_level.x];

            DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);
            DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            DTYPE den_j1j2 = num_j1 * num_j1 + num_j2 * num_j2;
            DTYPE den_j1j3 = num_j1 * num_j1 + num_j3 * num_j3;
            DTYPE den_j2j3 = num_j2 * num_j2 + num_j3 * num_j3;

            // x_fast
            DTYPE lwt_s_x = num_j3 / den_j1j3 * yv_in - num_j1 / den_j1j3 * zv_in;
            yv_in -= num_j3 * lwt_s_x;
            zv_in += num_j1 * lwt_s_x;

            // y_fast
            DTYPE lwt_s_y = num_j2 / den_j2j3 * zv_in - num_j3 / den_j2j3 * xv_in;
            xv_in += num_j3 * lwt_s_y;
            zv_in -= num_j2 * lwt_s_y;

            // z_fast
            DTYPE lwt_s_z = num_j1 / den_j1j2 * xv_in - num_j2 / den_j1j2 * yv_in;
            xv_in -= num_j1 * lwt_s_z;
            yv_in += num_j2 * lwt_s_z;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0   ** x-z plane
        {
            int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
            int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
            int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

            DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);
            DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            DTYPE den_j2j3 = num_j2 * num_j2 + num_j3 * num_j3;

            // y_fast
            DTYPE lwt_s_y = num_j2 / den_j2j3 * zv_in - num_j3 / den_j2j3 * xv_in;
            xv_in += num_j3 * lwt_s_y;
            zv_in -= num_j2 * lwt_s_y;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z > 0) // calculate L1L0L0  ** y-z plane
        {
            int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
            int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
            int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

            DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            DTYPE num_j3 = 4.f * dx_inv.z / (1 << jz);

            DTYPE den_j1j3 = num_j1 * num_j1 + num_j3 * num_j3;

            // x_fast
            DTYPE lwt_s_x = num_j3 / den_j1j3 * yv_in - num_j1 / den_j1j3 * zv_in;
            yv_in -= num_j3 * lwt_s_x;
            zv_in += num_j1 * lwt_s_x;
        }
        else if (idx_y > 0 && idx_x > 0 && idx_z == 0) // calculate L0L0L1  ** x-y plane
        {
            int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
            int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
            int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

            //DTYPE numJ1 = g_dPow2[jy + 2] * dx_inv.y / g_dPow2[dim_level.y];
            //DTYPE numJ2 = g_dPow2[jx + 2] * dx_inv.x / g_dPow2[dim_level.x];

            DTYPE num_j1 = 4.f * dx_inv.y / (1 << jy);
            DTYPE num_j2 = 4.f * dx_inv.x / (1 << jx);

            DTYPE den_j1j2 = num_j1 * num_j1 + num_j2 * num_j2;

            // z_fast
            DTYPE lwt_s_z = num_j1 / den_j1j2 * xv_in - num_j2 / den_j1j2 * yv_in;
            xv_in -= num_j1 * lwt_s_z;
            yv_in += num_j2 * lwt_s_z;
        }

        xv_lwt[idx_xv] = xv_in;
        yv_lwt[idx_yv] = yv_in;
        zv_lwt[idx_zv] = zv_in;
    }
}


void CuDivLocalProject3D::ProjectLocal(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 32;


    int blocks_y = std::ceil(double(dim.y) / threads_y);
    int blocks_x = std::ceil(double(dim.x) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
    }
    else
    {
        cuProjectLocal_q_3d_d << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
}

void CuDivLocalProject3D::ProjectLocal(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 32;

    int blocks_y = std::ceil(double(dim.y) / threads_y);
    int blocks_x = std::ceil(double(dim.x) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        cuProjectLocal_q_3d_n << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    else
    {
        cuProjectLocal_q_3d_d << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
}

void CuDivLocalProject3D::ProjectLocal_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    int threads_x = bc_base_ == 'd' ? 32 : 16;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        cuProjectLocal_q_3d_n_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    else
    {
        cuProjectLocal_q_3d_d_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    checkCudaErrors(cudaGetLastError());
}

void CuDivLocalProject3D::ProjectLocal_ccc(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 32;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        cuProjectLocal_q_3d_n_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, xv, yv, zv, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    else
    {
        cuProjectLocal_q_3d_d_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, xv, yv, zv, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
}

void CuDivLocalProject3D::ProjectLocal_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 32;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'd')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        //cuProjectLocal_q_3d_d_ttt << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, proj_coef_xv_, proj_coef_yv_, proj_coef_zv_, dim, levels_, dx_inv_, proj_len_);
        //cuProjectLocal_q_3d_d_ttt_old << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, proj_coef_xv_, proj_coef_yv_, proj_coef_zv_, dim, levels_, dx_inv_, proj_len_);
        cuProjectLocal_q_3d_d_single << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, proj_coef_xv_, proj_coef_yv_, proj_coef_zv_, dim, levels_, dx_inv_, proj_len_);
    }
    //else
    //{
    //    cuProjectLocal_q_3d_d_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
    //        ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    //}
}

void CuDivLocalProject3D::ProjectLocal_ttt(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 32;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'd')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        cuProjectLocal_q_3d_d_ttt << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, xv, yv, zv, proj_coef_xv_, proj_coef_yv_, proj_coef_zv_, dim, levels_, dx_inv_, proj_len_);
    }
    //else
    //{
    //    cuProjectLocal_q_3d_d_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
    //        ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    //}
}

__global__ void cuProjectLocal_q_3d_dq_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * (dim_dst.y + 1) + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_xy_ext;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;
        DTYPE lwt_s_x_t = 0.f;
        DTYPE lwt_s_y_t = 0.f;
        DTYPE lwt_s_z_t = 0.f;

        DTYPE fx = yv_in * szByBz - zv_in * syByBz;
        DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE fz = xv_in * syBxBy - yv_in * sxBxBy;

        if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        {
            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            //proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            ////proj_mat[2] = -scale_i_y * sxBxBz;
            //proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            //DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            //proj_mat[0] *= proj_mat_bot_inv;
            //proj_mat[1] *= -proj_mat_bot_inv;
            //proj_mat[2] *= -proj_mat_bot_inv;
            //proj_mat[3] *= proj_mat_bot_inv;

            //lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //fx = yv_in * szByBz - zv_in * syByBz;
            //fy = zv_in * sxBxBz - xv_in * szBxBz;

            //lwt_s_x = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;

            DTYPE scale_i_y_no4 = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
            DTYPE scale_i_x_no4 = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
            DTYPE scale_i_z_no4 = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

            DTYPE proj_mat_bot_inv = 1.f / (scale_i_y_no4 * scale_i_y_no4 + scale_i_x_no4 * scale_i_x_no4 + scale_i_z_no4 * scale_i_z_no4);
            lwt_s_z = (scale_i_y_no4 * xv_in - scale_i_x_no4 * yv_in) * proj_mat_bot_inv;
            lwt_s_x = (scale_i_z_no4 * yv_in - scale_i_y_no4 * zv_in) * proj_mat_bot_inv;
            lwt_s_y = (scale_i_x_no4 * zv_in - scale_i_z_no4 * xv_in) * proj_mat_bot_inv;
        }
        else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        {
            //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x = fx * ip_mid_x_inv;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        {
            //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y = fy * ip_mid_y_inv;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        {
            //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z = fz * ip_mid_z_inv;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }

        /////////////////////---------------------------------second-----------------------////////////////////////////
        //lwt_s_x_t += lwt_s_x;
        //lwt_s_y_t += lwt_s_y;
        //lwt_s_z_t += lwt_s_z;

        //yv_in -= scale_i_z * lwt_s_x;
        //zv_in += scale_i_y * lwt_s_x;
        //xv_in += scale_i_z * lwt_s_y;
        //zv_in -= scale_i_x * lwt_s_y;
        //xv_in -= scale_i_y * lwt_s_z;
        //yv_in += scale_i_x * lwt_s_z;

        //fx = yv_in * szByBz - zv_in * syByBz;
        //fy = zv_in * sxBxBz - xv_in * szBxBz;
        //fz = xv_in * syBxBy - yv_in * sxBxBy;

        //if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        //{
        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
        //    proj_mat[1] = -scale_i_x * syByBz;
        //    proj_mat[2] = proj_mat[1];
        //    //proj_mat[2] = -scale_i_y * sxBxBz;
        //    proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

        //    DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
        //    proj_mat[0] *= proj_mat_bot_inv;
        //    proj_mat[1] *= -proj_mat_bot_inv;
        //    proj_mat[2] *= -proj_mat_bot_inv;
        //    proj_mat[3] *= proj_mat_bot_inv;

        //    lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
        //    lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;
        //}
        //else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        //{
        //    //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
        //    lwt_s_x = fx * ip_mid_x_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        //{
        //    //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
        //    lwt_s_y = fy * ip_mid_y_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        //{
        //    //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
        //    lwt_s_z = fz * ip_mid_z_inv;

        //}

        q_lwt_x[idx_qx] = lwt_s_x + lwt_s_x_t;
        q_lwt_y[idx_qy] = lwt_s_y + lwt_s_y_t;
        q_lwt_z[idx_qz] = lwt_s_z + lwt_s_z_t;
    }
}

__global__ void cuProjectLocal_q_3d_dq_ccc_old(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * (dim_dst.y + 1) + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_xy_ext;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;
        DTYPE lwt_s_x_t = 0.f;
        DTYPE lwt_s_y_t = 0.f;
        DTYPE lwt_s_z_t = 0.f;

        DTYPE fx = yv_in * szByBz - zv_in * syByBz;
        DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE fz = xv_in * syBxBy - yv_in * sxBxBy;

        if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        {
            DTYPE proj_mat[4];
            proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            proj_mat[1] = -scale_i_x * syByBz;
            proj_mat[2] = proj_mat[1];
            //proj_mat[2] = -scale_i_y * sxBxBz;
            proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            proj_mat[0] *= proj_mat_bot_inv;
            proj_mat[1] *= -proj_mat_bot_inv;
            proj_mat[2] *= -proj_mat_bot_inv;
            proj_mat[3] *= proj_mat_bot_inv;

            lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            yv_in -= scale_i_z * lwt_s_x;
            zv_in += scale_i_y * lwt_s_x;

            xv_in += scale_i_z * lwt_s_y;
            zv_in -= scale_i_x * lwt_s_y;

            fx = yv_in * szByBz - zv_in * syByBz;
            fy = zv_in * sxBxBz - xv_in * szBxBz;

            lwt_s_x = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            lwt_s_y = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;
        }
        else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        {
            //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x = fx * ip_mid_x_inv;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        {
            //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y = fy * ip_mid_y_inv;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        {
            //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z = fz * ip_mid_z_inv;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }

        /////////////////////---------------------------------second-----------------------////////////////////////////
        //lwt_s_x_t += lwt_s_x;
        //lwt_s_y_t += lwt_s_y;
        //lwt_s_z_t += lwt_s_z;

        //yv_in -= scale_i_z * lwt_s_x;
        //zv_in += scale_i_y * lwt_s_x;
        //xv_in += scale_i_z * lwt_s_y;
        //zv_in -= scale_i_x * lwt_s_y;
        //xv_in -= scale_i_y * lwt_s_z;
        //yv_in += scale_i_x * lwt_s_z;

        //fx = yv_in * szByBz - zv_in * syByBz;
        //fy = zv_in * sxBxBz - xv_in * szBxBz;
        //fz = xv_in * syBxBy - yv_in * sxBxBy;

        //if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        //{
        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
        //    proj_mat[1] = -scale_i_x * syByBz;
        //    proj_mat[2] = proj_mat[1];
        //    //proj_mat[2] = -scale_i_y * sxBxBz;
        //    proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

        //    DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
        //    proj_mat[0] *= proj_mat_bot_inv;
        //    proj_mat[1] *= -proj_mat_bot_inv;
        //    proj_mat[2] *= -proj_mat_bot_inv;
        //    proj_mat[3] *= proj_mat_bot_inv;

        //    lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
        //    lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;
        //}
        //else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        //{
        //    //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
        //    lwt_s_x = fx * ip_mid_x_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        //{
        //    //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
        //    lwt_s_y = fy * ip_mid_y_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        //{
        //    //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
        //    lwt_s_z = fz * ip_mid_z_inv;

        //}

        q_lwt_x[idx_qx] = lwt_s_x + lwt_s_x_t;
        q_lwt_y[idx_qy] = lwt_s_y + lwt_s_y_t;
        q_lwt_z[idx_qz] = lwt_s_z + lwt_s_z_t;
    }
}

__global__ void cuProjectLocal_q_3d_nq_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * slice_qx;
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * slice_qy;
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_qz;

        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x_o = 0.f;
        DTYPE lwt_s_y_o = 0.f;
        DTYPE lwt_s_z_o = 0.f;

        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            //proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            //DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            //proj_mat[0] *= proj_mat_bot_inv;
            //proj_mat[1] *= -proj_mat_bot_inv;
            //proj_mat[2] *= -proj_mat_bot_inv;
            //proj_mat[3] *= proj_mat_bot_inv;

            //DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            //DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            //DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //fx = yv_in * szByBz - zv_in * syByBz;
            //fy = zv_in * sxBxBz - xv_in * szBxBz;

            //lwt_s_x_o = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y_o = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;

            DTYPE scale_i_y_no4 = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
            DTYPE scale_i_x_no4 = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
            DTYPE scale_i_z_no4 = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

            DTYPE proj_mat_bot_inv = 1.f / (scale_i_y_no4 * scale_i_y_no4 + scale_i_x_no4 * scale_i_x_no4 + scale_i_z_no4 * scale_i_z_no4);
            lwt_s_z_o = (scale_i_y_no4 * xv_in - scale_i_x_no4 * yv_in) * proj_mat_bot_inv;
            lwt_s_x_o = (scale_i_z_no4 * yv_in - scale_i_y_no4 * zv_in) * proj_mat_bot_inv;
            lwt_s_y_o = (scale_i_x_no4 * zv_in - scale_i_z_no4 * xv_in) * proj_mat_bot_inv;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            ////DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //zv_in_0 += scale_i_y * lwt_s_x.x;
            //zv_in_1 += scale_i_y * lwt_s_x.y;
            //yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            //lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = 0.f;

            //// y_fast
            ////DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z_o = proj_i_z * ip_mid_z_inv;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y_o = proj_i_y * ip_mid_y_inv;

            q_lwt_z[idx_qz - 1] = 0.f;

        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x_o = proj_i_x * ip_mid_x_inv;

            q_lwt_y[idx_qy - dim.y] = 0.f;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;


            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - 1] = lwt_s_z.x;

            //DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;

            //lwt_s_x_o = lwt_s_x_3;

            //q_lwt_x[idx_qx - 1] = lwt_s_x_1;
            //q_lwt_x[idx_qx - slice_qz - 1] = lwt_s_x_0;
            //q_lwt_x[idx_qx - slice_qz] = lwt_s_x_2;
            lwt_s_x_o = 0.f;

            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // y fast
            //DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //lwt_s_y_o = (div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1) / scale_i_y;

            //q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_0;
            //q_lwt_y[idx_qy - slice_qy] = lwt_s_y_2;
            //q_lwt_y[idx_qy - dim.y] = lwt_s_y_1;
            lwt_s_y_o = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y.x;

            // z fast
            //DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //lwt_s_z_o = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;

            //q_lwt_z[idx_qz - dim_ext.y - 1] = lwt_s_z_0;
            //q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z_2;
            //q_lwt_z[idx_qz - 1] = lwt_s_z_1;

            lwt_s_z_o = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            if (false)
            {
                DTYPE b = -((zv_in_1 - zv_in_0) / L_inv.z + (yv_in_1 - yv_in_0) / L_inv.y + (xv_in_1 - xv_in_0) / L_inv.x);

                printf("xv : %f, %f, yv: %f, %f, zv: %f, %f, b: %f\n", xv_in_0, xv_in_1, yv_in_0, yv_in_1, zv_in_0, zv_in_1, b);
                printf("xv + yv : %f, yv: %f\n", xv_in_1 - xv_in_0 + zv_in_1 - zv_in_0, yv_in_1 - yv_in_0);


                //DTYPE p = b / 12.f;
                //DTYPE p = b / 12.f;
                DTYPE p = b / (4.f / (L_inv.x * L_inv.x) + 4.f / (L_inv.y * L_inv.y) + 4.f / (L_inv.z * L_inv.z));
                zv_in_1 += 2.f * p / L_inv.z;
                zv_in_0 -= 2.f * p / L_inv.z;
                xv_in_1 += 2.f * p / L_inv.x;
                xv_in_0 -= 2.f * p / L_inv.x;
                yv_in_1 += 2.f * p / L_inv.y;
                yv_in_0 -= 2.f * p / L_inv.y;
            }



            DTYPE lwt_s_x_000 = 0.f;
            DTYPE lwt_s_x_010 = lwt_s_x_000 - zv_in_0 / L_inv.y;
            DTYPE lwt_s_x_001 = lwt_s_x_000 + yv_in_0 / L_inv.z;
            DTYPE lwt_s_x_011 = lwt_s_x_010 + yv_in_1 / L_inv.z;

            DTYPE lwt_s_y_000 = 0.f;
            DTYPE lwt_s_y_100 = 0.f;
            DTYPE lwt_s_y_001 = lwt_s_y_000 - xv_in_0 / L_inv.z;
            DTYPE lwt_s_y_101 = lwt_s_y_100 - xv_in_1 / L_inv.z;

            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            lwt_s_z_o = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = lwt_s_x_000;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x_010;
            q_lwt_x[idx_qx - 1] = lwt_s_x_001;
            lwt_s_x_o = lwt_s_x_011;

            q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_000;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y_100;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y_001;
            lwt_s_y_o = lwt_s_y_101;
        }

        q_lwt_x[idx_qx] = lwt_s_x_o;
        q_lwt_y[idx_qy] = lwt_s_y_o;
        q_lwt_z[idx_qz] = lwt_s_z_o;
    }
}

__global__ void cuProjectLocal_q_3d_nq_ccc_test(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * slice_qx;
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * slice_qy;
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_qz;

        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x_o = 0.f;
        DTYPE lwt_s_y_o = 0.f;
        DTYPE lwt_s_z_o = 0.f;

        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            //proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            //DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            //proj_mat[0] *= proj_mat_bot_inv;
            //proj_mat[1] *= -proj_mat_bot_inv;
            //proj_mat[2] *= -proj_mat_bot_inv;
            //proj_mat[3] *= proj_mat_bot_inv;

            //DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            //DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            //DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //fx = yv_in * szByBz - zv_in * syByBz;
            //fy = zv_in * sxBxBz - xv_in * szBxBz;

            //lwt_s_x_o = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y_o = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;

            DTYPE scale_i_y_no4 = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
            DTYPE scale_i_x_no4 = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
            DTYPE scale_i_z_no4 = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

            DTYPE proj_mat_bot_inv = 1.f / (scale_i_y_no4 * scale_i_y_no4 + scale_i_x_no4 * scale_i_x_no4 + scale_i_z_no4 * scale_i_z_no4);
            lwt_s_z_o = (scale_i_y_no4 * xv_in - scale_i_x_no4 * yv_in) * proj_mat_bot_inv;
            lwt_s_x_o = (scale_i_z_no4 * yv_in - scale_i_y_no4 * zv_in) * proj_mat_bot_inv;
            lwt_s_y_o = (scale_i_x_no4 * zv_in - scale_i_z_no4 * xv_in) * proj_mat_bot_inv;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            ////DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //zv_in_0 += scale_i_y * lwt_s_x.x;
            //zv_in_1 += scale_i_y * lwt_s_x.y;
            //yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            //lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = 0.f;

            //// y_fast
            ////DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z_o = proj_i_z * ip_mid_z_inv;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y_o = proj_i_y * ip_mid_y_inv;

            q_lwt_z[idx_qz - 1] = 0.f;

        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x_o = proj_i_x * ip_mid_x_inv;

            q_lwt_y[idx_qy - dim.y] = 0.f;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;


            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - 1] = lwt_s_z.x;

            //DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;

            //lwt_s_x_o = lwt_s_x_3;

            //q_lwt_x[idx_qx - 1] = lwt_s_x_1;
            //q_lwt_x[idx_qx - slice_qz - 1] = lwt_s_x_0;
            //q_lwt_x[idx_qx - slice_qz] = lwt_s_x_2;
            lwt_s_x_o = 0.f;

            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // y fast
            //DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //lwt_s_y_o = (div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1) / scale_i_y;

            //q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_0;
            //q_lwt_y[idx_qy - slice_qy] = lwt_s_y_2;
            //q_lwt_y[idx_qy - dim.y] = lwt_s_y_1;
            lwt_s_y_o = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y.x;

            // z fast
            //DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //lwt_s_z_o = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;

            //q_lwt_z[idx_qz - dim_ext.y - 1] = lwt_s_z_0;
            //q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z_2;
            //q_lwt_z[idx_qz - 1] = lwt_s_z_1;

            lwt_s_z_o = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE b = -((zv_in_1 - zv_in_0) / L_inv.z + (yv_in_1 - yv_in_0) / L_inv.y + (xv_in_1 - xv_in_0) / L_inv.x);

            DTYPE p = b / 12.f;
            zv_in_1 += 2.f * p / L_inv.z;
            zv_in_0 -= 2.f * p / L_inv.z;
            xv_in_1 += 2.f * p / L_inv.x;
            xv_in_0 -= 2.f * p / L_inv.x;
            yv_in_1 += 2.f * p / L_inv.y;
            yv_in_0 -= 2.f * p / L_inv.y;

            DTYPE lwt_s_x_000 = 0.f;
            DTYPE lwt_s_x_010 = lwt_s_x_000 - zv_in_0 / L_inv.y;
            DTYPE lwt_s_x_001 = lwt_s_x_000 + yv_in_0 / L_inv.z;
            DTYPE lwt_s_x_011 = lwt_s_x_010 + yv_in_1 / L_inv.z;

            DTYPE lwt_s_y_000 = 0.f;
            DTYPE lwt_s_y_100 = 0.f;
            DTYPE lwt_s_y_001 = lwt_s_y_000 - xv_in_0 / L_inv.z;
            DTYPE lwt_s_y_101 = lwt_s_y_100 - xv_in_1 / L_inv.z;

            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            lwt_s_z_o = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = lwt_s_x_000;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x_010;
            q_lwt_x[idx_qx - 1] = lwt_s_x_001;
            lwt_s_x_o = lwt_s_x_011;

            q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_000;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y_100;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y_001;
            lwt_s_y_o = lwt_s_y_101;
        }

        q_lwt_x[idx_qx] = lwt_s_x_o;
        q_lwt_y[idx_qy] = lwt_s_y_o;
        q_lwt_z[idx_qz] = lwt_s_z_o;
    }
}

void CuDivLocalProject3D::ProjectLocal_q_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    int threads_x = bc_base_ == 'd' ? 32 : 16;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        //cuProjectLocal_q_3d_n_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
        cuProjectLocal_q_3d_nq_ccc << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    else
    {
        cuProjectLocal_q_3d_dq_ccc << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
        //cuProjectLocal_q_3d_d << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    checkCudaErrors(cudaGetLastError());
}

__global__ void cuProjectLocal_q_3d_nq_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, 
    DTYPE* xv_lwt_out, DTYPE* yv_lwt_out, DTYPE* zv_lwt_out, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt,
    DTYPE* ip_base_x, DTYPE* ip_base_y, DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * dim_dst_ext.y + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * slice_qx;
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * slice_qy;
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_qz;

        DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE B0B1B0 = ip_0_x * ip_1_y * ip_0_z;
        DTYPE B1B0B0 = ip_1_x * ip_0_y * ip_0_z;
        DTYPE B0B0B1 = ip_0_x * ip_0_y * ip_1_z;

        int& idx_y = data_idx3.y;
        int& idx_x = data_idx3.x;
        int& idx_z = data_idx3.z;

        DTYPE szByBz = scale_i_z * B0B1B0;
        DTYPE syByBz = scale_i_y * B0B0B1;

        DTYPE sxBxBz = scale_i_x * B0B0B1;
        DTYPE szBxBz = scale_i_z * B1B0B0;

        DTYPE sxBxBy = scale_i_x * B0B1B0;
        DTYPE syBxBy = scale_i_y * B1B0B0;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x_o = 0.f;
        DTYPE lwt_s_y_o = 0.f;
        DTYPE lwt_s_z_o = 0.f;

        if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z > 0)
        {
            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            //proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            //proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            //DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            //proj_mat[0] *= proj_mat_bot_inv;
            //proj_mat[1] *= -proj_mat_bot_inv;
            //proj_mat[2] *= -proj_mat_bot_inv;
            //proj_mat[3] *= proj_mat_bot_inv;

            //DTYPE fx = yv_in * szByBz - zv_in * syByBz;
            //DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;

            //DTYPE lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //DTYPE lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //fx = yv_in * szByBz - zv_in * syByBz;
            //fy = zv_in * sxBxBz - xv_in * szBxBz;

            //lwt_s_x_o = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y_o = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;

            DTYPE scale_i_y_no4 = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
            DTYPE scale_i_x_no4 = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
            DTYPE scale_i_z_no4 = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

            DTYPE proj_mat_bot_inv = 1.f / (scale_i_y_no4 * scale_i_y_no4 + scale_i_x_no4 * scale_i_x_no4 + scale_i_z_no4 * scale_i_z_no4);
            lwt_s_z_o = (scale_i_y_no4 * xv_in - scale_i_x_no4 * yv_in) * proj_mat_bot_inv;
            lwt_s_x_o = (scale_i_z_no4 * yv_in - scale_i_y_no4 * zv_in) * proj_mat_bot_inv;
            lwt_s_y_o = (scale_i_x_no4 * zv_in - scale_i_z_no4 * xv_in) * proj_mat_bot_inv;
        }
        else if (data_idx3.y > 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L1L0L0  ** x-y plane
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            ////DTYPE2 lwt_s_x = cuCalculateP0Val(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            //zv_in_0 += scale_i_y * lwt_s_x.x;
            //zv_in_1 += scale_i_y * lwt_s_x.y;
            //yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;

            //lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = 0.f;

            //// y_fast
            ////DTYPE2 lwt_s_y = cuCalculateP0Val(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;

            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;

            // z_fast
            DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z_o = proj_i_z * ip_mid_z_inv;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z > 0) // calculate L1L0L0  ** x-z plane
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;

            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y_o = proj_i_y * ip_mid_y_inv;

            q_lwt_z[idx_qz - 1] = 0.f;

        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L0L0  ** y-z plane
        {
            // y fast
            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;

            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // x fast
            DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x_o = proj_i_x * ip_mid_x_inv;

            q_lwt_y[idx_qy - dim.y] = 0.f;

        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(zv_in_0, zv_in_1, -xv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_x, ip_0_x, scale_i_z_0, scale_i_x);
            zv_in_0 -= scale_i_x * lwt_s_y.x;
            zv_in_1 -= scale_i_x * lwt_s_y.y;
            xv_in += (lwt_s_y.y - lwt_s_y.x) * L_inv.z;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y.x;


            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(-yv_in_0, -yv_in_1, xv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_x, ip_0_x, scale_i_y_0, scale_i_x);
            yv_in_0 += scale_i_x * lwt_s_z.x;
            yv_in_1 += scale_i_x * lwt_s_z.y;
            xv_in -= (lwt_s_z.y - lwt_s_z.x) * L_inv.y;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - 1] = lwt_s_z.x;

            //DTYPE lwt_s_x_0 = div_l0_mat_yz[2] * yv_in_0 + div_l0_mat_yz[3] * yv_in_1 - div_l0_mat_yz[0] * zv_in_0 - div_l0_mat_yz[1] * zv_in_1;
            //DTYPE lwt_s_x_1 = div_l0_mat_yz[6] * yv_in_0 + div_l0_mat_yz[7] * yv_in_1 - div_l0_mat_yz[4] * zv_in_0 - div_l0_mat_yz[5] * zv_in_1;
            //DTYPE lwt_s_x_2 = div_l0_mat_yz[10] * yv_in_0 + div_l0_mat_yz[11] * yv_in_1 - div_l0_mat_yz[8] * zv_in_0 - div_l0_mat_yz[9] * zv_in_1;
            //DTYPE lwt_s_x_3 = div_l0_mat_yz[14] * yv_in_0 + div_l0_mat_yz[15] * yv_in_1 - div_l0_mat_yz[12] * zv_in_0 - div_l0_mat_yz[13] * zv_in_1;

            //lwt_s_x_o = lwt_s_x_3;

            //q_lwt_x[idx_qx - 1] = lwt_s_x_1;
            //q_lwt_x[idx_qx - slice_qz - 1] = lwt_s_x_0;
            //q_lwt_x[idx_qx - slice_qz] = lwt_s_x_2;
            lwt_s_x_o = 0.f;

            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            // x fast
            DTYPE& ip_1_z_00 = ip_1_z;
            DTYPE ip_1_z_01 = ip_base_z[dim_dst_ext.z * 3];
            DTYPE scale_i_z_0 = L_inv.z;
            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(-zv_in_0, -zv_in_1, yv_in, ip_1_z_00, ip_1_z_01, ip_0_z, ip_1_y, ip_0_y, scale_i_z_0, scale_i_y);
            zv_in_0 += scale_i_y * lwt_s_x.x;
            zv_in_1 += scale_i_y * lwt_s_x.y;
            yv_in -= (lwt_s_x.y - lwt_s_x.x) * L_inv.z;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x.x;

            // z fast
            DTYPE2 lwt_s_z = cuCalculateP0Val_single(xv_in_0, xv_in_1, -yv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_y, ip_0_y, scale_i_x_0, scale_i_y);
            xv_in_0 -= scale_i_y * lwt_s_z.x;
            xv_in_1 -= scale_i_y * lwt_s_z.y;
            yv_in += (lwt_s_z.y - lwt_s_z.x) * L_inv.x;
            lwt_s_z_o = lwt_s_z.y;
            q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z.x;

            // y fast
            //DTYPE lwt_s_y_0 = div_l0_mat_zx[2] * zv_in_0 + div_l0_mat_zx[3] * zv_in_1 - div_l0_mat_zx[0] * xv_in_0 - div_l0_mat_zx[1] * xv_in_1;
            //DTYPE lwt_s_y_1 = div_l0_mat_zx[6] * zv_in_0 + div_l0_mat_zx[7] * zv_in_1 - div_l0_mat_zx[4] * xv_in_0 - div_l0_mat_zx[5] * xv_in_1;
            //DTYPE lwt_s_y_2 = div_l0_mat_zx[10] * zv_in_0 + div_l0_mat_zx[11] * zv_in_1 - div_l0_mat_zx[8] * xv_in_0 - div_l0_mat_zx[9] * xv_in_1;
            //lwt_s_y_o = (div_l0_mat_zx[14] * zv_in_0 + div_l0_mat_zx[15] * zv_in_1 - div_l0_mat_zx[12] * xv_in_0 - div_l0_mat_zx[13] * xv_in_1) / scale_i_y;

            //q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_0;
            //q_lwt_y[idx_qy - slice_qy] = lwt_s_y_2;
            //q_lwt_y[idx_qy - dim.y] = lwt_s_y_1;
            lwt_s_y_o = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            // x fast
            DTYPE& ip_1_y_00 = ip_1_y;
            DTYPE ip_1_y_01 = ip_base_y[dim_dst_ext.y * 3];
            DTYPE scale_i_y_0 = L_inv.y;
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE& ip_1_x_00 = ip_1_x;
            DTYPE ip_1_x_01 = ip_base_x[dim_dst_ext.x * 3];
            DTYPE scale_i_x_0 = L_inv.x;
            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE2 lwt_s_x = cuCalculateP0Val_single(yv_in_0, yv_in_1, -zv_in, ip_1_y_00, ip_1_y_01, ip_0_y, ip_1_z, ip_0_z, scale_i_y_0, scale_i_z);
            yv_in_0 -= scale_i_z * lwt_s_x.x;
            yv_in_1 -= scale_i_z * lwt_s_x.y;
            zv_in += (lwt_s_x.y - lwt_s_x.x) * L_inv.y;
            lwt_s_x_o = lwt_s_x.y;
            q_lwt_x[idx_qx - 1] = lwt_s_x.x;

            // y fast
            DTYPE2 lwt_s_y = cuCalculateP0Val_single(-xv_in_0, -xv_in_1, zv_in, ip_1_x_00, ip_1_x_01, ip_0_x, ip_1_z, ip_0_z, scale_i_x_0, scale_i_z);
            xv_in_0 += scale_i_z * lwt_s_y.x;
            xv_in_1 += scale_i_z * lwt_s_y.y;
            zv_in -= (lwt_s_y.y - lwt_s_y.x) * L_inv.x;
            lwt_s_y_o = lwt_s_y.y;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y.x;

            // z fast
            //DTYPE lwt_s_z_0 = div_l0_mat_xy[2] * xv_in_0 + div_l0_mat_xy[3] * xv_in_1 - div_l0_mat_xy[0] * yv_in_0 - div_l0_mat_xy[1] * yv_in_1;
            //DTYPE lwt_s_z_1 = div_l0_mat_xy[6] * xv_in_0 + div_l0_mat_xy[7] * xv_in_1 - div_l0_mat_xy[4] * yv_in_0 - div_l0_mat_xy[5] * yv_in_1;
            //DTYPE lwt_s_z_2 = div_l0_mat_xy[10] * xv_in_0 + div_l0_mat_xy[11] * xv_in_1 - div_l0_mat_xy[8] * yv_in_0 - div_l0_mat_xy[9] * yv_in_1;
            //lwt_s_z_o = div_l0_mat_xy[14] * xv_in_0 + div_l0_mat_xy[15] * xv_in_1 - div_l0_mat_xy[12] * yv_in_0 - div_l0_mat_xy[13] * yv_in_1;

            //q_lwt_z[idx_qz - dim_ext.y - 1] = lwt_s_z_0;
            //q_lwt_z[idx_qz - dim_ext.y] = lwt_s_z_2;
            //q_lwt_z[idx_qz - 1] = lwt_s_z_1;

            lwt_s_z_o = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            DTYPE yv_in_0 = yv_lwt[idx_yv - 1];
            DTYPE& yv_in_1 = yv_in;

            DTYPE xv_in_0 = xv_lwt[idx_xv - dim_dst.y];
            DTYPE& xv_in_1 = xv_in;

            DTYPE zv_in_0 = zv_lwt[idx_zv - slice_xy];
            DTYPE& zv_in_1 = zv_in;

            if (false)
            {
                DTYPE b = -((zv_in_1 - zv_in_0) / L_inv.z + (yv_in_1 - yv_in_0) / L_inv.y + (xv_in_1 - xv_in_0) / L_inv.x);

                printf("xv : %f, %f, yv: %f, %f, zv: %f, %f, b: %f\n", xv_in_0, xv_in_1, yv_in_0, yv_in_1, zv_in_0, zv_in_1, b);
                printf("xv + yv : %f, yv: %f\n", xv_in_1 - xv_in_0 + zv_in_1 - zv_in_0, yv_in_1 - yv_in_0);


                //DTYPE p = b / 12.f;
                //DTYPE p = b / 12.f;
                DTYPE p = b / (4.f / (L_inv.x * L_inv.x) + 4.f / (L_inv.y * L_inv.y) + 4.f / (L_inv.z * L_inv.z));
                zv_in_1 += 2.f * p / L_inv.z;
                zv_in_0 -= 2.f * p / L_inv.z;
                xv_in_1 += 2.f * p / L_inv.x;
                xv_in_0 -= 2.f * p / L_inv.x;
                yv_in_1 += 2.f * p / L_inv.y;
                yv_in_0 -= 2.f * p / L_inv.y;
            }



            DTYPE lwt_s_x_000 = 0.f;
            DTYPE lwt_s_x_010 = lwt_s_x_000 - zv_in_0 / L_inv.y;
            DTYPE lwt_s_x_001 = lwt_s_x_000 + yv_in_0 / L_inv.z;
            DTYPE lwt_s_x_011 = lwt_s_x_010 + yv_in_1 / L_inv.z;

            DTYPE lwt_s_y_000 = 0.f;
            DTYPE lwt_s_y_100 = 0.f;
            DTYPE lwt_s_y_001 = lwt_s_y_000 - xv_in_0 / L_inv.z;
            DTYPE lwt_s_y_101 = lwt_s_y_100 - xv_in_1 / L_inv.z;

            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            lwt_s_z_o = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = lwt_s_x_000;
            q_lwt_x[idx_qx - slice_qx] = lwt_s_x_010;
            q_lwt_x[idx_qx - 1] = lwt_s_x_001;
            lwt_s_x_o = lwt_s_x_011;

            q_lwt_y[idx_qy - dim.y - slice_qy] = lwt_s_y_000;
            q_lwt_y[idx_qy - slice_qy] = lwt_s_y_100;
            q_lwt_y[idx_qy - dim.y] = lwt_s_y_001;
            lwt_s_y_o = lwt_s_y_101;
        }

        q_lwt_x[idx_qx] = lwt_s_x_o;
        q_lwt_y[idx_qy] = lwt_s_y_o;
        q_lwt_z[idx_qz] = lwt_s_z_o;
    }
}


__global__ void cuProjectLocal_q_3d_dq_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z,
    DTYPE* xv_lwt_out, DTYPE* yv_lwt_out, DTYPE* zv_lwt_out, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* ip_base_x, DTYPE* ip_base_y,
    DTYPE* ip_base_z, DTYPE* ip_proj_x, DTYPE* ip_proj_y, DTYPE* ip_proj_z, int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);

    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
        int3 dim_dst_ext = { dim_dst.x + 1, dim_dst.y + 1, dim_dst.z + 1 };
        int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
        int slice_xy_ext = dim_dst_ext.x * dim_dst_ext.y;
        int slice_xy = dim_dst.x * dim_dst.y;
        int slice_qz = dim_dst_ext.x * dim_dst_ext.y;
        int slice_qx = dim_dst.x * dim_dst_ext.y;
        int slice_qy = dim_dst_ext.x * dim_dst.y;

        int idx_x_0 = data_idx3.x + 1;
        int idx_y_0 = data_idx3.y + 1;
        int idx_z_0 = data_idx3.z + 1;

        int idx_xv = data_idx3.y + (idx_x_0 << levels.y) + (slice_xy + dim_dst.y) * data_idx3.z;
        int idx_yv = idx_y_0 + data_idx3.x * (dim_dst.y + 1) + (slice_xy + dim_dst.x) * data_idx3.z;
        int idx_zv = data_idx3.y + data_idx3.x * dim_dst.y + slice_xy * idx_z_0;

        int idx_qx = idx_y_0 + data_idx3.x * dim_dst_ext.y + idx_z_0 * (slice_xy + dim_dst.x);
        int idx_qy = data_idx3.y + idx_x_0 * dim_dst.y + idx_z_0 * (slice_xy + dim_dst.y);
        int idx_qz = idx_y_0 + idx_x_0 * dim_dst_ext.y + data_idx3.z * slice_xy_ext;

        int jy = levels.y - (32 - __clz(data_idx3.y)) + 1;
        int jx = levels.x - (32 - __clz(data_idx3.x)) + 1;
        int jz = levels.z - (32 - __clz(data_idx3.z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE ip_0_x = ip_proj_x[data_idx3.x];
        DTYPE ip_1_x = ip_base_x[data_idx3.x + 1];

        DTYPE ip_0_y = ip_proj_y[data_idx3.y];
        DTYPE ip_1_y = ip_base_y[data_idx3.y + 1];

        DTYPE ip_0_z = ip_proj_z[data_idx3.z];
        DTYPE ip_1_z = ip_base_z[data_idx3.z + 1];

        DTYPE xv_in = xv_lwt[idx_xv];
        DTYPE yv_in = yv_lwt[idx_yv];
        DTYPE zv_in = zv_lwt[idx_zv];

        DTYPE sxBxBy = scale_i_x * ip_0_x * ip_1_y * ip_0_z;
        DTYPE syBxBy = scale_i_y * ip_1_x * ip_0_y * ip_0_z;
        DTYPE sxBxBz = scale_i_x * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szBxBz = scale_i_z * ip_1_x * ip_0_y * ip_0_z;
        DTYPE syByBz = scale_i_y * ip_0_x * ip_0_y * ip_1_z;
        DTYPE szByBz = scale_i_z * ip_0_x * ip_1_y * ip_0_z;

        DTYPE ip_mid_x_inv = 1.f / (scale_i_y * syByBz + scale_i_z * szByBz + eps<DTYPE>);
        DTYPE ip_mid_y_inv = 1.f / (scale_i_x * sxBxBz + scale_i_z * szBxBz + eps<DTYPE>);
        DTYPE ip_mid_z_inv = 1.f / (scale_i_x * sxBxBy + scale_i_y * syBxBy + eps<DTYPE>);

        DTYPE lwt_s_x = 0.f;
        DTYPE lwt_s_y = 0.f;
        DTYPE lwt_s_z = 0.f;
        DTYPE lwt_s_x_t = 0.f;
        DTYPE lwt_s_y_t = 0.f;
        DTYPE lwt_s_z_t = 0.f;

        DTYPE fx = yv_in * szByBz - zv_in * syByBz;
        DTYPE fy = zv_in * sxBxBz - xv_in * szBxBz;
        DTYPE fz = xv_in * syBxBy - yv_in * sxBxBy;

        if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        {
            //DTYPE proj_mat[4];
            //proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
            //proj_mat[1] = -scale_i_x * syByBz;
            //proj_mat[2] = proj_mat[1];
            ////proj_mat[2] = -scale_i_y * sxBxBz;
            //proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

            //DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
            //proj_mat[0] *= proj_mat_bot_inv;
            //proj_mat[1] *= -proj_mat_bot_inv;
            //proj_mat[2] *= -proj_mat_bot_inv;
            //proj_mat[3] *= proj_mat_bot_inv;

            //lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;

            //yv_in -= scale_i_z * lwt_s_x;
            //zv_in += scale_i_y * lwt_s_x;

            //xv_in += scale_i_z * lwt_s_y;
            //zv_in -= scale_i_x * lwt_s_y;

            //fx = yv_in * szByBz - zv_in * syByBz;
            //fy = zv_in * sxBxBz - xv_in * szBxBz;

            //lwt_s_x = lwt_s_x + proj_mat[0] * fx + proj_mat[1] * fy;
            //lwt_s_y = lwt_s_y + proj_mat[2] * fx + proj_mat[3] * fy;

            DTYPE scale_i_y_no4 = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
            DTYPE scale_i_x_no4 = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
            DTYPE scale_i_z_no4 = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

            DTYPE proj_mat_bot_inv = 1.f / (scale_i_y_no4 * scale_i_y_no4 + scale_i_x_no4 * scale_i_x_no4 + scale_i_z_no4 * scale_i_z_no4);
            lwt_s_z = (scale_i_y_no4 * xv_in - scale_i_x_no4 * yv_in) * proj_mat_bot_inv;
            lwt_s_x = (scale_i_z_no4 * yv_in - scale_i_y_no4 * zv_in) * proj_mat_bot_inv;
            lwt_s_y = (scale_i_x_no4 * zv_in - scale_i_z_no4 * xv_in) * proj_mat_bot_inv;
        }
        else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        {
            //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
            lwt_s_x = fx * ip_mid_x_inv;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        {
            //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
            lwt_s_y = fy * ip_mid_y_inv;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        {
            //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
            lwt_s_z = fz * ip_mid_z_inv;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x > 0 && data_idx3.z == 0) // calculate L0L1L1  ** x line
        {
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx - 1] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
        }
        else if (data_idx3.y > 0 && data_idx3.x == 0 && data_idx3.z == 0) // calculate L1L0L1  ** y line
        {
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }
        else if (data_idx3.y == 0 && data_idx3.x == 0 && data_idx3.z > 0) // calculate L1L1L0  ** z line
        {
            q_lwt_x[idx_qx - 1] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;

            q_lwt_z[idx_qz - dim_ext.y - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
        }
        else
        {
            q_lwt_z[idx_qz - 1 - dim_ext.y] = 0.f;
            q_lwt_z[idx_qz - 1] = 0.f;
            q_lwt_z[idx_qz - dim_ext.y] = 0.f;

            q_lwt_x[idx_qx - 1 - slice_qx] = 0.f;
            q_lwt_x[idx_qx - slice_qx] = 0.f;
            q_lwt_x[idx_qx - 1] = 0.f;

            q_lwt_y[idx_qy - dim.y - slice_qy] = 0.f;
            q_lwt_y[idx_qy - slice_qy] = 0.f;
            q_lwt_y[idx_qy - dim.y] = 0.f;
        }

        /////////////////////---------------------------------second-----------------------////////////////////////////
        //lwt_s_x_t += lwt_s_x;
        //lwt_s_y_t += lwt_s_y;
        //lwt_s_z_t += lwt_s_z;

        //yv_in -= scale_i_z * lwt_s_x;
        //zv_in += scale_i_y * lwt_s_x;
        //xv_in += scale_i_z * lwt_s_y;
        //zv_in -= scale_i_x * lwt_s_y;
        //xv_in -= scale_i_y * lwt_s_z;
        //yv_in += scale_i_x * lwt_s_z;

        //fx = yv_in * szByBz - zv_in * syByBz;
        //fy = zv_in * sxBxBz - xv_in * szBxBz;
        //fz = xv_in * syBxBy - yv_in * sxBxBy;

        //if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y > 0)
        //{
        //    DTYPE proj_mat[4];
        //    proj_mat[0] = scale_i_z * szBxBz + scale_i_x * sxBxBz;
        //    proj_mat[1] = -scale_i_x * syByBz;
        //    proj_mat[2] = proj_mat[1];
        //    //proj_mat[2] = -scale_i_y * sxBxBz;
        //    proj_mat[3] = scale_i_z * szByBz + scale_i_y * syByBz;

        //    DTYPE proj_mat_bot_inv = 1.f / (proj_mat[0] * proj_mat[3] - proj_mat[1] * proj_mat[2] + eps<DTYPE>);
        //    proj_mat[0] *= proj_mat_bot_inv;
        //    proj_mat[1] *= -proj_mat_bot_inv;
        //    proj_mat[2] *= -proj_mat_bot_inv;
        //    proj_mat[3] *= proj_mat_bot_inv;

        //    lwt_s_x = proj_mat[0] * fx + proj_mat[1] * fy;
        //    lwt_s_y = proj_mat[2] * fx + proj_mat[3] * fy;
        //}
        //else if (data_idx3.y > 0 && data_idx3.z > 0 && data_idx3.x == 0)
        //{
        //    //DTYPE proj_i_x = yv_in * szByBz - zv_in * syByBz;
        //    lwt_s_x = fx * ip_mid_x_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.z > 0 && data_idx3.y == 0)
        //{
        //    //DTYPE proj_i_y = zv_in * sxBxBz - xv_in * szBxBz;
        //    lwt_s_y = fy * ip_mid_y_inv;

        //}
        //else if (data_idx3.x > 0 && data_idx3.y > 0 && data_idx3.z == 0)
        //{
        //    //DTYPE proj_i_z = xv_in * syBxBy - yv_in * sxBxBy;
        //    lwt_s_z = fz * ip_mid_z_inv;

        //}

        q_lwt_x[idx_qx] = lwt_s_x + lwt_s_x_t;
        q_lwt_y[idx_qy] = lwt_s_y + lwt_s_y_t;
        q_lwt_z[idx_qz] = lwt_s_z + lwt_s_z_t;

        DTYPE yv_out = scale_i_z * lwt_s_x;
        DTYPE zv_out = -scale_i_y * lwt_s_x;

        DTYPE xv_out = -scale_i_z * lwt_s_y;
        zv_out += scale_i_x * lwt_s_y;

        xv_out += scale_i_y * lwt_s_z;
        yv_out += -scale_i_x * lwt_s_z;

        xv_lwt_out[idx_xv] = xv_out;
        yv_lwt_out[idx_yv] = yv_out;
        zv_lwt_out[idx_zv] = zv_out;
    }
}

void CuDivLocalProject3D::ProjectLocal_q_ccc(DTYPE* q_lwt_x, DTYPE* q_lwt_y, DTYPE* q_lwt_z, 
    DTYPE* xv_lwt_out, DTYPE* yv_lwt_out, DTYPE* zv_lwt_out,
    DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, int3 levels)
{
    int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    int threads_x = bc_base_ == 'd' ? 32 : 16;

    int blocks_y = std::ceil(double(dim.y - 1) / threads_y);
    int blocks_x = std::ceil(double(dim.x - 1) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d\n", block.x, block.y, block.z);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    if (bc_base_ == 'n')
    {
        //cuProjectLocal_3d_z_n << <grid, block, sm_size >> > (p_lwt, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_, l_halo, r_halo);
        //cuProjectLocal_q_3d_n_ccc << <grid, block >> > (xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
        cuProjectLocal_q_3d_nq_ccc << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt_out, yv_lwt_out, zv_lwt_out, 
            xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    else
    {
        cuProjectLocal_q_3d_dq_ccc << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt_out, yv_lwt_out, zv_lwt_out, 
            xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
            ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
        //cuProjectLocal_q_3d_d << <grid, block >> > (q_lwt_x, q_lwt_y, q_lwt_z, xv_lwt, yv_lwt, zv_lwt, ip_base_x_, ip_base_y_, ip_base_z_,
        //    ip_proj_x_, ip_proj_y_, ip_proj_z_, dim, levels_, dx_inv_);
    }
    checkCudaErrors(cudaGetLastError());
}