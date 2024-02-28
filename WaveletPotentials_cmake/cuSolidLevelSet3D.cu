#include "cuSolidLevelSet3D.cuh"
#include "cuMemoryManager.cuh"
#include "cuMemoryManager.cuh"
#include "cudaMath.cuh"
#include "Interpolation.cuh"
#include "cuConvergence.cuh"
#include "cuGradient.cuh"
#include "cudaMath.cuh"
#include "StreamInitial.h"

__global__ void cuInitLsSphere(DTYPE* dst, int3 dim, DTYPE3 center, DTYPE radius, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - idx_z * dim.x;

        DTYPE3 pos = make_DTYPE3((idx_x + 0.5f) * dx.x, (idx_y + 0.5f) * dx.y, (idx_z + 0.5f) * dx.z);
        DTYPE dis = length(pos - center);
        dst[idx] = dis - radius;
    }
}

__device__ DTYPE cuComputeFrac(DTYPE dx, DTYPE ls_l, DTYPE ls_r)
{
    DTYPE ls_dif = ls_r - ls_l;
    DTYPE d = sqrtf(max(dx * dx - ls_dif * ls_dif, 1e-14f));
    //return clamp(((ls_r + ls_l) / d - 1.f) * 0.5f, 0.f, 1.f);
    return (1.f - clamp((1.f - (ls_r + ls_l) / d) * 0.5f, 0.f, 1.f));
}

__global__ void cuGetAx(DTYPE* ax, DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ax_x = dim.x + 1;
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim_ax_x);
        int idx_x = idx_xz - idx_z * (dim_ax_x);

        int idx_ls_r = idx_y + min(idx_x, dim.x - 1) * dim.y + idx_z * dim.x * dim.y;
        int idx_ls_l = idx_y + max(idx_x - 1, 0) * dim.y + idx_z * dim.x * dim.y;

        ax[idx] = cuComputeFrac(dx, ls[idx_ls_l], ls[idx_ls_r]);
    }
}

__global__ void cuGetAy(DTYPE* ay, DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ay_y = dim.y + 1;
        int idx_xz = idx / dim_ay_y;
        int idx_y = idx - idx_xz * dim_ay_y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        int idx_ls_r = min(idx_y, dim.y - 1) + idx_x * dim.y + idx_z * dim.x * dim.y;
        int idx_ls_l = max(idx_y - 1, 0) + idx_x * dim.y + idx_z * dim.x * dim.y;

        ay[idx] = cuComputeFrac(dx, ls[idx_ls_l], ls[idx_ls_r]);
    }
}

__global__ void cuGetAz(DTYPE* az, DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        int idx_ls_r = idx_y + idx_x * dim.y + min(idx_z, dim.z - 1) * dim.x * dim.y;
        int idx_ls_l = idx_y + idx_x * dim.y + max(idx_z - 1, 0) * dim.x * dim.y;

        az[idx] = cuComputeFrac(dx, ls[idx_ls_l], ls[idx_ls_r]);
    }
}

__global__ void cuGetAx_Interp(DTYPE* ax, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        
        int dim_ax_x = dim.x + 1;
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim_ax_x);
        int idx_x = idx_xz - idx_z * (dim_ax_x);

        DTYPE inv_dx_c = 1.f / dx_c;
        DTYPE3 xg_r = (make_DTYPE3(idx_x + 0.5f, idx_y + 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_r = InterpolateQuadratic3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        //DTYPE ls_r = InterpolateLinear3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        DTYPE3 xg_l = (make_DTYPE3(idx_x - 0.5f, idx_y + 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_l = InterpolateQuadratic3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);
        //DTYPE ls_l = InterpolateLinear3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);

        ax[idx] = cuComputeFrac(dx, ls_l, ls_r);
    }
}

__global__ void cuGetAy_Interp(DTYPE* ay, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ay_y = dim.y + 1;
        int idx_xz = idx / dim_ay_y;
        int idx_y = idx - idx_xz * dim_ay_y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        DTYPE inv_dx_c = 1.f / dx_c;
        DTYPE3 xg_r = (make_DTYPE3(idx_x + 0.5f, idx_y + 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_r = InterpolateQuadratic3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        //DTYPE ls_r = InterpolateLinear3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        DTYPE3 xg_l = (make_DTYPE3(idx_x + 0.5f, idx_y - 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_l = InterpolateQuadratic3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);
        //DTYPE ls_l = InterpolateLinear3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);

        ay[idx] = cuComputeFrac(dx, ls_l, ls_r);
    }
}

__global__ void cuGetAz_Interp(DTYPE* az, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        DTYPE inv_dx_c = 1.f / dx_c;
        DTYPE3 xg_r = (make_DTYPE3(idx_x + 0.5f, idx_y + 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_r = InterpolateQuadratic3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        //DTYPE ls_r = InterpolateLinear3D(ls, xg_r.x, xg_r.y, xg_r.z, dim_c);
        DTYPE3 xg_l = (make_DTYPE3(idx_x + 0.5f, idx_y + 0.5f, idx_z - 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_l = InterpolateQuadratic3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);
        //DTYPE ls_l = InterpolateLinear3D(ls, xg_l.x, xg_l.y, xg_l.z, dim_c);

        az[idx] = cuComputeFrac(dx, ls_l, ls_r);
    }
}

__global__ void cuGetAx_divide(DTYPE* ax, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ax_x = dim.x + 1;
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim_ax_x);
        int idx_x = idx_xz - idx_z * (dim_ax_x);

        DTYPE inv_dx_c = 1.f / dx_c;
        
        DTYPE3 xg = (make_DTYPE3(idx_x, idx_y + 0.5f, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x, xg.y, xg.z, dim_c);
        DTYPE res = 0.f;
        if (ls_p > 1.f * dx)
        {
            res = 1.f;
        }
        else if (ls_p > -1.f * dx)
        {
            DTYPE pos_x = idx_x * dx;
            const int sub_face = 128;
            const DTYPE dx_sub = dx / sub_face;
            const DTYPE dxdx_inv = 1.f / (sub_face * sub_face);

            DTYPE pos_y_base = idx_y * dx + 0.5f * dx_sub;
            DTYPE pos_z_base = idx_z * dx + 0.5f * dx_sub;

            int total_pos_face = 0.f;
#pragma unroll
            for (int k = 0; k < sub_face; k++)
            {
                for (int i = 0; i < sub_face; i++)
                {
                    DTYPE3 xg_sub_c = (make_DTYPE3(pos_x, pos_y_base + i * dx_sub, pos_z_base + k * dx_sub) - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
                    ls_p = InterpolateLinear3D(ls, xg_sub_c.x, xg_sub_c.y, xg_sub_c.z, dim_c);
                    if (ls_p > 0.f)
                    {
                        total_pos_face += 1;
                    }
                }
            }
            res = total_pos_face * dxdx_inv;
        }

        ax[idx] = clamp(res, 0.f, 1.f);
    }
}

__global__ void cuGetAy_divide(DTYPE* ay, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ay_y = dim.y + 1;
        int idx_xz = idx / dim_ay_y;
        int idx_y = idx - idx_xz * dim_ay_y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        DTYPE inv_dx_c = 1.f / dx_c;

        DTYPE3 xg = (make_DTYPE3(idx_x + 0.5f, idx_y, idx_z + 0.5f) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x, xg.y, xg.z, dim_c);
        DTYPE res = 0.f;
        if (ls_p > 1.f * dx)
        {
            res = 1.f;
        }
        else if (ls_p > -1.f * dx)
        {
            DTYPE pos_y = idx_y * dx;
            const int sub_face = 128;
            const DTYPE dx_sub = dx / sub_face;
            const DTYPE dxdx_inv = 1.f / (sub_face * sub_face);

            DTYPE pos_x_base = idx_x * dx + 0.5f * dx_sub;
            DTYPE pos_z_base = idx_z * dx + 0.5f * dx_sub;

            int total_pos_face = 0;
#pragma unroll
            for (int k = 0; k < sub_face; k++)
            {
                for (int j = 0; j < sub_face; j++)
                {
                    DTYPE3 xg_sub_c = (make_DTYPE3(pos_x_base + j * dx_sub, pos_y, pos_z_base + k * dx_sub) - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
                    ls_p = InterpolateLinear3D(ls, xg_sub_c.x, xg_sub_c.y, xg_sub_c.z, dim_c);
                    if (ls_p > 0.f)
                    {
                        total_pos_face += 1;
                    }
                }
            }
            res = total_pos_face * dxdx_inv;
        }

        ay[idx] = res;
    }
}

__global__ void cuGetAz_divide(DTYPE* az, DTYPE* ls, int3 dim, int3 dim_c, DTYPE dx, DTYPE dx_c, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        DTYPE inv_dx_c = 1.f / dx_c;

        DTYPE3 xg = (make_DTYPE3(idx_x + 0.5f, idx_y + 0.5f, idx_z) * dx - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x, xg.y, xg.z, dim_c);
        DTYPE res = 0.f;
        if (ls_p > 1.f * dx)
        {
            res = 1.f;
        }
        else if (ls_p > -1.f * dx)
        {
            DTYPE pos_z = idx_z * dx;
            const int sub_face = 128;
            const DTYPE dx_sub = dx / sub_face;
            const DTYPE dxdx_inv = 1.f / (sub_face * sub_face);

            DTYPE pos_x_base = idx_x * dx + 0.5f * dx_sub;
            DTYPE pos_y_base = idx_y * dx + 0.5f * dx_sub;

            int total_pos_face = 0;
#pragma unroll
            for (int j = 0; j < sub_face; j++)
            {
                for (int i = 0; i < sub_face; i++)
                {
                    DTYPE3 xg_sub_c = (make_DTYPE3(pos_x_base + j * dx_sub, pos_y_base + i * dx_sub, pos_z) - c_min) * inv_dx_c - make_DTYPE3(0.5f, 0.5f, 0.5f);
                    ls_p = InterpolateLinear3D(ls, xg_sub_c.x, xg_sub_c.y, xg_sub_c.z, dim_c);
                    if (ls_p > 0.f)
                    {
                        total_pos_face += 1;
                    }
                }
            }
            res = total_pos_face * dxdx_inv;
        }

        az[idx] = res;
    }
}

CuSolidLevelSet3D::CuSolidLevelSet3D(int3 dim, DTYPE3 dx) : dim_(dim), dx_(dx)
{
    dim_ax_ = { dim.x + 1, dim.y, dim.z };
    dim_ay_ = { dim.x, dim.y + 1, dim.z };
    dim_az_ = { dim.x, dim.y, dim.z + 1 };
    size_ = dim.x * dim.y * dim.z;
    ls_ = CuMemoryManager::GetInstance()->GetData("solidls", size_);

    size_ax_ = getsize(dim_ax_);
    size_ay_ = getsize(dim_ay_);
    size_az_ = getsize(dim_az_);
}

CuSolidLevelSet3D::CuSolidLevelSet3D(int3 dim, DTYPE3 dx, DTYPE dx_c) : dim_(dim), dx_(dx)
{
    dim_ax_ = { dim.x + 1, dim.y, dim.z };
    dim_ay_ = { dim.x, dim.y + 1, dim.z };
    dim_az_ = { dim.x, dim.y, dim.z + 1 };
    size_ = dim.x * dim.y * dim.z;
    ls_ = CuMemoryManager::GetInstance()->GetData("solidls", size_);

    size_ax_ = getsize(dim_ax_);
    size_ay_ = getsize(dim_ay_);
    size_az_ = getsize(dim_az_);
}

CuSolidLevelSet3D::~CuSolidLevelSet3D()
{
}

void CuSolidLevelSet3D::GetFrac(DTYPE* ax, DTYPE* ay, DTYPE* az)
{
    cuGetAx << <BLOCKS(size_ax_), THREADS(size_ax_) >> > (ax, ls_, dim_, dx_.x, size_ax_);
    cuGetAy << <BLOCKS(size_ay_), THREADS(size_ay_) >> > (ay, ls_, dim_, dx_.y, size_ay_);
    cuGetAz << <BLOCKS(size_az_), THREADS(size_az_) >> > (az, ls_, dim_, dx_.z, size_az_);
}

void CuSolidLevelSet3D::GetFrac(DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, int3 dim, DTYPE3 dx)
{
    int3 dim_ax = dim + make_int3(1, 0, 0);
    int3 dim_ay = dim + make_int3(0, 1, 0);
    int3 dim_az = dim + make_int3(0, 0, 1);
    int size_ax = getsize(dim_ax);
    int size_ay = getsize(dim_ay);
    int size_az = getsize(dim_az);

    cuGetAx_divide << <BLOCKS(size_ax), THREADS(size_ax) >> > (ax, ls, dim, dim, dx.x, dx.x, { 0.f, 0.f, 0.f }, size_ax);
    cuGetAy_divide << <BLOCKS(size_ay), THREADS(size_ay) >> > (ay, ls, dim, dim, dx.y, dx.y, { 0.f, 0.f, 0.f }, size_ay);
    cuGetAz_divide << <BLOCKS(size_az), THREADS(size_az) >> > (az, ls, dim, dim, dx.z, dx.z, { 0.f, 0.f, 0.f }, size_az);

    //cuGetAx << <BLOCKS(size_ax), THREADS(size_ax) >> > (ax, ls, dim, dx.x, size_ax);
    //cuGetAy << <BLOCKS(size_ay), THREADS(size_ay) >> > (ay, ls, dim, dx.y, size_ay);
    //cuGetAz << <BLOCKS(size_az), THREADS(size_az) >> > (az, ls, dim, dx.z, size_az);
}

void CuSolidLevelSet3D::InitLsSphere(DTYPE3 center, DTYPE radius)
{
    cuInitLsSphere << <BLOCKS(size_), THREADS(size_) >> > (ls_, dim_, center, radius, dx_, size_);
}

void CuSolidLevelSet3D::InitLsSphere(DTYPE* ls, DTYPE3 center, DTYPE radius, int3 dim, DTYPE3 dx)
{
    int size = getsize(dim);
    cuInitLsSphere << <BLOCKS(size), THREADS(size) >> > (ls, dim, center, radius, dx, size);
}