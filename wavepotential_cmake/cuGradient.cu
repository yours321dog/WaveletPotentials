#include "cuGradient.cuh"
#include "cuMemoryManager.cuh"
#include "cuWeightedJacobi3D.cuh"

std::auto_ptr<CuGradient> CuGradient::instance_;

__global__ void cuGradientMinusLwtP_n(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_x = idx >> levels.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE p_lwt = p[idx];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtP_d(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        //idxD1 is the index of columm, idxD2 is the index of row
        DTYPE2 L_inv = { dx_inv.x / dim.x, dx_inv.y / dim.y };
        //DTYPE2 L = { 1.f / L_inv.x, 1.f / L_inv.y };

        int idx_x = idx >> levels.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE p_lwt = p[idx];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0) // calculate L01
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;
        }
        else if (idx_y > 0 && idx_x == 0) // calculate 1L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim.y] -= p_lwt * L_inv.x * 2.f;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            xv[0] -= p_lwt_2 * L_inv.x;
            yv[0] -= p_lwt_2 * L_inv.y;
            yv_res += p_lwt_2 * L_inv.y;
        }

        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtQ_n(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

    if (idx < max_size)
    {
        DTYPE2 L_inv = { dx_inv.x / dim.x, dx_inv.y / dim.y };

        int idx_x = idx >> levels.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim.y + 1);
        int idx_q = (idx_y + 1) + (idx_x + 1) * (dim.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE q_lwt = q[idx_q];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= q_lwt * scale_i_y;
            yv_res += q_lwt * scale_i_x;
        }
        else if (idx_y == 0 && idx_x > 0) // calculate L01
        {
            DTYPE q_lwt_1 = q[idx_q - 1];
            xv_res -= (q_lwt - q_lwt_1) * L_inv.y;
            yv_res += q_lwt * scale_i_x;
            yv[idx_yv - 1] += q_lwt_1 * scale_i_x;
        }
        else if (idx_y > 0 && idx_x == 0) // calculate 1L0
        {
            DTYPE q_lwt_1 = q[idx_q - (dim.y + 1)];
            xv[idx_xv - dim.y] -= scale_i_y * q_lwt_1;
            xv_res -= scale_i_y * q_lwt;

            int idx_yv = idx_y + 1;
            yv_res += (q_lwt - q_lwt_1) * L_inv.x;
        }
        else
        {
            DTYPE q_lwt_00 = q[0];
            DTYPE q_lwt_01 = q[1];
            DTYPE q_lwt_10 = q[dim.y + 1];

            xv[0] -= (q_lwt_01 - q_lwt_00) * L_inv.y;
            xv_res -= (q_lwt - q_lwt_10) * L_inv.y;
            yv[0] += (q_lwt_10 - q_lwt_00) * L_inv.x;
            yv_res += (q_lwt - q_lwt_01) * L_inv.x;
        }
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtQ_d(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim.x, dx_inv.y / dim.y);

    if (idx < max_size)
    {
        DTYPE2 L_inv = { dx_inv.x / dim.x, dx_inv.y / dim.y };

        int idx_x = idx >> levels.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim.y + 1);
        int idx_q = (idx_y + 1) + (idx_x + 1) * (dim.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE q_lwt = q[idx_q];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= q_lwt * scale_i_y;
            yv_res += q_lwt * scale_i_x;
        }
        
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtP_n(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };

    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx & (dim.y - 1);

        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_p = idx_y + idx_x * dim_dst.y;
        int idx_xv = idx_p + dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;

        //printf("idx: (%d, %d), val: %f\n", idx_x, idx_y, p_lwt);
    }
}

__global__ void cuGradientMinusLwtP_d(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    DTYPE2 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y };

    if (idx < max_size)
    {
        //idxD1 is the index of columm, idxD2 is the index of row
        //DTYPE2 L = { 1.f / L_inv.x, 1.f / L_inv.y };

        int idx_x = idx / dim.y;
        int idx_y = idx & (dim.y - 1);

        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_p = idx_y + idx_x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0) // calculate L01
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;
        }
        else if (idx_y > 0 && idx_x == 0) // calculate 1L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            xv[0] -= p_lwt_2 * L_inv.x;
            yv[0] -= p_lwt_2 * L_inv.y;
            yv_res += p_lwt_2 * L_inv.y;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtP_z(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    DTYPE2 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y };

    if (idx < max_size)
    {
        //idxD1 is the index of columm, idxD2 is the index of row
        //DTYPE2 L = { 1.f / L_inv.x, 1.f / L_inv.y };

        int idx_x = idx / dim.y;
        int idx_y = idx & (dim.y - 1);

        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_p = idx_y + idx_x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0) // calculate L01
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res += p_lwt * L_inv.y;
            yv[idx_yv - 1] -= p_lwt * L_inv.y;
        }
        else if (idx_y > 0 && idx_x == 0) // calculate 1L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x;
        }
        else
        {
            xv_res += p_lwt * L_inv.x;
            xv[0] -= p_lwt * L_inv.x;
            yv[0] -= p_lwt * L_inv.y;
            yv_res += p_lwt * L_inv.y;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
    }
}

__global__ void cuGradientMinusLwtQ_n(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y);

    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_q = (idx_y + 1) + (idx_x + 1) * (dim_dst.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE q_lwt = q[idx_q];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= q_lwt * scale_i_y;
            yv_res += q_lwt * scale_i_x;
        }
        else if (idx_y == 0 && idx_x > 0) // calculate L01
        {
            DTYPE q_lwt_1 = q[idx_q - 1];
            xv_res -= (q_lwt - q_lwt_1) * L_inv.y;
            yv_res += q_lwt * scale_i_x;
            yv[idx_yv - 1] += q_lwt_1 * scale_i_x;

            new_q[idx_q - 1] += q_lwt_1;
        }
        else if (idx_y > 0 && idx_x == 0) // calculate 1L0
        {
            DTYPE q_lwt_1 = q[idx_q - (dim_dst.y + 1)];
            xv[idx_xv - dim_dst.y] -= scale_i_y * q_lwt_1;
            xv_res -= scale_i_y * q_lwt;

            int idx_yv = idx_y + 1;
            yv_res += (q_lwt - q_lwt_1) * L_inv.x;

            new_q[idx_q - (dim_dst.y + 1)] += q_lwt_1;
        }
        else
        {
            DTYPE q_lwt_00 = q[0];
            DTYPE q_lwt_01 = q[1];
            DTYPE q_lwt_10 = q[dim_dst.y + 1];

            xv[0] -= (q_lwt_01 - q_lwt_00) * L_inv.y;
            xv_res -= (q_lwt - q_lwt_10) * L_inv.y;
            yv[0] += (q_lwt_10 - q_lwt_00) * L_inv.x;
            yv_res += (q_lwt - q_lwt_01) * L_inv.x;

            new_q[0] += q_lwt_00;
            new_q[1] += q_lwt_01;
            new_q[dim_dst.y + 1] += q_lwt_10;
        }
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        new_q[idx_q] += q_lwt;
        //printf("idx: (%d, %d): q_lwt: %f, new_q[idx_q]: %f, q[idx_q]: %f\n", idx_x, idx_y, q_lwt, new_q[idx_q], q[idx_q]);
    }
}

__global__ void cuGradientMinusLwtQ_d(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int2 dim_dst = { 1 << levels.x, 1 << levels.y };
    DTYPE2 L_inv = make_DTYPE2(dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y);

    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx & (dim.y - 1);

        int idx_xv = idx_y + ((idx_x + 1) << levels.y);
        int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);
        int idx_q = (idx_y + 1) + (idx_x + 1) * (dim_dst.y + 1);

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);

        DTYPE q_lwt = q[idx_q];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0)
        {
            xv_res -= q_lwt * scale_i_y;
            yv_res += q_lwt * scale_i_x;
        }

        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        new_q[idx_q] += q_lwt;
        //printf("idx: %d(%d, %d), idx_xv: %d, idx_yv: %d, idx_q: %d,  val: %f, xv: %f, yv: %f, new_q[idx_q]: %f\n", 
        //    idx, idx_x, idx_y, idx_xv, idx_yv, idx_q, q_lwt, xv_res, yv_res, new_q[idx_q]);
    }
}

__global__ void cuGradientMinusLwtP_n(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_dst = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx & (dim.y - 1);
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz & (dim.x - 1);
        int slice_xy = dim_dst.x * dim_dst.y;
        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        if (idx_z > 0)
        {
            zv_res -= scale_i_z * p_lwt;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;

        //printf("idx: (%d, %d), val: %f\n", idx_x, idx_y, p_lwt);
    }
}

__global__ void cuGradientMinusLwtP_d(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, 
    int3 dim, int3 levels, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
    DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx & (dim.y - 1);
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz & (dim.x - 1);
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0 && idx_z > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0
        {
            xv_res -= scale_i_x * p_lwt;

            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z > 0) // calculate L1L0L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x > 0 && idx_z == 0) // calculate L0L0L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z == 0) // calculate L1L0L1
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z == 0) // calculate L0L1L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x == 0 && idx_z > 0) // calculate L1L1L0
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            yv_res += p_lwt_2 * L_inv.y;
            zv_res += p_lwt_2 * L_inv.z;
            xv[0] -= p_lwt_2 * L_inv.x;
            yv[0] -= p_lwt_2 * L_inv.y;
            zv[0] -= p_lwt_2 * L_inv.z;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;
    }
}

__global__ void cuGradientMinusLwtP_n_block(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 dim_dst = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    int idx_y = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_x = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockDim.z * blockIdx.z;

    if (idx_x < dim.x && idx_y < dim.y)
    {
        int slice_xy = dim_dst.x * dim_dst.y;
        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        //DTYPE p_lwt = 0.f;
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        if (idx_z > 0)
        {
            zv_res -= scale_i_z * p_lwt;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;

        //printf("idx: (%d, %d), val: %f\n", idx_x, idx_y, p_lwt);
    }
}

__global__ void cuGradientMinusLwtP_d_block(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
    DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        const int& idx_y = data_idx3.y;
        const int& idx_x = data_idx3.x;
        const int& idx_z = data_idx3.z;

        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0 && idx_z > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res += p_lwt * L_inv.y * 2.f;

            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z > 0) // calculate L1L0L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x > 0 && idx_z == 0) // calculate L0L0L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z == 0) // calculate L1L0L1
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z == 0) // calculate L0L1L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x == 0 && idx_z > 0) // calculate L1L1L0
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            yv_res += p_lwt_2 * L_inv.y;
            zv_res += p_lwt_2 * L_inv.z;
            xv[0] -= p_lwt_2 * L_inv.x;
            yv[0] -= p_lwt_2 * L_inv.y;
            zv[0] -= p_lwt_2 * L_inv.z;
        }

        new_p[idx_p] += p_lwt;
        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;
    }
}

__global__ void cuGradientMinusLwtP_n(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_dst = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx & (dim.y - 1);
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz & (dim.x - 1);
        int slice_xy = dim_dst.x * dim_dst.y;
        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        if (idx_z > 0)
        {
            zv_res -= scale_i_z * p_lwt;
        }

        xv_lwt[idx_xv] = xv_res;
        yv_lwt[idx_yv] = yv_res;
        zv_lwt[idx_zv] = zv_res;

        //printf("idx: (%d, %d), val: %f\n", idx_x, idx_y, p_lwt);
    }
}

__global__ void cuGradientMinusLwtP_d(DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
    DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx & (dim.y - 1);
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz & (dim.x - 1);
        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0 && idx_z > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0
        {
            xv_res -= scale_i_x * p_lwt;

            yv_res += p_lwt * L_inv.y * 2.f;
            yv_lwt[idx_yv - 1] = yv[idx_yv - 1] - p_lwt * L_inv.y * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z > 0) // calculate L1L0L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv_lwt[idx_xv - dim_dst.y] = xv[idx_xv - dim_dst.y] - p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x > 0 && idx_z == 0) // calculate L0L0L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv_lwt[idx_zv - slice_xy] = zv[idx_zv - slice_xy] - 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z == 0) // calculate L1L0L1
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv_lwt[idx_yv - 1] = yv[idx_yv - 1] - p_lwt * L_inv.y * 2.f;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv_lwt[idx_zv - slice_xy] = zv[idx_zv - slice_xy] - 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z == 0) // calculate L0L1L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv_lwt[idx_xv - dim_dst.y] = xv[idx_xv - dim_dst.y] - p_lwt * L_inv.x * 2.f;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv_lwt[idx_zv - slice_xy] = zv[idx_zv - slice_xy] - 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x == 0 && idx_z > 0) // calculate L1L1L0
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv_lwt[idx_yv - 1] = yv[idx_yv - 1] - p_lwt * L_inv.y * 2.f;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv_lwt[idx_xv - dim_dst.y] = xv[idx_xv - dim_dst.y] - p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            yv_res += p_lwt_2 * L_inv.y;
            zv_res += p_lwt_2 * L_inv.z;
            xv_lwt[0] = xv[0] - p_lwt_2 * L_inv.x;
            yv_lwt[0] = yv[0] - p_lwt_2 * L_inv.y;
            zv_lwt[0] = zv[0] - p_lwt_2 * L_inv.z;
        }
        
        xv_lwt[idx_xv] = xv_res;
        yv_lwt[idx_yv] = yv_res;
        zv_lwt[idx_zv] = zv_res;
    }
}

__global__ void cuGradientMinusLwtP_n_block(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 dim_dst = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        const int& idx_y = data_idx3.y;
        const int& idx_x = data_idx3.x;
        const int& idx_z = data_idx3.z;
        int slice_xy = dim_dst.x * dim_dst.y;
        //int idx_xv = idx_y + ((idx_x + 1) * dim_dst.y);
        //int idx_yv = idx_y + 1 + idx_x * (dim_dst.y + 1);

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0)
        {
            yv_res -= scale_i_y * p_lwt;
        }

        if (idx_x > 0)
        {
            xv_res -= scale_i_x * p_lwt;
        }

        if (idx_z > 0)
        {
            zv_res -= scale_i_z * p_lwt;
        }

        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;

        //printf("idx: (%d, %d), val: %f\n", idx_x, idx_y, p_lwt);
    }
}

__global__ void cuGradientMinusLwtP_d_block(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p,
    int3 dim, int3 levels, DTYPE3 dx_inv)
{
    int3 dim_dst = { 1 << levels.x, 1 << levels.y , 1 << levels.z };
    DTYPE3 L_inv = { dx_inv.x / dim_dst.x, dx_inv.y / dim_dst.y, dx_inv.z / dim_dst.z };

    int3 thread_start_idx3 = make_int3(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int3 data_idx3 = make_int3(thread_start_idx3.y + threadIdx.y, thread_start_idx3.x + threadIdx.x,
        thread_start_idx3.z);

    if (data_idx3.x < dim.x && data_idx3.y < dim.y)
    {
        const int& idx_y = data_idx3.y;
        const int& idx_x = data_idx3.x;
        const int& idx_z = data_idx3.z;

        int slice_xy = dim_dst.x * dim_dst.y;

        int idx_p = idx_y + idx_x * dim_dst.y + idx_z * slice_xy;
        int idx_xv = idx_p + dim_dst.y + idx_z * dim_dst.y;
        int idx_yv = idx_p + 1 + idx_x + idx_z * dim_dst.x;
        int idx_zv = idx_p + dim_dst.x * dim_dst.y;

        int jy = levels.y - (32 - __clz(idx_y)) + 1;
        int jx = levels.x - (32 - __clz(idx_x)) + 1;
        int jz = levels.z - (32 - __clz(idx_z)) + 1;

        DTYPE scale_i_y = 4.f * dx_inv.y / ((1 << jy) + eps<DTYPE>);
        DTYPE scale_i_x = 4.f * dx_inv.x / ((1 << jx) + eps<DTYPE>);
        DTYPE scale_i_z = 4.f * dx_inv.z / ((1 << jz) + eps<DTYPE>);

        DTYPE p_lwt = p[idx_p];
        DTYPE xv_res = xv[idx_xv];
        DTYPE yv_res = yv[idx_yv];
        DTYPE zv_res = zv[idx_zv];

        // calculate L0L0
        if (idx_y > 0 && idx_x > 0 && idx_z > 0)
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res -= scale_i_y * p_lwt;
            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z > 0) // calculate L0L1L0
        {
            xv_res -= scale_i_x * p_lwt;
            yv_res += p_lwt * L_inv.y * 2.f;

            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z > 0) // calculate L1L0L0
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else if (idx_y > 0 && idx_x > 0 && idx_z == 0) // calculate L0L0L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x > 0 && idx_z == 0) // calculate L1L0L1
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res -= scale_i_x * p_lwt;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y > 0 && idx_x == 0 && idx_z == 0) // calculate L0L1L1
        {
            yv_res -= scale_i_y * p_lwt;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res += 2.f * L_inv.z * p_lwt;
            zv[idx_zv - slice_xy] -= 2.f * L_inv.z * p_lwt;
        }
        else if (idx_y == 0 && idx_x == 0 && idx_z > 0) // calculate L1L1L0
        {
            yv_res += p_lwt * L_inv.y * 2.f;
            yv[idx_yv - 1] -= p_lwt * L_inv.y * 2.f;

            xv_res += p_lwt * L_inv.x * 2.f;
            xv[idx_xv - dim_dst.y] -= p_lwt * L_inv.x * 2.f;

            zv_res -= scale_i_z * p_lwt;
        }
        else
        {
            DTYPE p_lwt_2 = p_lwt * 2.f;
            xv_res += p_lwt_2 * L_inv.x;
            yv_res += p_lwt_2 * L_inv.y;
            zv_res += p_lwt_2 * L_inv.z;
            xv[0] -= p_lwt_2 * L_inv.x;
            yv[0] -= p_lwt_2 * L_inv.y;
            zv[0] -= p_lwt_2 * L_inv.z;
        }

        xv[idx_xv] = xv_res;
        yv[idx_yv] = yv_res;
        zv[idx_zv] = zv_res;
    }
}

CuGradient* CuGradient::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuGradient>(new CuGradient); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void CuGradient::GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc)
{
    int2 levels = { std::log2(dim.x), std::log2(dim.y) };
    GradientMinusLwtP(xv, yv, p, dim, levels, dx, bc);
}

void CuGradient::GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx, char bc)
{
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim.x * dim.y;
    if (bc == 'n')
    {
        cuGradientMinusLwtP_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, levels, dx_inv,  max_size);
    }
    else
    {
        cuGradientMinusLwtP_d << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, levels, dx_inv, max_size);
    }
}

void CuGradient::GradientMinusLwtQ(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, DTYPE2 dx, char bc)
{
    int2 levels = { std::log2(dim.x), std::log2(dim.y) };
    GradientMinusLwtQ(xv, yv, q, dim, levels, dx, bc);
}

void CuGradient::GradientMinusLwtQ(DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx, char bc)
{
    int2 dim_p2 = { dim.x - (dim.x & 1), dim.y - (dim.y & 1) };
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim_p2.x * dim_p2.y;
    if (bc == 'n')
    {
        cuGradientMinusLwtQ_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, q, dim_p2, levels, dx_inv, max_size);
    }
    else
    {
        cuGradientMinusLwtQ_d << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, q, dim_p2, levels, dx_inv, max_size);
    }
}

void CuGradient::GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc)
{
    int2 levels = { std::log2(dim.x), std::log2(dim.y) };
    GradientMinusLwtP(new_p, xv, yv, p, dim, levels, dx, bc);
}

void CuGradient::GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, int2 levels, DTYPE2 dx, char bc)
{
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim.x * dim.y;
    if (bc == 'n')
    {
        cuGradientMinusLwtP_n << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, p, dim, levels, dx_inv, max_size);
    }
    else if (bc == 'd')
    {
        cuGradientMinusLwtP_d << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, p, dim, levels, dx_inv, max_size);
    }
    else if (bc == 'z')
    {
        cuGradientMinusLwtP_z << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, p, dim, levels, dx_inv, max_size);
    }
}

void CuGradient::GradientMinusLwtQ(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, DTYPE2 dx, char bc)
{
    int2 levels = { std::log2(dim.x), std::log2(dim.y) };
    GradientMinusLwtQ(new_q, xv, yv, q, dim, levels, dx, bc);
}

void CuGradient::GradientMinusLwtQ(DTYPE* new_q, DTYPE* xv, DTYPE* yv, DTYPE* q, int2 dim, int2 levels, DTYPE2 dx, char bc)
{
    int2 dim_p2 = { dim.x - (dim.x & 1), dim.y - (dim.y & 1) };
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim_p2.x * dim_p2.y;
    if (bc == 'd')
    {
        cuGradientMinusLwtQ_d << <BLOCKS(max_size), THREADS(max_size) >> > (new_q, xv, yv, q, dim_p2, levels, dx_inv, max_size);
    }
    else
    {
        cuGradientMinusLwtQ_n << <BLOCKS(max_size), THREADS(max_size) >> > (new_q, xv, yv, q, dim_p2, levels, dx_inv, max_size);
    }
}

void CuGradient::GradientMinusLwtP(DTYPE* new_p, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc)
{
    DTYPE3 dx_inv = { 1.f / dx.x, 1.f / dx.y, 1.f / dx.z };
    //int max_size = dim.x * dim.y * dim.z;
    //if (bc == 'n')
    //{
    //    cuGradientMinusLwtP_n << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv, max_size);
    //}
    //else
    //{
    //    cuGradientMinusLwtP_d << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv, max_size);
    //}

    //int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 16;

    int blocks_y = std::ceil(double(dim.y) / threads_y);
    int blocks_x = std::ceil(double(dim.x) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);
    if (bc == 'n')
    {
        cuGradientMinusLwtP_n_block << <grid, block >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv);
    }
    else
    {
        cuGradientMinusLwtP_d_block << <grid, block >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv);
    }
}

void CuGradient::GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* xv_lwt, DTYPE* yv_lwt, DTYPE* zv_lwt,
    DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc)
{
    DTYPE3 dx_inv = { 1.f / dx.x, 1.f / dx.y, 1.f / dx.z };
    int max_size = dim.x * dim.y * dim.z;
    if (bc == 'n')
    {
        cuGradientMinusLwtP_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, xv_lwt, yv_lwt, zv_lwt, p, dim, levels, dx_inv, max_size);
    }
    else
    {
        cuGradientMinusLwtP_d << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, xv_lwt, yv_lwt, zv_lwt, p, dim, levels, dx_inv, max_size);
    }
}

void CuGradient::GradientMinusLwtP(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, int3 levels, DTYPE3 dx, char bc)
{
    DTYPE3 dx_inv = { 1.f / dx.x, 1.f / dx.y, 1.f / dx.z };
    //int max_size = dim.x * dim.y * dim.z;
    //if (bc == 'n')
    //{
    //    cuGradientMinusLwtP_n << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv, max_size);
    //}
    //else
    //{
    //    cuGradientMinusLwtP_d << <BLOCKS(max_size), THREADS(max_size) >> > (new_p, xv, yv, zv, p, dim, levels, dx_inv, max_size);
    //}

    //int3 dim = { 1 << levels.x, 1 << levels.y, 1 << levels.z };

    const int threads_y = 32;
    const int threads_x = 16;

    int blocks_y = std::ceil(double(dim.y) / threads_y);
    int blocks_x = std::ceil(double(dim.x) / threads_x);

    dim3 grid(blocks_y, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);
    if (bc == 'n')
    {
        cuGradientMinusLwtP_n_block << <grid, block >> > (xv, yv, zv, p, dim, levels, dx_inv);
    }
    else
    {
        cuGradientMinusLwtP_d_block << <grid, block >> > (xv, yv, zv, p, dim, levels, dx_inv);
    }
}

/**********************************************************************************************************************************************************/
__global__ void cuGradient2D_p_n(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;
        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0)
        {
            xv_val = (p[idx] - p[idx - dim.y]) * dx_inv.x;
        }
        DTYPE yv_val = 0.f;
        if (idx_y > 0)
        {
            yv_val = (p[idx] - p[idx - 1]) * dx_inv.y;
        }
        xv[idx_xv] = xv_val;
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
    }
}

__global__ void cuGradient2D_frac_p_n(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) * dx_inv.x;
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) * dx_inv.y;
        }
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
    }
}

__global__ void cuGradient2D_p_z(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = p[idx] * dx_inv.x;
        if (idx_x > 0)
        {
            xv_val -= p[idx - dim.y] * dx_inv.x;
        }
        DTYPE yv_val = p[idx] * dx_inv.y;
        if (idx_y > 0)
        {
            yv_val -= p[idx - 1] * dx_inv.y;
        }
        xv[idx_xv] = xv_val;
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = -p[idx] * dx_inv.x;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = -p[idx] * dx_inv.y;
        }
    }
}

__global__ void cuGradient2D_frac_p_z(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) * dx_inv.x;
        }
        else if (idx_x == 0)
        {
            xv_val = p[idx] * dx_inv.x;
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) * dx_inv.y;
        }
        else if (idx_y == 0)
        {
            yv_val = p[idx] * dx_inv.y;
        }
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = -p[idx] * dx_inv.x;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = -p[idx] * dx_inv.y;
        }
    }
}

__global__ void cuGradient2D_p_d(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = p[idx] * dx_inv.x;
        if (idx_x > 0)
        {
            xv_val -= p[idx - dim.y] * dx_inv.x;
        }
        else
        {
            xv_val *= 2.f;
        }
        DTYPE yv_val = p[idx] * dx_inv.y;
        if (idx_y > 0)
        {
            yv_val -= p[idx - 1] * dx_inv.y;
        }
        else
        {
            yv_val *= 2.f;
        }
        xv[idx_xv] = xv_val;
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = -p[idx] * dx_inv.x * 2.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = -p[idx] * dx_inv.y * 2.f;
        }
    }
}

__global__ void cuGradient2D_frac_p_d(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) * dx_inv.x;
        }
        else if (idx_x == 0)
        {
            xv_val = p[idx] * dx_inv.x * 2.f;
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) * dx_inv.y;
        }
        else if (idx_y == 0)
        {
            yv_val = p[idx] * dx_inv.y * 2.f;
        }
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = -p[idx] * dx_inv.x * 2.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = -p[idx] * dx_inv.y * 2.f;
        }
    }
}

__global__ void cuGradient2D_p_n(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0)
        {
            xv_val = (p[idx + dim.y] - p[idx]) / (dx[idx_x] + dx[idx_x - 1]) * 2.f;
        }
        DTYPE yv_val = 0.f;
        if (idx_y > 0)
        {
            yv_val = (p[idx + 1] - p[idx]) / (dy[idx_y] + dy[idx_y - 1]) * 2.f;
        }
        xv[idx_xv] = xv_val;
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
    }
}

__global__ void cuGradient2D_frac_p_n(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) / (dx[idx_x] + dx[idx_x - 1]) * 2.f;
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) / (dy[idx_y] + dy[idx_y - 1]) * 2.f;
        }
        yv[idx_yv] = yv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
    }
}

__global__ void cuGradient2D_p_z(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE dx_nei = dx_e.x;
        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            dx_nei = dx[idx_x - 1];
            p_nei = p[idx - dim.y];
        }
        xv[idx_xv] = (p[idx] - p_nei) / (dx[idx_x] + dx_nei) * 2.f;

        DTYPE dy_nei = dx_e.y;
        p_nei = 0.f;
        if (idx_y > 0)
        {
            dy_nei = dy[idx_y - 1];
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv[idx_yv] = (p[idx] - p_nei) / (dy[idx_y] + dy_nei) * 2.f;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = -p[idx] / (dx[idx_x] + dx_e.x) * 2.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = -p[idx] / (dy[idx_y] + dx_e.y) * 2.f;
        }
    }
}

__global__ void cuGradient2D_frac_p_z(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        DTYPE dx_nei = dx_e.x;
        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            dx_nei = dx[idx_x - 1];
            p_nei = p[idx - dim.y];
        }
        xv[idx_xv] = ax[idx_xv] > eps<DTYPE> ? (p[idx] - p_nei) / (dx[idx_x] + dx_nei) * 2.f : 0.f;

        DTYPE dy_nei = dx_e.y;
        p_nei = 0.f;
        if (idx_y > 0)
        {
            dy_nei = dy[idx_y - 1];
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv[idx_yv] = ay[idx_yv] > eps<DTYPE> ? (p[idx] - p_nei) / (dy[idx_y] + dy_nei) * 2.f : 0.f;

        if (idx_x == dim.x - 1)
            xv[idx_xv + dim.y] = ax[idx_xv] > eps<DTYPE> ? -p[idx] / (dx[idx_x] + dx_e.x) * 2.f : 0.f;
        if (idx_y == dim.y - 1)
            yv[idx_yv + 1] = ay[idx_yv] > eps<DTYPE> ? -p[idx] / (dy[idx_y] + dx_e.y) * 2.f : 0.f;
    }
}

void CuGradient::Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE2 dx, char bc)
{
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    case 'z':
        cuGradient2D_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, dx_inv, max_size);
        break;
    default:
        cuGradient2D_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, dx_inv, max_size);
        break;
    }
}

void CuGradient::Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, char bc)
{
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    default:
        cuGradient2D_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, dx, dy, max_size);
        break;
    }
}

void CuGradient::Gradient2D_P(DTYPE* xv, DTYPE* yv, DTYPE* p, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc)
{
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    case 'z':
        cuGradient2D_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, dx, dy, dx_e, max_size);
        break;
    default:
        cuGradient2D_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, dim, dx, dy, max_size);
        break;
    }
}

void CuGradient::Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx, char bc)
{
    DTYPE2 dx_inv = { 1.f / dx.x, 1.f / dx.y };
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    case 'z':
        cuGradient2D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, ax, ay, dim, dx_inv, max_size);
        break;
    default:
        cuGradient2D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, ax, ay, dim, dx_inv, max_size);
        break;
    }
}

void CuGradient::Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc)
{
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    default:
        cuGradient2D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, ax, ay, dim, dx, dy, max_size);
        break;
    }
}

void CuGradient::Gradient2D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* p, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc)
{
    int max_size = dim.x * dim.y;

    switch (bc)
    {
    case 'z':
        cuGradient2D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, ax, ay, dim, dx, dy, dx_e, max_size);
        break;
    default:
        cuGradient2D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, p, ax, ay, dim, dx, dy, max_size);
        break;
    }
}

/*******************************************************************---3 D---***********************************************************************************/
__global__ void cuGradient3D_frac_p_n(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) / (dx[idx_x] + dx[idx_x - 1]) * 2.f;
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) / (dy[idx_y] + dy[idx_y - 1]) * 2.f;
        }
        yv[idx_yv] = yv_val;

        DTYPE zv_val = 0.f;
        if (idx_z > 0 && az[idx_zv] > eps<DTYPE>)
        {
            zv_val = (p[idx] - p[idx - slice_xy]) / (dz[idx_z] + dz[idx_z - 1]) * 2.f;
        }
        zv[idx_zv] = zv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            zv[idx_zv + slice_xy] = 0.f;
        }
    }
}

__global__ void cuGradient3D_frac_p_z(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE dx_nei = dx_e.x;
        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            dx_nei = dx[idx_x - 1];
            p_nei = p[idx - dim.y];
        }
        xv[idx_xv] = ax[idx_xv] > eps<DTYPE> ? (p[idx] - p_nei) / (dx[idx_x] + dx_nei) * 2.f : 0.f;

        DTYPE dy_nei = dx_e.y;
        p_nei = 0.f;
        if (idx_y > 0)
        {
            dy_nei = dy[idx_y - 1];
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv[idx_yv] = ay[idx_yv] > eps<DTYPE> ? (p[idx] - p_nei) / (dy[idx_y] + dy_nei) * 2.f : 0.f;

        DTYPE dz_nei = dx_e.z;
        p_nei = 0.f;
        if (idx_z > 0)
        {
            dz_nei = dz[idx_z - 1];
            p_nei = p[idx - slice_xy];
        }
        zv[idx_zv] = az[idx_zv] > eps<DTYPE> ? (p[idx] - p_nei) / (dz[idx_z] + dz_nei) * 2.f : 0.f;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = ax[idx_xv] > eps<DTYPE> ? -p[idx] / (dx[idx_x] + dx_e.x) * 2.f : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = ay[idx_yv] > eps<DTYPE> ? -p[idx] / (dy[idx_y] + dx_e.y) * 2.f : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            zv[idx_zv + slice_xy] = az[idx_zv] > eps<DTYPE> ? -p[idx] / (dz[idx_z] + dx_e.z) * 2.f : 0.f;
        }
    }
}

__global__ void cuGradient3D_frac_p_n(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE xv_val = 0.f;
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = (p[idx] - p[idx - dim.y]) / (dx.x);
        }
        xv[idx_xv] = xv_val;

        DTYPE yv_val = 0.f;
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = (p[idx] - p[idx - 1]) / dx.y;
        }
        yv[idx_yv] = yv_val;

        DTYPE zv_val = 0.f;
        if (idx_z > 0 && az[idx_zv] > eps<DTYPE>)
        {
            zv_val = (p[idx] - p[idx - slice_xy]) / dx.z;
        }
        zv[idx_zv] = zv_val;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            zv[idx_zv + slice_xy] = 0.f;
        }
    }
}

__global__ void cuGradient3D_frac_p_z(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            p_nei = p[idx - dim.y];
        }
        xv[idx_xv] = ax[idx_xv] > eps<DTYPE> ? (p[idx] - p_nei) / dx.x : 0.f;

        p_nei = 0.f;
        if (idx_y > 0)
        {
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv[idx_yv] = ay[idx_yv] > eps<DTYPE> ? (p[idx] - p_nei) / dx.y : 0.f;

        p_nei = 0.f;
        if (idx_z > 0)
        {
            p_nei = p[idx - slice_xy];
        }
        zv[idx_zv] = az[idx_zv] > eps<DTYPE> ? (p[idx] - p_nei) / dx.z : 0.f;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = ax[idx_xv + dim.y] > eps<DTYPE> ? -p[idx] / dx.x : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = ay[idx_yv + 1] > eps<DTYPE> ? -p[idx] / dx.y : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            zv[idx_zv + slice_xy] = az[idx_zv + slice_xy] > eps<DTYPE> ? -p[idx] / dx.z : 0.f;
        }
    }
}

__global__ void cuGradient3D_frac_p_z(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE3 dx, DTYPE3 dx_e, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE p_nei = 0.f;
        DTYPE dx_nei = (dx_e.x + dx.x) * 0.5f;
        if (idx_x > 0)
        {
            p_nei = p[idx - dim.y];
            dx_nei = dx.x;
        }
        xv[idx_xv] = ax[idx_xv] > eps<DTYPE> ? (p[idx] - p_nei) / dx_nei : 0.f;

        p_nei = 0.f;
        dx_nei = (dx_e.y + dx.y) * 0.5f;
        if (idx_y > 0)
        {
            p_nei = p[idx - 1];
            dx_nei = dx.y;
        }
        //xv[idx_xv] = xv_val;
        yv[idx_yv] = ay[idx_yv] > eps<DTYPE> ? (p[idx] - p_nei) / dx_nei : 0.f;

        p_nei = 0.f;
        dx_nei = (dx_e.z + dx.z) * 0.5f;
        if (idx_z > 0)
        {
            p_nei = p[idx - slice_xy];
            dx_nei = dx.z;
        }
        zv[idx_zv] = az[idx_zv] > eps<DTYPE> ? (p[idx] - p_nei) / dx_nei : 0.f;

        if (idx_x == dim.x - 1)
        {
            xv[idx_xv + dim.y] = ax[idx_xv + dim.y] > eps<DTYPE> ? -p[idx] / ((dx.x + dx_e.x) * 0.5f) : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            yv[idx_yv + 1] = ay[idx_yv + 1] > eps<DTYPE> ? -p[idx] / ((dx.y + dx_e.y) * 0.5f) : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            zv[idx_zv + slice_xy] = az[idx_zv + slice_xy] > eps<DTYPE> ? -p[idx] / ((dx.z + dx_e.z) * 0.5f) : 0.f;
        }
    }
}

__global__ void cuGradient3D_minus_frac_p_n(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv, 
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE xv_val = xv[idx_xv] * ax[idx_xv];
        if (idx_x > 0 && ax[idx_xv] > eps<DTYPE>)
        {
            xv_val = xv[idx_xv] - (p[idx] - p[idx - dim.y]) / (dx.x);
        }
        xv_out[idx_xv] = xv_val;

        DTYPE yv_val = yv[idx_yv] * ay[idx_yv];
        if (idx_y > 0 && ay[idx_yv] > eps<DTYPE>)
        {
            yv_val = yv[idx_yv] - (p[idx] - p[idx - 1]) / dx.y;
        }
        yv_out[idx_yv] = yv_val;

        DTYPE zv_val = zv[idx_zv] * az[idx_zv];
        if (idx_z > 0 && az[idx_zv] > eps<DTYPE>)
        {
            zv_val = zv[idx_zv] - (p[idx] - p[idx - slice_xy]) / dx.z;
        }
        zv_out[idx_zv] = zv_val;

        if (idx_x == dim.x - 1)
        {
            xv_out[idx_xv + dim.y] = xv[idx_xv + dim.y] * ax[idx_xv + dim.y];
        }
        if (idx_y == dim.y - 1)
        {
            yv_out[idx_yv + 1] = yv[idx_yv + 1] * ay[idx_yv + 1];
        }
        if (idx_z == dim.z - 1)
        {
            zv_out[idx_zv + slice_xy] = zv[idx_zv + slice_xy] * az[idx_zv + slice_xy];
        }
    }
}

__global__ void cuGradient3D_minus_frac_p_n(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ax_select, DTYPE* ay_select, DTYPE* az_select, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE xv_val = xv[idx_xv] * ax[idx_xv];
        if (idx_x > 0 && ax_select[idx_xv] > eps<DTYPE>)
        {
            xv_val = xv[idx_xv] - (p[idx] - p[idx - dim.y]) / (dx.x);
        }
        xv_out[idx_xv] = xv_val;

        DTYPE yv_val = yv[idx_yv] * ay[idx_yv];
        if (idx_y > 0 && ay_select[idx_yv] > eps<DTYPE>)
        {
            yv_val = yv[idx_yv] - (p[idx] - p[idx - 1]) / dx.y;
        }
        yv_out[idx_yv] = yv_val;

        DTYPE zv_val = zv[idx_zv] * az[idx_zv];
        if (idx_z > 0 && az_select[idx_zv] > eps<DTYPE>)
        {
            zv_val = zv[idx_zv] - (p[idx] - p[idx - slice_xy]) / dx.z;
        }
        zv_out[idx_zv] = zv_val;

        if (idx_x == dim.x - 1)
        {
            xv_out[idx_xv + dim.y] = xv[idx_xv + dim.y] * ax[idx_xv + dim.y];
        }
        if (idx_y == dim.y - 1)
        {
            yv_out[idx_yv + 1] = yv[idx_yv + 1] * ay[idx_yv + 1];
        }
        if (idx_z == dim.z - 1)
        {
            zv_out[idx_zv + slice_xy] = zv[idx_zv + slice_xy] * az[idx_zv + slice_xy];
        }
    }
}

__global__ void cuGradient3D_minus_frac_p_z(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv, 
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            p_nei = p[idx - dim.y];
        }
        xv_out[idx_xv] = ax[idx_xv] > eps<DTYPE> ? xv[idx_xv] - (p[idx] - p_nei) / dx.x : 0.f;

        p_nei = 0.f;
        if (idx_y > 0)
        {
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv_out[idx_yv] = ay[idx_yv] > eps<DTYPE> ? yv[idx_yv] - (p[idx] - p_nei) / dx.y : 0.f;

        p_nei = 0.f;
        if (idx_z > 0)
        {
            p_nei = p[idx - slice_xy];
        }
        zv_out[idx_zv] = az[idx_zv] > eps<DTYPE> ? zv[idx_zv] - (p[idx] - p_nei) / dx.z : 0.f;

        if (idx_x == dim.x - 1)
        {
            int idx_xv_n = idx_xv + dim.y;
            xv_out[idx_xv_n] = ax[idx_xv_n] > eps<DTYPE> ? xv[idx_xv_n] - -p[idx] / dx.x : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            int idx_yv_n = idx_yv + 1;
            yv_out[idx_yv_n] = ay[idx_yv_n] > eps<DTYPE> ? yv[idx_yv_n] - -p[idx] / dx.y : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            int idx_zv_n = idx_zv + slice_xy;
            zv_out[idx_zv_n] = az[idx_zv_n] > eps<DTYPE> ? zv[idx_zv_n] - -p[idx] / dx.z : 0.f;
        }
    }
}

__global__ void cuGradient3D_minus_frac_p_z(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, DTYPE3 dx_e, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE p_nei = 0.f;
        DTYPE dx_nei = (dx_e.x + dx.x) * 0.5f;
        if (idx_x > 0)
        {
            p_nei = p[idx - dim.y];
            dx_nei = dx.x;
        }
        xv_out[idx_xv] = ax[idx_xv] > eps<DTYPE> ? xv[idx_xv] - (p[idx] - p_nei) / dx_nei : 0.f;

        p_nei = 0.f;
        dx_nei = (dx_e.y + dx.y) * 0.5f;
        if (idx_y > 0)
        {
            p_nei = p[idx - 1];
            dx_nei = dx.y;
        }
        yv_out[idx_yv] = ay[idx_yv] > eps<DTYPE> ? yv[idx_yv] - (p[idx] - p_nei) / dx_nei : 0.f;

        p_nei = 0.f;
        dx_nei = (dx_e.z + dx.z) * 0.5f;
        if (idx_z > 0)
        {
            p_nei = p[idx - slice_xy];
            dx_nei = dx.z;
        }
        zv_out[idx_zv] = az[idx_zv] > eps<DTYPE> ? zv[idx_zv] - (p[idx] - p_nei) / dx_nei : 0.f;

        if (idx_x == dim.x - 1)
        {
            int idx_xv_n = idx_xv + dim.y;
            dx_nei = (dx_e.x + dx.x) * 0.5f;
            xv_out[idx_xv_n] = ax[idx_xv_n] > eps<DTYPE> ? xv[idx_xv_n] - -p[idx] / dx_nei : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            int idx_yv_n = idx_yv + 1;
            dx_nei = (dx_e.y + dx.y) * 0.5f;
            yv_out[idx_yv_n] = ay[idx_yv_n] > eps<DTYPE> ? yv[idx_yv_n] - -p[idx] / dx_nei : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            int idx_zv_n = idx_zv + slice_xy;
            dx_nei = (dx_e.z + dx.z) * 0.5f;
            zv_out[idx_zv_n] = az[idx_zv_n] > eps<DTYPE> ? zv[idx_zv_n] - -p[idx] / dx_nei : 0.f;
        }
    }
}

__global__ void cuGradient3D_LastPhi_X(DTYPE* qx, DTYPE* phi, int3 dim_qx, DTYPE dx_inv)
{
    int idx_z = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_z < dim_qx.z && idx_y < dim_qx.y)
    {
        int idx = idx_y + (dim_qx.x - 1) * dim_qx.y + idx_z * dim_qx.x * dim_qx.y;
        int idx_phi = idx_y + idx_z * dim_qx.y;
        qx[idx] = (0.f - phi[idx_phi]) * dx_inv;
    }
}

__global__ void cuGradientMinus3D_zero_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_phi.x * dim_phi.y;
    int dim_qx_y = dim_phi.y + 2;
    int dim_qy_y = dim_phi.y + 1;
    int dim_qz_y = dim_phi.y + 2;
    int slice_qx = dim_qx_y * (dim_phi.x + 1);
    int slice_qy = dim_qy_y * (dim_phi.x + 2);
    int slice_qz = dim_qz_y * (dim_phi.x + 2);

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        int i_1 = i + 1;
        int j_1 = j + 1;
        int k_1 = k + 1;

        int idx_qb = idx;
        int idx_qx = i_1 + j_1 * dim_qx_y + k_1 * slice_qx;
        int idx_qy = i_1 + j_1 * dim_qy_y + k_1 * slice_qy;
        int idx_qz = i_1 + j_1 * dim_qz_y + k_1 * slice_qz;

        DTYPE phi_i_dx = phi[idx] * dx_inv.x;
        DTYPE phi_i_dy = phi[idx] * dx_inv.y;
        DTYPE phi_i_dz = phi[idx] * dx_inv.z;

        if (i == 0)
            qy[idx_qy - 1] -= phi_i_dx;
        if (j == 0)
            qx[idx_qx - dim_qx_y] -= phi_i_dy;
        if (k == 0)
            qz[idx_qz - slice_qz] -= phi_i_dz;

        DTYPE qy_n = -phi_i_dy;
        DTYPE qx_n = -phi_i_dx;
        DTYPE qz_n = -phi_i_dz;

        if (i < dim_phi.y - 1)
            qy_n += phi[idx + 1] * dx_inv.y;
        if (j < dim_phi.x - 1)
            qx_n += phi[idx + dim_phi.y] * dx_inv.x;
        if (k < dim_phi.z - 1)
            qz_n += phi[idx + slice_qb] * dx_inv.z;

        qy[idx_qy] -= qy_n;
        qx[idx_qx] -= qx_n;
        qz[idx_qz] -= qz_n;
    }
}

__global__ void cuGradientMinus3D_d_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int slice_qb = dim_phi.x * dim_phi.y;
        int dim_qx_y = dim_phi.y;
        int dim_qy_y = dim_phi.y - 1;
        int dim_qz_y = dim_phi.y;
        int slice_qx = dim_qx_y * (dim_phi.x - 1);
        int slice_qy = dim_qy_y * (dim_phi.x);
        int slice_qz = dim_qz_y * (dim_phi.x);

        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        //int idx_qb = idx;
        int idx_qx = i + j * dim_qx_y + k * slice_qx;
        int idx_qy = i + j * dim_qy_y + k * slice_qy;
        int idx_qz = i + j * dim_qz_y + k * slice_qz;

        DTYPE phi_i_dx = phi[idx] * dx_inv.x;
        DTYPE phi_i_dy = phi[idx] * dx_inv.y;
        DTYPE phi_i_dz = phi[idx] * dx_inv.z;

        DTYPE qy_n = -phi_i_dy;
        DTYPE qx_n = -phi_i_dx;
        DTYPE qz_n = -phi_i_dz;

        if (i < dim_phi.y - 1)
            qy[idx_qy] -= phi[idx + 1] * dx_inv.y + qy_n;
        if (j < dim_phi.x - 1)
            qx[idx_qx] -= phi[idx + dim_phi.y] * dx_inv.x + qx_n;
        if (k < dim_phi.z - 1)
            qz[idx_qz] -= phi[idx + slice_qb] * dx_inv.z + qz_n;
    }
}

void CuGradient::Gradient3D_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, int3 dim, DTYPE* dx, DTYPE* dy,
    DTYPE* dz, char bc)
{

}

void CuGradient::Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE3 dx, char bc)
{
    int max_size = dim.x * dim.y * dim.z;

    switch (bc)
    {
    case 'd':
        cuGradient3D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, make_DTYPE3(0.f), max_size);
        break;
    case 'z':
        cuGradient3D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    default:
        cuGradient3D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    }
    cudaCheckError(cudaGetLastError());
}

void CuGradient::Gradient3D_minus_Frac_P(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, char bc)
{
    int max_size = dim.x * dim.y * dim.z;

    switch (bc)
    {
    case 'd':
        cuGradient3D_minus_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, 
            ax, ay, az, dim, dx, make_DTYPE3(0.f), max_size);
        break;
    case 'z':
        cuGradient3D_minus_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    default:
        cuGradient3D_minus_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    }
    cudaCheckError(cudaGetLastError());
}

void CuGradient::Gradient3D_minus_Frac_P_select(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ax_select, DTYPE* ay_select, DTYPE* az_select, int3 dim, DTYPE3 dx, char bc)
{
    int max_size = dim.x * dim.y * dim.z;

    switch (bc)
    {
    case 'z':
        //cuGradient3D_minus_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    default:
        cuGradient3D_minus_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p,
            ax, ay, az, ax_select, ay_select, az_select, dim, dx, max_size);
        break;
    }
    cudaCheckError(cudaGetLastError());
}

void CuGradient::Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc)
{
    int max_size = dim.x * dim.y * dim.z;

    switch (bc)
    {
    default:
        cuGradient3D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, dy, dz, max_size);
        break;
    }
}

void CuGradient::Gradient3D_Frac_P(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
    int max_size = dim.x * dim.y * dim.z;

    switch (bc)
    {
    case 'z':
        cuGradient3D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, dy, dz, dx_e, max_size);
    case 'd':
        cuGradient3D_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, dy, dz, make_DTYPE3(0.f), max_size);
        break;
    default:
        cuGradient3D_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv, yv, zv, p, ax, ay, az, dim, dx, dy, dz, max_size);
        break;
    }
    cudaCheckError(cudaGetLastError());
}

void CuGradient::Gradient3D_LastPhi_X(DTYPE* qx, DTYPE* phi, int3 dim_qx, DTYPE dx_inv)
{
    dim3 blocks(THREAD_DIM_2D_16, THREAD_DIM_2D_16, 1);
    dim3 grids;
    grids.x = std::ceil(DTYPE(dim_qx.y) / THREAD_DIM_2D_16);
    grids.y = std::ceil(DTYPE(dim_qx.z) / THREAD_DIM_2D_16);

    cuGradient3D_LastPhi_X << <grids, blocks >> > (qx, phi, dim_qx, dx_inv);
}

void CuGradient::GradientMinus3D_zero_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx)
{
    int max_size = dim_phi.x * dim_phi.y * dim_phi.z;

    DTYPE3 dx_inv = { 1.f / dx.x, 1.f / dx.y, 1.f / dx.z };
    cuGradientMinus3D_zero_Q << <BLOCKS(max_size), THREADS(max_size) >> > (qx, qy, qz, phi, dim_phi, dx_inv, max_size);
}

void CuGradient::GradientMinus3D_d_Q(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* phi, int3 dim_phi, DTYPE3 dx)
{
    int max_size = dim_phi.x * dim_phi.y * dim_phi.z;

    DTYPE3 dx_inv = { 1.f / dx.x, 1.f / dx.y, 1.f / dx.z };
    cuGradientMinus3D_d_Q << <BLOCKS(max_size), THREADS(max_size) >> > (qx, qy, qz, phi, dim_phi, dx_inv, max_size);
}

__global__ void cuGradient3D_minus_frac_p_n_bound(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* is_bound, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE xv_val = xv[idx_xv] * ax[idx_xv];
        DTYPE yv_val = yv[idx_yv] * ay[idx_yv];
        DTYPE zv_val = zv[idx_zv] * az[idx_zv];
        DTYPE cf_d = 2.f;
        if (is_bound[idx] > 0.5f)
        {
            if (idx_x > 0)
            {
                //xv_val = xv[idx_xv] - (p[idx] - p[idx - dim.y]) / (dx.x);
                if (is_bound[idx - dim.y] > 0.5f)
                    xv_val = xv[idx_xv] - (p[idx] - p[idx - dim.y]) / (dx.x);
                else
                    xv_val = xv[idx_xv] - cf_d * (p[idx]) / (dx.x);
            }

            if (idx_y > 0)
            {
                //yv_val = yv[idx_yv] - (p[idx] - p[idx - 1]) / dx.y;
                if (is_bound[idx - 1] > 0.5f)
                    yv_val = yv[idx_yv] - (p[idx] - p[idx - 1]) / dx.y;
                else
                    yv_val = yv[idx_yv] - cf_d * (p[idx]) / dx.y;
            }

            if (idx_z > 0)
            {
                //zv_val = zv[idx_zv] - (p[idx] - p[idx - slice_xy]) / dx.z;
                if (is_bound[idx - slice_xy] > 0.5f)
                    zv_val = zv[idx_zv] - (p[idx] - p[idx - slice_xy]) / dx.z;
                else
                    zv_val = zv[idx_zv] - cf_d * (p[idx]) / dx.z;
            }
        }
        else
        {
            if (idx_x > 0)
            {
                //xv_val = xv[idx_xv] - (p[idx] - p[idx - dim.y]) / (dx.x);
                if (is_bound[idx - dim.y] > 0.5f)
                    xv_val = xv[idx_xv] - cf_d * (-p[idx - dim.y]) / (dx.x);
            }

            if (idx_y > 0)
            {
                //yv_val = yv[idx_yv] - (p[idx] - p[idx - 1]) / dx.y;
                if (is_bound[idx - 1] > 0.5f)
                    yv_val = yv[idx_yv] - cf_d * (-p[idx - 1]) / dx.y;
            }

            if (idx_z > 0)
            {
                //zv_val = zv[idx_zv] - (p[idx] - p[idx - slice_xy]) / dx.z;
                if (is_bound[idx - slice_xy] > 0.5f)
                    zv_val = zv[idx_zv] - cf_d * (-p[idx - slice_xy]) / dx.z;
            }
        }
        yv_out[idx_yv] = yv_val;
        xv_out[idx_xv] = xv_val;
        zv_out[idx_zv] = zv_val;
        if (idx_x == dim.x - 1)
        {
            xv_out[idx_xv + dim.y] = xv[idx_xv + dim.y] * ax[idx_xv + dim.y];
        }
        if (idx_y == dim.y - 1)
        {
            yv_out[idx_yv + 1] = yv[idx_yv + 1] * ay[idx_yv + 1];
        }
        if (idx_z == dim.z - 1)
        {
            zv_out[idx_zv + slice_xy] = zv[idx_zv + slice_xy] * az[idx_zv + slice_xy];
        }
    }
}

__global__ void cuGradient3D_minus_frac_p_z_bound(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice_xy = dim.x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
        int idx_zv = idx;

        DTYPE p_nei = 0.f;
        if (idx_x > 0)
        {
            p_nei = p[idx - dim.y];
        }
        xv_out[idx_xv] = ax[idx_xv] > eps<DTYPE> ? xv[idx_xv] - (p[idx] - p_nei) / dx.x : 0.f;

        p_nei = 0.f;
        if (idx_y > 0)
        {
            p_nei = p[idx - 1];
        }
        //xv[idx_xv] = xv_val;
        yv_out[idx_yv] = ay[idx_yv] > eps<DTYPE> ? yv[idx_yv] - (p[idx] - p_nei) / dx.y : 0.f;

        p_nei = 0.f;
        if (idx_z > 0)
        {
            p_nei = p[idx - slice_xy];
        }
        zv_out[idx_zv] = az[idx_zv] > eps<DTYPE> ? zv[idx_zv] - (p[idx] - p_nei) / dx.z : 0.f;

        if (idx_x == dim.x - 1)
        {
            int idx_xv_n = idx_xv + dim.y;
            xv_out[idx_xv_n] = ax[idx_xv_n] > eps<DTYPE> ? xv[idx_xv_n] - -p[idx] / dx.x : 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            int idx_yv_n = idx_yv + 1;
            yv_out[idx_yv_n] = ay[idx_yv_n] > eps<DTYPE> ? yv[idx_yv_n] - -p[idx] / dx.y : 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            int idx_zv_n = idx_zv + slice_xy;
            zv_out[idx_zv_n] = az[idx_zv_n] > eps<DTYPE> ? zv[idx_zv_n] - -p[idx] / dx.z : 0.f;
        }
    }
}

void CuGradient::Gradient3D_minus_Frac_P_bound(DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, DTYPE* xv, DTYPE* yv, DTYPE* zv,
    DTYPE* p, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* is_bound, int3 dim, DTYPE3 dx, char bc)
{
    int max_size = getsize(dim);
    if (is_bound == nullptr)
    {
        is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
        CuWeightedJacobi3D::GetBound(is_bound, ax, ay, az, dim);
        //CudaPrintfMat(is_bound, dim);
    }

    switch (bc)
    {
    case 'z':
        cuGradient3D_minus_frac_p_z << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    default:
        cuGradient3D_minus_frac_p_n_bound << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, 
            is_bound, dim, dx, max_size);
        //cuGradient3D_minus_frac_p_n << <BLOCKS(max_size), THREADS(max_size) >> > (xv_out, yv_out, zv_out, xv, yv, zv, p, ax, ay, az, dim, dx, max_size);
        break;
    }
    cudaCheckError(cudaGetLastError());
}



void CuGradient::GetVelocityFromQ(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, DTYPE dx)
{

}