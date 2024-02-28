#include "cuParallelSweep3D.cuh"
#include "cuMemoryManager.cuh"
#include "cuMultigrid3D.cuh"
#include "cudaMath.cuh"
#include "cuGradient.cuh"
#include "cuConvergence.cuh"
#include "Interpolation.cuh"

__global__ void cuSweepingQ_x(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_z = idx / dim_dst.y;
        int idx_y = idx - idx_z * dim_dst.y;
        int idx_dst_base = idx_y + idx_z * dim_dst.x * dim_dst.y;
        int idx_src_base = idx_y + idx_z * dim_src.x * dim_src.y;

        dst[idx_dst_base] = 0.f;
        for (int i = 1; i < dim_dst.x; i++)
        {
            dst[idx_dst_base + i * dim_dst.y] = dst[idx_dst_base + (i - 1) * dim_dst.y] + src[idx_src_base + (i - 1) * dim_src.y] * dx;
        }
    }
}

__global__ void SweepingQx(DTYPE* qx, DTYPE* yv, DTYPE3 dx, int3 dim_qx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice = dim_qx.x * dim_qx.y;
    int3 dim_yv = { dim_qx.x, dim_qx.y, dim_qx.z - 1 };
    int slice_yv = dim_yv.x * dim_yv.y;

    if (idx < max_size)
    {
        int idx_x = idx / dim_qx.y;
        int idx_y = idx - idx_x * dim_qx.y;
        int idx_yv_base = idx_x * dim_yv.y + idx_y;

        for (int i = 1; i < dim_qx.z; i++)
        {
            qx[idx + i * slice] = qx[idx + (i - 1) * slice] + yv[idx_yv_base + (i - 1) * slice_yv] * dx.z;
        }
    }
}

__global__ void SweepingQy(DTYPE* qy, DTYPE* xv, DTYPE3 dx, int3 dim_qy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice = dim_qy.x * dim_qy.y;
    int3 dim_xv = { dim_qy.x, dim_qy.y, dim_qy.z - 1 };
    int slice_xv = dim_xv.x * dim_xv.y;

    if (idx < max_size)
    {
        int idx_x = idx / dim_qy.y;;
        int idx_y = idx - idx_x * dim_qy.y;
        int idx_xv_base = idx_x * dim_xv.y + idx_y;

        for (int i = 1; i < dim_qy.z; i++)
        {
            qy[idx + i * slice] = qy[idx + (i - 1) * slice] - xv[idx_xv_base + (i - 1) * slice_xv] * dx.z;
        }
    }
}

__global__ void SweepingQy_x(DTYPE* qy, DTYPE* zv, DTYPE3 dx, int3 dim_qy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_z = idx / dim_qy.y;
        int idx_y = idx - idx_z * dim_qy.y;
        int idx_qy_base = idx_y + idx_z * dim_qy.x * dim_qy.y;
        int idx_zv_base = idx_y + idx_z * (dim_qy.x - 1) * dim_qy.y;

        qy[idx_qy_base] = 0.f;
        for (int i = 1; i < dim_qy.x; i++)
        {
            qy[idx_qy_base + i * dim_qy.y] = qy[idx_qy_base + (i - 1) * dim_qy.y] + zv[idx_zv_base + (i - 1) * dim_qy.y] * dx.x;
        }
    }
}

__global__ void SweepingQz_x(DTYPE* qz, DTYPE* yv, DTYPE dx, int3 dim_qz, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_yv = { dim_qz.x - 1, dim_qz.y, dim_qz.z };
    int slice_yv = dim_yv.x * dim_yv.y;

    if (idx < max_size)
    {
        int idx_z = idx / dim_qz.y;
        int idx_y = idx - idx_z * dim_qz.y;
        int idx_yv_base = idx_y + idx_z * slice_yv;

        for (int i = 1; i < dim_qz.x; i++)
        {
            qz[idx + i * dim_qz.y] = qz[idx + (i - 1) * dim_qz.y] - yv[idx_yv_base + (i - 1) * dim_yv.y] * dx;
        }
    }
}

__global__ void SweepingQz_x_0(DTYPE* qz, DTYPE* yv, DTYPE dx, int3 dim_qz, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_yv = { dim_qz.x - 1, dim_qz.y, dim_qz.z };
    int slice_yv = dim_yv.x * dim_yv.y;

    if (idx < max_size)
    {
        int idx_z = idx / dim_qz.y;
        int idx_y = idx - idx_z * dim_qz.y;
        int idx_yv_base = idx_y + idx_z * slice_yv;
        qz[idx] = 0.f;

        for (int i = 1; i < dim_qz.x; i++)
        {
            qz[idx + i * dim_qz.y] = qz[idx + (i - 1) * dim_qz.y] - yv[idx_yv_base + (i - 1) * dim_yv.y] * dx;
        }
    }
}

__global__ void SweepingQz_y(DTYPE* qz, DTYPE* xv, DTYPE dx, int3 dim_qz, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_xv = { dim_qz.x, dim_qz.y - 1, dim_qz.z };
    int slice_xv = dim_xv.x * dim_xv.y;

    if (idx < max_size)
    {
        int idx_z = idx / dim_qz.x;;
        int idx_x = idx - idx_z * dim_qz.x;
        int idx_xv_base = idx_x * dim_xv.y + idx_z * slice_xv;

        for (int i = 1; i < dim_qz.y; i++)
        {
            qz[idx + i] = qz[idx + (i - 1)] + xv[idx_xv_base + (i - 1)] * dx;
        }
    }
}

__global__ void SweepingQy_x_mid(DTYPE* qy, DTYPE* zv, DTYPE3 dx, int3 dim_qy, int mid_i, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_z = idx / dim_qy.y;
        int idx_y = idx - idx_z * dim_qy.y;
        int idx_qy_base = idx_y + idx_z * dim_qy.x * dim_qy.y;
        int idx_zv_base = idx_y + idx_z * (dim_qy.x - 1) * dim_qy.y;
        
        qy[idx_qy_base] = 0.f;
        for (int i = mid_i - 1; i >= 0; i--)
        {
            qy[idx_qy_base + i * dim_qy.y] = qy[idx_qy_base + (i + 1) * dim_qy.y] - zv[idx_zv_base + i * dim_qy.y] * dx.x;
        }
        for (int i = mid_i + 1; i < dim_qy.x; i++)
        {
            qy[idx_qy_base + i * dim_qy.y] = qy[idx_qy_base + (i - 1) * dim_qy.y] + zv[idx_zv_base + (i - 1) * dim_qy.y] * dx.x;
        }
    }
}

__global__ void SweepingQx_mid(DTYPE* qx, DTYPE* yv, DTYPE3 dx, int3 dim_qx, int mid_i, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice = dim_qx.x * dim_qx.y;
    int3 dim_yv = { dim_qx.x, dim_qx.y, dim_qx.z - 1 };
    int slice_yv = dim_yv.x * dim_yv.y;

    if (idx < max_size)
    {
        int idx_x = idx / dim_qx.y;
        int idx_y = idx - idx_x * dim_qx.y;
        int idx_yv_base = idx_x * dim_yv.y + idx_y;

        for (int i = mid_i - 1; i >= 0; i--)
        {
            qx[idx + i * slice] = qx[idx + (i + 1) * slice] - yv[idx_yv_base + i * slice_yv] * dx.z;
        }
        for (int i = mid_i + 1; i < dim_qx.z; i++)
        {
            qx[idx + i * slice] = qx[idx + (i - 1) * slice] + yv[idx_yv_base + (i - 1) * slice_yv] * dx.z;
        }
    }
}

__global__ void SweepingQy_mid(DTYPE* qy, DTYPE* xv, DTYPE3 dx, int3 dim_qy, int mid_i, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice = dim_qy.x * dim_qy.y;
    int3 dim_xv = { dim_qy.x, dim_qy.y, dim_qy.z - 1 };
    int slice_xv = dim_xv.x * dim_xv.y;

    if (idx < max_size)
    {
        int idx_x = idx / dim_qy.y;;
        int idx_y = idx - idx_x * dim_qy.y;
        int idx_xv_base = idx_x * dim_xv.y + idx_y;

        for (int i = mid_i - 1; i >= 0; i--)
        {
            qy[idx + i * slice] = qy[idx + (i + 1) * slice] + xv[idx_xv_base + i * slice_xv] * dx.z;
        }
        for (int i = mid_i + 1; i < dim_qy.z; i++)
        {
            qy[idx + i * slice] = qy[idx + (i - 1) * slice] - xv[idx_xv_base + (i - 1) * slice_xv] * dx.z;
        }
    }
}

__global__ void Sweeping_x_sumup(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, int tx_level, DTYPE dx_inv)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[CONFLICT_OFFSET(dim_dst.x * blockDim.x)];
    int write_len = blockDim.y;
    int3 thread_start_idx3 = make_int3(blockIdx.y * blockDim.y, blockIdx.x * blockDim.x,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_src.x * dim_src.y;
    int sm_in_block = blockDim.x * blockDim.y;

    //int lls = dim_src.x > 32 ? (dim_src.x >> 5) + (dim_src.x & 1) : 1;
    int lls = ceil(DTYPE(dim_src.x) / 32);
#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim_src.x && idx_y < dim_src.y)
        {
            int idx_gm = idx_y + idx_x * dim_src.y + thread_start_idx3.z * slice_xy;
            //printf("idx_x, idx_y_thread: %d, %d, idx_sm: %d, idx_gm: %d, val: %f\n", idx_x, idx_y_sm, idx_sm, idx_gm, src[idx_gm]);
            sm[CONFLICT_OFFSET(idx_sm)] = src[idx_gm];
        }
        else
        {
            sm[CONFLICT_OFFSET(idx_sm)] = 0.f;
        }
    }

    __syncthreads();
    int idx_sm_block = threadIdx.x + (threadIdx.y * blockDim.x);
    int idx_thread_y = idx_sm_block >> tx_level;
    int idx_thread_x = idx_sm_block - (idx_thread_y << tx_level);

    //printf("threadIdx.x: %d, idx_thread_x : %d, idx_sm_block: %d, \n", threadIdx.x, idx_thread_x, idx_sm_block);

    DTYPE d_val;
    DTYPE a_val;

    // forward
    //int l = 1;
    {
        int next_dim = (dim_src.x + 1) >> 1;
        int write_patches = ceil(DTYPE(next_dim) / write_len);
        DTYPE start_d = 0.f;

        for (int i = 0; i < write_patches; i++)
        {
            int data_idx3_y = i * 32 + idx_thread_x;
            int a_idx_y = data_idx3_y * 2;
            int d_idx_y = data_idx3_y * 2 + 1;
            int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
            int idx_d_val = idx_thread_y + d_idx_y * blockDim.x;

            a_val = sm[CONFLICT_OFFSET(idx_a_val)] * dx_inv + start_d;
            d_val = sm[CONFLICT_OFFSET(idx_d_val)] * dx_inv + a_val;

            __syncwarp();

            //printf("idx_thread_x : %d, a_idx_y, d_idx_y: %d, %d, a_val: %f, d_val: %f\n", idx_thread_x, a_idx_y, d_idx_y, a_val, d_val);
            DTYPE expected_a = a_val + __shfl_up_sync(0xffffffff, d_val, 1);
            DTYPE expected_d = d_val + __shfl_up_sync(0xffffffff, d_val, 1);
            if ((idx_thread_x & 1))
            {
                a_val = expected_a;
                d_val = expected_d;
            }
            __syncwarp();

            expected_a = a_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 1) + 1);
            expected_d = d_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 1) + 1);
            if ((idx_thread_x & 2))
            {
                a_val = expected_a;
                d_val = expected_d;
            }
            __syncwarp();

            expected_a = a_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 3) + 1);
            expected_d = d_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 3) + 1);
            if ((idx_thread_x & 4))
            {
                a_val = expected_a;
                d_val = expected_d;
            }
            __syncwarp();

            expected_a = a_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 7) + 1);
            expected_d = d_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 7) + 1);
            if ((idx_thread_x & 8))
            {
                a_val = expected_a;
                d_val = expected_d;
            }
            __syncwarp();

            expected_a = a_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 15) + 1);
            expected_d = d_val + __shfl_up_sync(0xffffffff, d_val, (idx_thread_x & 15) + 1);
            if ((idx_thread_x & 16))
            {
                a_val = expected_a;
                d_val = expected_d;
            }
            __syncwarp();

            expected_d = __shfl_sync(0xffffffff, d_val, 31, 32);
            if (idx_thread_x == 0)
                start_d = expected_d;
            //printf("after a_idx_y, d_idx_y: %d, %d, expected_d: %f\n", a_idx_y, d_idx_y, start_d);

            //unsigned mask = __activemask();
            if (data_idx3_y < next_dim)
            {
                int idx_a_val = idx_thread_y + (a_idx_y + 1) * blockDim.x;
                sm_res[CONFLICT_OFFSET(idx_a_val)] = a_val;

                //DTYPE scale_i_y = 1.f;
                int idx_d_val = idx_thread_y + (d_idx_y + 1) * blockDim.x;
                sm_res[CONFLICT_OFFSET(idx_d_val)] = d_val;
            }
            //__syncwarp(mask);
        }
        //++l;
    }

    __syncthreads();

    slice_xy = dim_dst.x * dim_dst.y;
    sm_in_block = blockDim.x * blockDim.y;

    lls = ceil(DTYPE(dim_dst.x) / 32);

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim_dst.x && idx_y < dim_dst.y)
        {
            int idx_gm = idx_y + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            //if (thread_start_idx3.z == 0)
                //printf("idx_x, idx_y_thread: %d, %d, idx_gm: %d, val: %f\n", idx_x, idx_y_sm, idx_gm, sm_dst[CONFLICT_OFFSET(idx_sm)]);
            DTYPE val = idx_x == 0 ? 0.f : sm_res[CONFLICT_OFFSET(idx_sm)];
            dst[idx_gm] = val;
        }
    }
}

__global__ void cuComputeConvergence(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_qp, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_qp.x * dim_qp.y;
    int slice_qx = (dim_qp.x - 1) * dim_qp.y;
    int slice_qy = dim_qp.x * (dim_qp.y - 1);
    int slice_qz = dim_qp.x * dim_qp.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_qp.y;
        int i = idx_xy - j * dim_qp.y;

        int idx_qb = i + j * dim_qp.y + k * slice_qb;
        int idx_qx_r = i + j * dim_qp.y + k * slice_qx;
        int idx_qx_l = idx_qx_r - dim_qp.y;
        int idx_qy_t = i + j * (dim_qp.y - 1) + k * slice_qy;
        int idx_qy_b = idx_qy_t - 1;
        int idx_qz_b = i + j * dim_qp.y + k * slice_qz;
        int idx_qz_f = idx_qz_b - slice_qz;

        DTYPE val = 0.f;
        if (j > 0)
        {
            val += -qx[idx_qx_l] * dx_inv.x;
        }
        else
        {
            val += -qx[idx_qx_r] * dx_inv.x;
        }
        if (j < dim_qp.x - 1)
        {
            val += qx[idx_qx_r] * dx_inv.x;
        }
        if (i > 0)
        {
            val += -qy[idx_qy_b] * dx_inv.y;
        }
        else
        {
            val += -qy[idx_qy_t] * dx_inv.y;
        }
        if (i < dim_qp.y - 1)
        {
            val += qy[idx_qy_t] * dx_inv.y;
        }
        if (k > 0)
        {
            val += -qz[idx_qz_f] * dx_inv.z;
        }
        else
        {
            val += -qz[idx_qz_b] * dx_inv.z;
        }
        if (k < dim_qp.z - 1)
        {
            val += qz[idx_qz_b] * dx_inv.z;
        }

        qb[idx_qb] = -val;
    }
}

__global__ void cuComputeConvergence_bak(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_qp, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_qp.x * dim_qp.y;
    int slice_qx = (dim_qp.x - 1) * dim_qp.y;
    int slice_qy = dim_qp.x * (dim_qp.y - 1);
    int slice_qz = dim_qp.x * dim_qp.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_qp.y;
        int i = idx_xy - j * dim_qp.y;

        int idx_qb = i + j * dim_qp.y + k * slice_qb;
        int idx_qx_r = i + j * dim_qp.y + k * slice_qx;
        int idx_qx_l = idx_qx_r - dim_qp.y;
        int idx_qy_t = i + j * (dim_qp.y - 1) + k * slice_qy;
        int idx_qy_b = idx_qy_t - 1;
        int idx_qz_b = i + j * dim_qp.y + k * slice_qz;
        int idx_qz_f = idx_qz_b - slice_qz;

        DTYPE val = 0.f;
        if (j > 0)
        {
            val += -qx[idx_qx_l] * dx_inv.x;
        }
        else
        {
            val += -qx[idx_qx_r] * dx_inv.x;
        }
        if (j < dim_qp.x - 1)
        {
            val += qx[idx_qx_r] * dx_inv.x;
        }
        if (i > 0)
        {
            val += -qy[idx_qy_b] * dx_inv.y;
        }
        else
        {
            val += -qy[idx_qy_t] * dx_inv.y;
        }
        if (i < dim_qp.y - 1)
        {
            val += qy[idx_qy_t] * dx_inv.y;
        }
        if (k > 0)
        {
            val += -qz[idx_qz_f] * dx_inv.z;
        }
        else
        {
            val += -qz[idx_qz_b] * dx_inv.z;
        }
        if (k < dim_qp.z - 1)
        {
            val += qz[idx_qz_b] * dx_inv.z;
        }

        qb[idx_qb] = -val;
    }
}

__global__ void cuComputeConvergenceCommon(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim.x * dim.y;
    int slice_xv = (dim.x + 1) * dim.y;
    int slice_yv = dim.x * (dim.y + 1);
    int slice_zv = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = i + j * dim.y + k * slice_qb;
        int idx_xv = i + j * dim.y + k * slice_xv;
        int idx_yv = i + j * (dim.y + 1) + k * slice_yv;
        int idx_zv = i + j * dim.y + k * slice_zv;

        DTYPE val = (xv[idx_xv + dim.y] - xv[idx_xv]) * dx_inv.x + (yv[idx_yv + 1] - yv[idx_yv]) * dx_inv.y +
            (zv[idx_zv + slice_zv] - zv[idx_zv]) * dx_inv.z;
        
        qb[idx_qb] = -val;
    }
}

__global__ void cuGetConvergencePart(DTYPE* qb_zero, DTYPE* qb, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = (dim.x + 1) * (dim.y + 1);
    int slice = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice;
        int idx_xy = idx - k * slice;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = (i + 1) + (j + 1) * (dim.y + 1) + (k + 1) * slice_qb;
        qb_zero[idx] = qb[idx_qb];
    }
}

__global__ void cuRecoverConvergencePart(DTYPE* qb, DTYPE* qb_zero, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = (dim.x + 1) * (dim.y + 1);
    int slice = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice;
        int idx_xy = idx - k * slice;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = (i + 1) + (j + 1) * (dim.y + 1) + (k + 1) * slice_qb;
        qb[idx_qb] = qb_zero[idx];
    }
}

__global__ void cuGradientQxyz(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* qp, int3 dim_qp, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_qp.x * dim_qp.y;
    int slice_qx = (dim_qp.x - 1) * dim_qp.y;
    int slice_qy = dim_qp.x * (dim_qp.y - 1);
    int slice_qz = dim_qp.x * dim_qp.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_qp.y;
        int i = idx_xy - j * dim_qp.y;

        int idx_qb = i + j * dim_qp.y + k * slice_qb;
        int idx_qx = i + j * dim_qp.y + k * slice_qx;
        int idx_qy = i + j * (dim_qp.y - 1) + k * slice_qy;
        int idx_qz = i + j * dim_qp.y + k * slice_qz;

        if (i < dim_qp.y - 1)
        {
            qy[idx_qy] -= (qp[idx_qb + 1] - qp[idx_qb]) * dx_inv.y;
        }
        if (j < dim_qp.x - 1)
        {
            qx[idx_qx] -= (qp[idx_qb + dim_qp.y] - qp[idx_qb]) * dx_inv.x;
        }
        if (k < dim_qp.z - 1)
        {
            qz[idx_qz] -= (qp[idx_qb + dim_qp.y * dim_qp.x] - qp[idx_qb]) * dx_inv.z;
        }
    }
}

__device__ DTYPE cuGetFracQ(DTYPE ls_q, DTYPE3 d_ls_q, DTYPE3 line_norm, DTYPE dx)
{
    //DTYPE dir = dot(d_ls_q, line_norm);
    line_norm *= -dot(d_ls_q, line_norm);
    DTYPE3 q2s = -d_ls_q * ls_q;
    DTYPE b_l = dot(q2s, q2s) / dot(q2s, line_norm);
    return clamp(0.5f  + b_l / dx, 0.f, 1.f);
}

__global__ void cuRescaleQxyz(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ls, int3 dim, DTYPE3 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice = dim.x * dim.y;
    int slice_qx = dim.x * (dim.y + 1);
    int slice_qy = (dim.x + 1) * (dim.y);
    int slice_qz = (dim.x + 1) * (dim.y + 1);
    DTYPE3 inv_dx = 1.f / dx;

    if (idx < max_size)
    {
        int k = idx / slice;
        int idx_xy = idx - k * slice;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qx = (i + 1) + j * (dim.y + 1) + (k + 1) * slice_qx;
        int idx_qy = i + (j + 1) * dim.y + (k + 1) * slice_qy;
        int idx_qz = (i + 1) + (j + 1) * (dim.y + 1) + k * slice_qz;
        DTYPE frac;
        DTYPE3 xg_qx = { DTYPE(j), DTYPE(i + 0.5f), DTYPE(k + 0.5f) };
        frac = cuGetFracQ(InterpolateQuadratic3D(ls, xg_qx.x, xg_qx.y, xg_qx.z, dim), 
            InterpolateQuadratic3D_dxyz(ls, xg_qx.x, xg_qx.y, xg_qx.z, dim, inv_dx.x),
            { 1.f, 0.f, 0.f }, dx.x);
        if (i < dim.y - 1 && k < dim.z - 1 && frac > eps<DTYPE> && frac < 1.f)
        {
            qx[idx_qx] /= frac;
        }
        if (i < dim.y - 1 && k < dim.z - 1 )
        {
            qx[idx_qx] = frac;
        }

        DTYPE3 xg_qy = { DTYPE(j + 0.5f), DTYPE(i), DTYPE(k + 0.5f) };
        frac = cuGetFracQ(InterpolateQuadratic3D(ls, xg_qy.x, xg_qy.y, xg_qy.z, dim),
            InterpolateQuadratic3D_dxyz(ls, xg_qy.x, xg_qy.y, xg_qy.z, dim, inv_dx.x),
            { 0.f, 1.f, 0.f }, dx.x);
        if (j < dim.x - 1 && k < dim.z - 1 && frac > eps<DTYPE> && frac < 1.f)
        {
            qy[idx_qy] /= frac;
        }
        //if (j < dim.x - 1 && k < dim.z - 1 )
        //{
        //    qy[idx_qy] = frac;
        //}

        //DTYPE3 xg_qz = { DTYPE(j + 0.5f), DTYPE(i + 0.5f), DTYPE(k) };
        //frac = cuGetFracQ(InterpolateQuadratic3D(ls, xg_qz.x, xg_qz.y, xg_qz.z, dim),
        //    InterpolateQuadratic3D_dxyz(ls, xg_qz.x, xg_qz.y, xg_qz.z, dim, inv_dx.x),
        //    { 0.f, 0.f, 1.f }, dx.x);
        //if (i < dim.y - 1 && j < dim.x - 1 && frac > eps<DTYPE> && frac < 1.f)
        //{
        //    qz[idx_qz] /= frac;
        //}
    }
}

__global__ void cuSetQxFracByAyAz(DTYPE* qx_frac, DTYPE* ay, DTYPE* az, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_qx = dim + make_int3(0, 1, 1);
    int3 dim_ay = dim + make_int3(0, 1, 0);
    int3 dim_az = dim + make_int3(0, 0, 1);


    if (idx < max_size)
    {
        int idx_xz = idx / dim_qx.y;
        int idx_y = idx - idx_xz * dim_qx.y;
        int idx_z = idx_xz / dim_qx.x;
        int idx_x = idx_xz - idx_z * dim_qx.x;

        int slice_ay = dim_ay.x * dim_ay.y;
        int slice_az = dim_az.x * dim_az.y;
        int idx_ay = idx_y + idx_x * dim_ay.y + idx_z * slice_ay;
        int idx_az = idx_y + idx_x * dim_az.y + idx_z * slice_az;

        DTYPE val = 1.f;
        if (idx_z < dim.z && (ay[idx_ay] < eps<DTYPE>))
        {
            val = 0.f;
        }
        if (idx_z > 0 && (ay[idx_ay - slice_ay] < eps<DTYPE>))
        {
            val = 0.f;
        }
        if (idx_y < dim.y && az[idx_az] < eps<DTYPE>)
        {
            val = 0.f;
        }
        if (idx_y > 0 && az[idx_az - 1] < eps<DTYPE>)
        {
            val = 0.f;
        }
        int idx_frac = idx_y + (idx_x + 1) * (dim_qx.y) + (idx_z) * (dim_qx.y) * (dim_qx.x + 2);
        qx_frac[idx_frac] = val;
    }
}

__global__ void cuSetQyFracByAxAz(DTYPE* qy_frac, DTYPE* ax, DTYPE* az, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_qy = dim + make_int3(1, 0, 1);
    int3 dim_ax = dim + make_int3(1, 0, 0);
    int3 dim_az = dim + make_int3(0, 0, 1);


    if (idx < max_size)
    {
        int idx_xz = idx / dim_qy.y;
        int idx_y = idx - idx_xz * dim_qy.y;
        int idx_z = idx_xz / dim_qy.x;
        int idx_x = idx_xz - idx_z * dim_qy.x;

        int slice_ax = dim_ax.x * dim_ax.y;
        int slice_az = dim_az.x * dim_az.y;
        int idx_ax = idx_y + idx_x * dim_ax.y + idx_z * slice_ax;
        int idx_az = idx_y + idx_x * dim_az.y + idx_z * slice_az;

        DTYPE val = 1.f;
        if (idx_z < dim.z && (ax[idx_ax ] < eps<DTYPE>))
        {
            val = 0.f;
        }
        if (idx_z > 0 && (ax[idx_ax - slice_ax] < eps<DTYPE>))
        {
            val = 0.f;
        }
        if (idx_x < dim.x && az[idx_az] < eps<DTYPE>)
        {
            val = 0.f;
        }
        if (idx_x> 0 && az[idx_az- dim_az.y] < eps<DTYPE>)
        {
            val = 0.f;
        }
        int idx_frac = idx_y + 1 + (idx_x) * (dim_qy.y + 2) + (idx_z) * (dim_qy.y + 2) * (dim_qy.x);
        qy_frac[idx_frac] = val;
    }
}

__global__ void cuSetQzFracByAxAy(DTYPE* qz_frac, DTYPE* ax, DTYPE* ay, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int3 dim_qz = dim + make_int3(1, 1, 0);
    int3 dim_ax = dim + make_int3(1, 0, 0);
    int3 dim_ay = dim + make_int3(0, 1, 0);


    if (idx < max_size)
    {
        int idx_xz = idx / dim_qz.y;
        int idx_y = idx - idx_xz * dim_qz.y;
        int idx_z = idx_xz / dim_qz.x;
        int idx_x = idx_xz - idx_z * dim_qz.x;

        int slice_ax = dim_ax.x * dim_ax.y;
        int slice_ay = dim_ay.x * dim_ay.y;
        int idx_ax = idx_y + idx_x * dim_ax.y + idx_z * slice_ax;
        int idx_ay = idx_y + idx_x * dim_ay.y + idx_z * slice_ay;

        DTYPE val = 1.f;
        if (idx_y < dim.y && ax[idx_ax] < eps<DTYPE>)
        {
            val = 0.f;
        }
        if (idx_y > 0 && ax[idx_ax - 1] < eps<DTYPE>)
        {
            val = 0.f;
        }
        if (idx_x < dim.x && ay[idx_ay] < eps<DTYPE>)
        {
            val = 0.f;
        }
        if (idx_x > 0 && ay[idx_ay - dim_ay.y] < eps<DTYPE>)
        {
            val = 0.f;
        }
        int idx_frac = idx_y + (idx_x) * (dim_qz.y) + (idx_z + 1) * (dim_qz.y) * (dim_qz.x);
        qz_frac[idx_frac] = val;
    }
}

__global__ void cuMulQxFrac(DTYPE* qx_out, DTYPE* qx, DTYPE* qx_frac, int3 dim_qx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim_qx.y;
        int idx_y = idx - idx_xz * dim_qx.y;
        int idx_z = idx_xz / dim_qx.x;
        int idx_x = idx_xz - idx_z * dim_qx.x;

        int idx_frac = idx_y + (idx_x + 1) * (dim_qx.y) + (idx_z) * (dim_qx.y) * (dim_qx.x + 2);
        //if (abs(qx[idx]) < eps<DTYPE> && qx_frac[idx_frac] < eps<DTYPE>)
        //{
        //    qx_out[idx] = 1.f;
        //}
        //else if (/*abs(qx[idx]) > eps<DTYPE> && */qx_frac[idx_frac] > eps<DTYPE>)
        //{
        //    qx_out[idx] = 1.f;
        //}
        //else
        //{
        //    qx_out[idx] = 0.f;
        //}

        qx_out[idx] = qx[idx] * qx_frac[idx_frac];
    }
}


__global__ void cuMulQyFrac(DTYPE* qy_out, DTYPE* qy, DTYPE* qy_frac, int3 dim_qy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim_qy.y;
        int idx_y = idx - idx_xz * dim_qy.y;
        int idx_z = idx_xz / dim_qy.x;
        int idx_x = idx_xz - idx_z * dim_qy.x;

        int idx_frac = idx_y + 1 + (idx_x) * (dim_qy.y + 2) + (idx_z) * (dim_qy.y + 2) * (dim_qy.x);
        //if (abs(qy[idx]) < eps<DTYPE> && qy_frac[idx_frac] < eps<DTYPE>)
        //{
        //    qy_out[idx] = 1.f;
        //}
        //else if (/*abs(qy[idx]) > eps<DTYPE> && */qy_frac[idx_frac] > eps<DTYPE>)
        //{
        //    qy_out[idx] = 1.f;
        //}
        //else
        //{
        //    qy_out[idx] = 0.f;
        //}

        qy_out[idx] = qy[idx] * qy_frac[idx_frac];
    }
}

__global__ void cuMulQzFrac(DTYPE* qz_out, DTYPE* qz, DTYPE* qz_frac, int3 dim_qz, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim_qz.y;
        int idx_y = idx - idx_xz * dim_qz.y;
        int idx_z = idx_xz / dim_qz.x;
        int idx_x = idx_xz - idx_z * dim_qz.x;

        int idx_frac = idx_y + (idx_x) * (dim_qz.y) + (idx_z + 1) * (dim_qz.y) * (dim_qz.x);
        //if (abs(qz[idx]) < eps<DTYPE> && qz_frac[idx_frac] < eps<DTYPE>)
        //{
        //    qz_out[idx] = 1.f;
        //}
        //else if (/*abs(qz[idx]) > eps<DTYPE> && */qz_frac[idx_frac] > eps<DTYPE>)
        //{
        //    qz_out[idx] = 1.f;
        //}
        //else
        //{
        //    qz_out[idx] = 0.f;
        //}

        qz_out[idx] = qz[idx] * qz_frac[idx_frac];
    }
}


CuParallelSweep3D::CuParallelSweep3D(int3 dim, DTYPE3 dx, char bc) : dim_(dim), dx_(dx), bc_(bc),
    cudst3d_(make_int3(dim.x + (bc == 'n' ? -1 : 1), dim.y + (bc == 'n' ? -1 : 1), dim.z + (bc == 'n' ? -1 : 1)), dx), cumg3d_(nullptr)
{
    dx_inv_ = { DTYPE(1) / dx_.x, DTYPE(1) / dx_.y, DTYPE(1) / dx_.z };
    dim_qx_ = { dim.x, dim.y + 1, dim.z + 1 };
    dim_qy_ = { dim.x + 1, dim.y, dim.z + 1 };
    dim_qz_ = { dim.x + 1, dim.y + 1, dim.z };
    dim_qp_ = { dim.x + 1, dim.y + 1, dim.z + 1 };

    slice_qx_ = dim_qx_.x * dim_qx_.y;
    slice_qy_ = dim_qy_.x * dim_qy_.y;
    slice_qz_ = dim_qz_.x * dim_qz_.y;

    size_ = dim.x * dim.y * dim.z;
    size_qp_ = dim_qp_.x * dim_qp_.y * dim_qp_.z;
    size_qx_ = dim_qx_.x * dim_qx_.y * dim_qx_.z;
    size_qy_ = dim_qy_.x * dim_qy_.y * dim_qy_.z;
    size_qz_ = dim_qz_.x * dim_qz_.y * dim_qz_.z;
}

CuParallelSweep3D::CuParallelSweep3D(int2 dim_2d, DTYPE2 dx_2d, char bc) : dim_({ dim_2d.x, dim_2d.y, 1 }), dx_({ dx_2d.x, dx_2d.y, 1.f }), bc_(bc),
cudst3d_(make_int3(dim_2d.x + (bc == 'n' ? -1 : 1), dim_2d.y + (bc == 'n' ? -1 : 1), 1), { dx_2d.x, dx_2d.y, 1.f }), cumg3d_(nullptr)
{
    int3 dim = { dim_2d.x, dim_2d.y, 1 };
    dx_inv_ = { DTYPE(1) / dx_.x, DTYPE(1) / dx_.y, DTYPE(1) / dx_.z };
    dim_qx_ = { dim.x, dim.y + 1, dim.z + 1 };
    dim_qy_ = { dim.x + 1, dim.y, dim.z + 1 };
    dim_qz_ = { dim.x + 1, dim.y + 1, dim.z };
    dim_qp_ = { dim.x + 1, dim.y + 1, dim.z + 1 };

    slice_qx_ = dim_qx_.x * dim_qx_.y;
    slice_qy_ = dim_qy_.x * dim_qy_.y;
    slice_qz_ = dim_qz_.x * dim_qz_.y;

    size_ = dim.x * dim.y * dim.z;
    size_qp_ = dim_qp_.x * dim_qp_.y * dim_qp_.z;
    size_qx_ = dim_qx_.x * dim_qx_.y * dim_qx_.z;
    size_qy_ = dim_qy_.x * dim_qy_.y * dim_qy_.z;
    size_qz_ = dim_qz_.x * dim_qz_.y * dim_qz_.z;
}

CuParallelSweep3D::~CuParallelSweep3D()
{
    if (cumg3d_)
    {
        delete cumg3d_;
    }
}

void CuParallelSweep3D::SetQxQyBoundary(DTYPE* qx, DTYPE* qy, DTYPE* xv, DTYPE* yv, DTYPE* zv, char type)
{
    if (type == 'n')
    {
        cudaCheckError(cudaMemset(qx, 0, sizeof(DTYPE) * dim_qx_.x * dim_qx_.y));
        cudaCheckError(cudaMemset(qy, 0, sizeof(DTYPE) * dim_qy_.x * dim_qy_.y));
    }
    else
    {
        cudaCheckError(cudaMemset(qx, 0, sizeof(DTYPE) * dim_qx_.x * dim_qx_.y));
        SweepingQy_x << <BLOCKS(dim_qy_.y), THREADS(dim_qy_.y) >> > (qy, zv, dx_, dim_qy_, dim_qy_.y);
    }
}

void CuParallelSweep3D::Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv)
{
    // step 1 
    cudaCheckError(cudaMemset(qz, 0, sizeof(DTYPE) * dim_qz_.x * dim_qz_.y * dim_qz_.z));

    // step 2
    SetQxQyBoundary(qx, qy, xv, yv, zv, bc_);

    // step 3
    SweepingQx << <BLOCKS(slice_qx_), THREADS(slice_qx_) >> > (qx, yv, dx_, dim_qx_, slice_qx_);
    SweepingQy << <BLOCKS(slice_qy_), THREADS(slice_qy_) >> > (qy, xv, dx_, dim_qy_, slice_qy_);

    // step 4 set boundary of qx for neumann boundary
    if (bc_ == 'n')
    {
        DTYPE* qp = CuMemoryManager::GetInstance()->GetData("tmp", dim_qp_.x * dim_qp_.y);
        cuSweepingQ_x << <BLOCKS(dim_qp_.y), THREADS(dim_qp_.y) >> > (qp, &qx[size_qx_ - slice_qx_], dim_qp_, dim_qx_, dx_.x, dim_qp_.y);
        CudaSetZero(&qx[size_qx_ - slice_qx_], slice_qx_);
        CudaSetZero(&qy[size_qy_ - slice_qy_], slice_qy_);
        CudaMatAdd(&qz[size_qz_ - slice_qz_], qp, slice_qz_, -1.f / dx_.z);

        //CudaPrintfMat(qp, make_int3(dim_qp_.x, dim_qp_.y, 1));
        //CudaPrintfMat(qx, dim_qx_);
        //CudaPrintfMat(qy, dim_qy_);
        //CudaPrintfMat(qz, dim_qz_);
    }
}

void CuParallelSweep3D::SolveFromMid(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 mid_3d)
{
    // step 1 
    cudaCheckError(cudaMemset(qz, 0, sizeof(DTYPE) * dim_qz_.x * dim_qz_.y * dim_qz_.z));

    // step 2
    int slice_qx = dim_qx_.x * dim_qx_.y;
    int slice_qy = dim_qy_.x * dim_qy_.y;
    int slice_zv = dim_.x * dim_.y;
    cudaCheckError(cudaMemset(&qx[mid_3d.z * slice_qx], 0, sizeof(DTYPE) * slice_qx));
    SweepingQy_x_mid << <BLOCKS(dim_qy_.y), THREADS(dim_qy_.y) >> > (&qy[mid_3d.z * slice_qy], &zv[mid_3d.z * slice_zv], dx_, dim_qy_, mid_3d.x, dim_qy_.y);
    //CudaPrintfMat((DTYPE*)&qy[mid_3d.z * slice_qy], { dim_qy_.x, dim_qy_.y, 1 });
    //CudaPrintfMat(&zv[mid_3d.z * slice_zv], { dim_.x, dim_.y, 1 });

    // step 3
    SweepingQx_mid << <BLOCKS(slice_qx_), THREADS(slice_qx_) >> > (qx, yv, dx_, dim_qx_, mid_3d.z, slice_qx_);
    SweepingQy_mid << <BLOCKS(slice_qy_), THREADS(slice_qy_) >> > (qy, xv, dx_, dim_qy_, mid_3d.z, slice_qy_);
}

void CuParallelSweep3D::Solve(DTYPE* qz, DTYPE* xv, DTYPE* yv)
{
    if (bc_ == 'n')
    {
        //checkCudaErrors(cudaMemset(qz, 0, sizeof(DTYPE) * dim_qz_.y));
        SweepingQz_x_0 << <BLOCKS(dim_qz_.y), THREADS(dim_qz_.y) >> > (qz, yv, dx_.x, dim_qz_, dim_qz_.y);
    }
    else
    {
        SweepingQz_y << <1, 1 >> > (qz, xv, dx_.y, dim_qz_, 1);
        SweepingQz_x << <BLOCKS(dim_qz_.y), THREADS(dim_qz_.y) >> > (qz, yv, dx_.x, dim_qz_, dim_qz_.y);
    }
}

void CuParallelSweep3D::Project(DTYPE* qx_curl, DTYPE* qy_curl, DTYPE* qz_curl, DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol)
{
    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("qb", size_qp_);
    DTYPE* qp = CuMemoryManager::GetInstance()->GetData("qp", size_qp_);
    ComputeDivergence(qb, qx, qy, qz);
    //CudaPrintfMat(qb, dim_qp_);
    //culs3d_.Solve(qp, qb, tol);
    //CudaPrintfMat(qp, dim_qp_);
    if (qx_curl != qx)
    {
        checkCudaErrors(cudaMemcpy(qx_curl, qx, sizeof(DTYPE) * size_qx_, cudaMemcpyDeviceToDevice));
    }
    if (qy_curl != qy)
    {
        checkCudaErrors(cudaMemcpy(qy_curl, qy, sizeof(DTYPE) * size_qy_, cudaMemcpyDeviceToDevice));
    }
    if (qz_curl != qz)
    {
        checkCudaErrors(cudaMemcpy(qz_curl, qz, sizeof(DTYPE) * size_qz_, cudaMemcpyDeviceToDevice));
    }
    GradientQxyz(qx_curl, qy_curl, qz_curl, qp);
    //ComputeDivergence(qb, qx_curl, qy_curl, qz_curl);
    //CudaPrintfMat(qb, dim_qp_);
}

void CuParallelSweep3D::ProjectZero(DTYPE* qx_curl, DTYPE* qy_curl, DTYPE* qz_curl, DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol)
{
    int3 dim_phi = { dim_.x - 1, dim_.y - 1, dim_.z - 1 };
    int size_phi = dim_phi.x * dim_phi.y * dim_phi.z;
    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    CuConvergence::GetInstance()->GetB_3d_q_zeros(qb, qx, qy, qz, dim_phi, dx_);

    cudst3d_.Solve(qb, qb);

    CuGradient::GetInstance()->GradientMinus3D_zero_Q(qx, qy, qz, qb, dim_phi, dx_);
}

void CuParallelSweep3D::ProjectZero(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol)
{
    int3 dim_phi = { dim_.x - 1, dim_.y - 1, dim_.z - 1 };
    int size_phi = dim_phi.x * dim_phi.y * dim_phi.z;
    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    CuConvergence::GetInstance()->GetB_3d_q_zeros(qb, qx, qy, qz, dim_phi, dx_);
//#ifdef USE_DOUBLE
//    if (dim_phi.x != dim_phi.y || dim_phi.x != dim_phi.z)
//    {
//        if (cumg3d_ == nullptr)
//        {
//            DTYPE* ax;
//            DTYPE* ay;
//            DTYPE* az;
//            int3 dim_ax = dim_phi + make_int3(1, 0, 0);
//            int3 dim_ay = dim_phi + make_int3(0, 1, 0);
//            int3 dim_az = dim_phi + make_int3(0, 0, 1);
//            int size_ax = getsize(dim_ax);
//            int size_ay = getsize(dim_ay);
//            int size_az = getsize(dim_az);
//            checkCudaErrors(cudaMalloc((void**)&ax, sizeof(DTYPE) * size_ax));
//            checkCudaErrors(cudaMalloc((void**)&ay, sizeof(DTYPE) * size_ay));
//            checkCudaErrors(cudaMalloc((void**)&az, sizeof(DTYPE) * size_az));
//
//            CudaSetValue(ax, size_ax, 1.f);
//            CudaSetValue(ay, size_ay, 1.f);
//            CudaSetValue(az, size_az, 1.f);
//
//            cumg3d_ = new CuMultigridFrac3D(ax, ay, az, dim_phi, dx_, 'z');
//        }
//        cumg3d_->Solve(qb, nullptr, qb, 20, 5, 5, 0.8f);
//    }
//    else
//    {
//        cudst3d_.Solve(qb, qb);
//    }
//#else
//    cudst3d_.Solve(qb, qb);
//#endif // USE_DOUBLE

    cudst3d_.Solve(qb, qb);
    CuGradient::GetInstance()->GradientMinus3D_zero_Q(qx, qy, qz, qb, dim_phi, dx_);

    //CuConvergence::GetInstance()->GetB_3d_q_zeros(qb, qx, qy, qz, dim_phi, dx_);
    //printf("max_convergence: %e\n", CudaFindMaxValue(qb, size_phi));
}

void CuParallelSweep3D::ProjectD(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol)
{
    int3 dim_phi = { dim_.x + 1, dim_.y + 1, dim_.z + 1 };
    int size_phi = getsize(dim_phi);
    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qb, dim_phi);
    cudst3d_.Solve(qb, qb);
    //CudaPrintfMat(qb, dim_phi);
    CuGradient::GetInstance()->GradientMinus3D_d_Q(qx, qy, qz, qb, dim_phi, dx_);

    CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qb, dim_phi);
}


__global__ void cuSetBoundByLs(DTYPE* is_bound, DTYPE* ls, int3 dim_phi, int3 dim, DTYPE inv_dx, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim_phi.y;
        int idx_y = idx - idx_xz * dim_phi.y;
        int idx_z = idx_xz / (dim_phi.x);
        int idx_x = idx_xz - idx_z * (dim_phi.x);

        //if (idx_z == 0 && idx_y == 0)
        //    printf("idx: %d (%d, %d, %d), dim_phi: (%d, %d, %d)\n", idx, idx_x, idx_y, idx_z, dim_phi.x, dim_phi.y, dim_phi.z);

        DTYPE dx = 1.f / inv_dx;

        DTYPE3 xg = make_DTYPE3(idx_x, idx_y, idx_z) + c_min * inv_dx - make_DTYPE3(0.5f, 0.5f, 0.5f);
        DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x, xg.y, xg.z, dim);
        if (abs(ls_p) > 1.f * dx)
        {
            is_bound[idx] = 0.f;
        }
        //if (idx_z == 10)
        //    printf("idx: %d (%d, %d, %d), xg: (%f, %f, %f), ls_p: %f, 5 * dx: %f\n", idx, idx_x, idx_y, idx_z, xg.x, xg.y, xg.z, ls_p, 5 * dx);
    }
}

void CuParallelSweep3D::ProjectFrac(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE tol)
{
    int3 dim_phi = { dim_.x + 1, dim_.y + 1, dim_.z + 1 };
    int size_phi = getsize(dim_phi);


    DTYPE* is_bound = CuMemoryManager::GetInstance()->GetData("is_bound", size_phi);
    int3 dim_qx_frac = dim_qx_ + make_int3(2, 0, 0);
    int3 dim_qy_frac = dim_qy_ + make_int3(0, 2, 0);
    int3 dim_qz_frac = dim_qz_ + make_int3(0, 0, 2);

    int size_qx_frac = getsize(dim_qx_frac);
    int size_qy_frac = getsize(dim_qy_frac);
    int size_qz_frac = getsize(dim_qz_frac);

    if (cumg3d_ == nullptr)
    {


        qx_frac_ = CuMemoryManager::GetInstance()->GetData("qx_frac", getsize(dim_qx_frac));
        qy_frac_ = CuMemoryManager::GetInstance()->GetData("qy_frac", getsize(dim_qy_frac));
        qz_frac_ = CuMemoryManager::GetInstance()->GetData("qz_frac", getsize(dim_qz_frac));

        CudaSetValue(qx_frac_, getsize(dim_qx_frac), 1.f);
        CudaSetValue(qy_frac_, getsize(dim_qy_frac), 1.f);
        CudaSetValue(qz_frac_, getsize(dim_qz_frac), 1.f);
        cuSetQxFracByAyAz<<<BLOCKS(size_qx_), THREADS(size_qx_)>>>(qx_frac_, ay, az, dim_, size_qx_);
        cuSetQyFracByAxAz<<<BLOCKS(size_qy_), THREADS(size_qy_)>>>(qy_frac_, ax, az, dim_, size_qy_);
        cuSetQzFracByAxAy<<<BLOCKS(size_qz_), THREADS(size_qz_)>>>(qz_frac_, ax, ay, dim_, size_qz_);

        CudaSetValue(is_bound, size_phi, 1.f);
        CuWeightedJacobi3D::BoundB(is_bound, qx_frac_, qy_frac_, qz_frac_, dim_phi);


        //CudaPrintfMat(qx_frac_, dim_qx_frac);
        //CudaPrintfMat(qy_frac_, dim_qy_frac);
        //CudaPrintfMat(qz_frac_, dim_qz_frac);
        //CudaPrintfMat(qy, dim_qy_);
        cumg3d_ = new CuMultigridFrac3D(qx_frac_, qy_frac_, qz_frac_, dim_phi, dx_, 'z');

    }

    cuMulQxFrac << <BLOCKS(size_qx_), THREADS(size_qx_) >> > (qx, qx, qx_frac_, dim_qx_, size_qx_);
    cuMulQyFrac << <BLOCKS(size_qy_), THREADS(size_qy_) >> > (qy, qy, qy_frac_, dim_qy_, size_qy_);
    cuMulQzFrac << <BLOCKS(size_qz_), THREADS(size_qz_) >> > (qz, qz, qz_frac_, dim_qz_, size_qz_);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qy, dim_qy_);
    //CudaPrintfMat(qz, dim_qz_);


    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    DTYPE* qp = CuMemoryManager::GetInstance()->GetData("ps3d_qp", size_phi);
    //CudaPrintfMat(qx, dim_qx_);
    
    CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);
    
    //CudaPrintfMat(qb, dim_phi);
    //cudst3d_.Solve(qp, qb);
    //CuGradient::GetInstance()->GradientMinus3D_d_Q(qx, qy, qz, qp, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);
    //CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);

    //cumg3d_->Solve(qp, nullptr, qb, 5, 5, 5);
    //DTYPE err_wj = cumg3d_->GetResdualError(qp, qb);
    //printf("err_wj: %e\n", err_wj);

    DTYPE* dev_dx = CuMemoryManager::GetInstance()->GetData("dev_dx", dim_phi.x);
    DTYPE* dev_dy = CuMemoryManager::GetInstance()->GetData("dev_dy", dim_phi.y);
    DTYPE* dev_dz = CuMemoryManager::GetInstance()->GetData("dev_dz", dim_phi.z);
    CudaSetValue(dev_dx, dim_phi.x, dx_.x);
    CudaSetValue(dev_dy, dim_phi.y, dx_.y);
    CudaSetValue(dev_dz, dim_phi.z, dx_.z);
    CuWeightedJacobi3D::BoundB(qb, qx_frac_, qy_frac_, qz_frac_, dim_phi);
    CudaSetValue(qx_frac_, getsize(dim_qx_frac), 1.f);
    CudaSetValue(qy_frac_, getsize(dim_qy_frac), 1.f);
    CudaSetValue(qz_frac_, getsize(dim_qz_frac), 1.f);

    CuWeightedJacobi3D::GetInstance()->SolveFracInt(qp, nullptr, qb, qx_frac_, qy_frac_, qz_frac_, is_bound, dim_phi, dev_dx, dev_dy, dev_dz, dx_, 600, 0.8f, bc_);
    DTYPE err_wj = CuWeightedJacobi3D::GetFracRhsInt(qp, qb, qx_frac_, qy_frac_, qz_frac_, is_bound, dim_phi, dev_dx, dev_dy, dev_dz, dx_, bc_);
    //printf("err_wj: %e\n", err_wj);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qp, dim_phi);
    CuGradient::GetInstance()->GradientMinus3D_d_Q(qx, qy, qz, qp, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qy, dim_qy_);
    //cuMulQxFrac << <BLOCKS(size_qx_), THREADS(size_qx_) >> > (qx, qx, qx_frac_, dim_qx_, size_qx_);
    //cuMulQyFrac << <BLOCKS(size_qy_), THREADS(size_qy_) >> > (qy, qy, qy_frac_, dim_qy_, size_qy_);
    //cuMulQzFrac << <BLOCKS(size_qz_), THREADS(size_qz_) >> > (qz, qz, qz_frac_, dim_qz_, size_qz_);

    CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qb, dim_phi);
}

void CuParallelSweep3D::ProjectFrac(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, DTYPE tol)
{
    int3 dim_phi = { dim_.x + 1, dim_.y + 1, dim_.z + 1 };
    int size_phi = getsize(dim_phi);


    DTYPE* is_bound = CuMemoryManager::GetInstance()->GetData("is_bound", size_phi);
    int3 dim_qx_frac = dim_qx_ + make_int3(2, 0, 0);
    int3 dim_qy_frac = dim_qy_ + make_int3(0, 2, 0);
    int3 dim_qz_frac = dim_qz_ + make_int3(0, 0, 2);

    int size_qx_frac = getsize(dim_qx_frac);
    int size_qy_frac = getsize(dim_qy_frac);
    int size_qz_frac = getsize(dim_qz_frac);

    if (cumg3d_ == nullptr)
    {


        qx_frac_ = CuMemoryManager::GetInstance()->GetData("qx_frac", getsize(dim_qx_frac));
        qy_frac_ = CuMemoryManager::GetInstance()->GetData("qy_frac", getsize(dim_qy_frac));
        qz_frac_ = CuMemoryManager::GetInstance()->GetData("qz_frac", getsize(dim_qz_frac));

        CudaSetValue(qx_frac_, getsize(dim_qx_frac), 1.f);
        CudaSetValue(qy_frac_, getsize(dim_qy_frac), 1.f);
        CudaSetValue(qz_frac_, getsize(dim_qz_frac), 1.f);
        cuSetQxFracByAyAz << <BLOCKS(size_qx_), THREADS(size_qx_) >> > (qx_frac_, ay, az, dim_, size_qx_);
        cuSetQyFracByAxAz << <BLOCKS(size_qy_), THREADS(size_qy_) >> > (qy_frac_, ax, az, dim_, size_qy_);
        cuSetQzFracByAxAy << <BLOCKS(size_qz_), THREADS(size_qz_) >> > (qz_frac_, ax, ay, dim_, size_qz_);

        CudaSetValue(is_bound, size_phi, 1.f);
        CuWeightedJacobi3D::BoundB(is_bound, qx_frac_, qy_frac_, qz_frac_, dim_phi);

        cuSetBoundByLs << <BLOCKS(size_phi), THREADS(size_phi) >> > (is_bound, ls, dim_phi, dim_, dx_inv_.x, { 0.f, 0.f, 0.f }, size_phi);
        //CudaPrintfMat(is_bound, dim_phi);
        //CudaPrintfMat(qx_frac_, dim_qx_frac);
        //CudaPrintfMat(qy_frac_, dim_qy_frac);
        //CudaPrintfMat(qz_frac_, dim_qz_frac);
        //CudaPrintfMat(qy, dim_qy_);
        cumg3d_ = new CuMultigridFrac3D(qx_frac_, qy_frac_, qz_frac_, dim_phi, dx_, 'z');

    }

    cuMulQxFrac << <BLOCKS(size_qx_), THREADS(size_qx_) >> > (qx, qx, qx_frac_, dim_qx_, size_qx_);
    cuMulQyFrac << <BLOCKS(size_qy_), THREADS(size_qy_) >> > (qy, qy, qy_frac_, dim_qy_, size_qy_);
    cuMulQzFrac << <BLOCKS(size_qz_), THREADS(size_qz_) >> > (qz, qz, qz_frac_, dim_qz_, size_qz_);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qy, dim_qy_);
    //CudaPrintfMat(qz, dim_qz_);


    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    DTYPE* qp = CuMemoryManager::GetInstance()->GetData("ps3d_qp", size_phi);
    //CudaPrintfMat(qx, dim_qx_);

    CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);

    //CudaPrintfMat(qb, dim_phi);
    //cudst3d_.Solve(qp, qb);
    //CuGradient::GetInstance()->GradientMinus3D_d_Q(qx, qy, qz, qp, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);
    //CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);

    //cumg3d_->Solve(qp, nullptr, qb, 5, 5, 5);
    //DTYPE err_wj = cumg3d_->GetResdualError(qp, qb);
    //printf("err_wj: %e\n", err_wj);

    DTYPE* dev_dx = CuMemoryManager::GetInstance()->GetData("dev_dx", dim_phi.x);
    DTYPE* dev_dy = CuMemoryManager::GetInstance()->GetData("dev_dy", dim_phi.y);
    DTYPE* dev_dz = CuMemoryManager::GetInstance()->GetData("dev_dz", dim_phi.z);
    CudaSetValue(dev_dx, dim_phi.x, dx_.x);
    CudaSetValue(dev_dy, dim_phi.y, dx_.y);
    CudaSetValue(dev_dz, dim_phi.z, dx_.z);
    CuWeightedJacobi3D::BoundB(qb, qx_frac_, qy_frac_, qz_frac_, dim_phi);
    CudaSetValue(qx_frac_, getsize(dim_qx_frac), 1.f);
    CudaSetValue(qy_frac_, getsize(dim_qy_frac), 1.f);
    CudaSetValue(qz_frac_, getsize(dim_qz_frac), 1.f);

    CuWeightedJacobi3D::GetInstance()->SolveFracInt(qp, nullptr, qb, qx_frac_, qy_frac_, qz_frac_, is_bound, dim_phi, dev_dx, dev_dy, dev_dz, dx_, 600, 0.8f, bc_);
    DTYPE err_wj = CuWeightedJacobi3D::GetFracRhsInt(qp, qb, qx_frac_, qy_frac_, qz_frac_, is_bound, dim_phi, dev_dx, dev_dy, dev_dz, dx_, bc_);
    printf("err_wj: %e\n", err_wj);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qp, dim_phi);
    CuGradient::GetInstance()->GradientMinus3D_d_Q(qx, qy, qz, qp, dim_phi, dx_);
    //CudaPrintfMat(qx, dim_qx_);
    //CudaPrintfMat(qy, dim_qy_);
    //cuMulQxFrac << <BLOCKS(size_qx_), THREADS(size_qx_) >> > (qx, qx, qx_frac_, dim_qx_, size_qx_);
    //cuMulQyFrac << <BLOCKS(size_qy_), THREADS(size_qy_) >> > (qy, qy, qy_frac_, dim_qy_, size_qy_);
    //cuMulQzFrac << <BLOCKS(size_qz_), THREADS(size_qz_) >> > (qz, qz, qz_frac_, dim_qz_, size_qz_);

    //CuConvergence::GetInstance()->GetB_3d_q_d(qb, qx, qy, qz, dim_phi, dx_);
    //CudaPrintfMat(qb, dim_phi);
}

void CuParallelSweep3D::Project(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE tol)
{
    if (bc_ == 'n')
    {
        ProjectZero(qx, qy, qz, tol);
    }
    else
    {
        ProjectD(qx, qy, qz, tol);
    }
}

void CuParallelSweep3D::ProjectZero(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* qp_o,
    int3 dim_o, DTYPE dx_inv_o, DTYPE3 off_o, DTYPE3 c_min)
{
    int3 dim_phi = { dim_.x - 1, dim_.y - 1, dim_.z - 1 };
    int size_phi = dim_phi.x * dim_phi.y * dim_phi.z;
    DTYPE* qb = CuMemoryManager::GetInstance()->GetData("ps3d_qb", size_phi);
    CuConvergence::GetInstance()->GetB_3d_q_zeros_interp(qb, qx, qy, qz, dim_phi, dx_, qp_o, dim_o, dx_inv_o, off_o, c_min);
    //CudaPrintfMat(qb, dim_phi);
    cudst3d_.Solve(qb, qb);
    //CudaPrintfMat(qb, dim_phi);
    CuGradient::GetInstance()->GradientMinus3D_zero_Q(qx, qy, qz, qb, dim_phi, dx_);

}

void CuParallelSweep3D::ComputeDivergence(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz)
{
    cuComputeConvergence << <BLOCKS(size_qp_), THREADS(size_qp_) >> > (qb, qx, qy, qz, dim_qp_, dx_inv_, size_qp_);
    //for (int k = 0; k < dim_qp_.z; k++)
    //{
    //    for (int j = 0; j < dim_qp_.x; j++)
    //    {
    //        for (int i = 0; i < dim_qp_.y; i++)
    //        {
    //            int idx_qb = i + j * dim_qp_.y + k * dim_qp_.x * dim_qp_.y;
    //            int idx_qx_r = i + j * dim_qx_.y + k * slice_qx_;
    //            int idx_qx_l = idx_qx_r - dim_qx_.y;
    //            int idx_qy_t = i + j * dim_qy_.y + k * slice_qy_;
    //            int idx_qy_b = idx_qy_t - 1;
    //            int idx_qz_b = i + j * dim_qz_.y + k * slice_qz_;
    //            int idx_qz_f = idx_qz_b - slice_qz_;
    //            DTYPE val = 0.f;

    //            if (j > 0)
    //            {
    //                val += -qx[idx_qx_l] * dx_inv_.x;
    //            }
    //            if (j < dim_qx_.x)
    //            {
    //                val += qx[idx_qx_r] * dx_inv_.x;
    //            }
    //            if (i > 0)
    //            {
    //                val += -qy[idx_qy_b] * dx_inv_.y;
    //            }
    //            if (i < dim_qy_.y)
    //            {
    //                val += qy[idx_qy_t] * dx_inv_.y;
    //            }
    //            if (k > 0)
    //            {
    //                val += -qz[idx_qz_f] * dx_inv_.z;
    //            }
    //            if (k < dim_qz_.z)
    //            {
    //                val += qz[idx_qz_b] * dx_inv_.z;
    //            }
    //            qb[idx_qb] = -val;
    //        }
    //    }
    //}
}

void CuParallelSweep3D::GradientQxyz(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* qp)
{
    //for (int k = 0; k < dim_qp_.z; k++)
    //{
    //    for (int j = 0; j < dim_qp_.x; j++)
    //    {
    //        for (int i = 0; i < dim_qp_.y; i++)
    //        {
    //            int idx_qb = i + j * dim_qp_.y + k * dim_qp_.x * dim_qp_.y;
    //            int idx_qx = i + j * dim_qx_.y + k * slice_qx_;
    //            int idx_qy = i + j * dim_qy_.y + k * slice_qy_;
    //            int idx_qz = i + j * dim_qz_.y + k * slice_qz_;

    //            if (i < dim_qy_.y)
    //            {
    //                qy[idx_qy] = (qp[idx_qb + 1] - qp[idx_qb]) * dx_inv_.y;
    //            }
    //            if (j < dim_qx_.x)
    //            {
    //                qx[idx_qx] = (qp[idx_qb + dim_qp_.y] - qp[idx_qb]) * dx_inv_.x;
    //            }
    //            if (k < dim_qz_.z)
    //            {
    //                qz[idx_qz] = (qp[idx_qb + dim_qp_.y * dim_qp_.x] - qp[idx_qb]) * dx_inv_.z;
    //            }
    //        }
    //    }
    //}
    cuGradientQxyz << <BLOCKS(size_qp_), THREADS(size_qp_) >> > (qx, qy, qz, qp, dim_qp_, dx_inv_, size_qp_);
}

void CuParallelSweep3D::ComputeDivergence(DTYPE* div, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx)
{
    DTYPE3 dx_inv = { DTYPE(1) / dx.x, DTYPE(1) / dx.y, DTYPE(1) / dx.z };
    int size = dim.x * dim.y * dim.z;

    cuComputeConvergenceCommon << <BLOCKS(size), THREADS(size) >> > (div, xv, yv, zv, dim, dx_inv, size);
}

void CuParallelSweep3D::UpdateGridBlockDir(dim3& grid_q, dim3& block_q, int& sm_q, int3 dim, char dir)
{
    if (dir == 'y')
    {
        const int threads_y = 32;
        const int max_thread_x = std::max(MAX_EXPECT_SM_SIZE / (dim.y), 1);
        int threads_x = std::min(8, max_thread_x);
        threads_x = dim.x > threads_x ? threads_x : dim.x;

        unsigned int blocks_x = std::ceil(double(dim.x) / threads_x);

        dim3 grid(1, blocks_x, dim.z);
        dim3 block(threads_y, threads_x, 1);

        grid_q = grid;
        block_q = block;
        sm_q = sizeof(DTYPE) * std::max(dim.y, threads_y) * threads_x * 2;
    }
    else if (dir == 'x')
    {
        const int max_thread_y = MAX_EXPECT_SM_SIZE / dim.x;
        const int expected_thread_y = std::min(max_thread_y, 8);
        const int tx_level = 5;
        const int threads_x = 1 << tx_level;
        const int threads_y = dim.y > expected_thread_y ? expected_thread_y : dim.y;

        int blocks_y = std::ceil(double(dim.y) / threads_y);

        dim3 grid(blocks_y, 1, dim.z);
        dim3 block(threads_y, threads_x, 1);

        grid_q = grid;
        block_q = block;
        sm_q = CONFLICT_OFFSET((std::max(dim.x, 32) * threads_y)) * sizeof(DTYPE) * 2;

        //printf("threads: %d, %d, %d, sm: %d\n", block.x, block.y, block.z, sm_q);
        //printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
    }
    else if (dir == 'z')
    {
        const int max_thread_y = MAX_EXPECT_SM_SIZE / dim.z;
        const int expected_thread_y = std::min(max_thread_y, 16);
        const int tx_level = 5;
        const int threads_x = 1 << tx_level;
        const int threads_y = dim.y > expected_thread_y ? expected_thread_y : dim.y;

        int blocks_y = std::ceil(double(dim.y) / threads_y);

        dim3 grid(blocks_y, 1, dim.x);
        dim3 block(threads_y, threads_x, 1);

        grid_q = grid;
        block_q = block;
        sm_q = CONFLICT_OFFSET((dim.z * threads_y)) * sizeof(DTYPE) * 2;
    }
}

void CuParallelSweep3D::Rescale(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* ls, int3 dim, DTYPE3 dx)
{
    int max_size = getsize(dim);
    cuRescaleQxyz << <BLOCKS(max_size), THREADS(max_size) >> > (qx, qy, qz, ls, dim, dx, max_size);
}