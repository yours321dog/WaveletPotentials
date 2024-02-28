#include "cuWHHDForward.cuh"
#include "cuWaveletCompute.cuh"
#include "cudaMath.cuh"

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_l1_func>
__global__ void cuWHHDForward_y_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[dim.y * blockDim.y];
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(0, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;

    int lls = dim.y > 32 ? (dim.y >> 5) + (dim.y & 1) : 1;

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        int idx_sm = threadIdx.x + l * blockDim.x;
        int idx_x = thread_start_idx3.y + threadIdx.y;
        if (idx_sm < dim.y && idx_x < dim.x)
        {
            int idx_gm = idx_sm + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            idx_sm += threadIdx.y * dim.y;
            sm[idx_sm] = src[idx_gm];
        }
    }

    DTYPE d_val;
    DTYPE a_val;
    int ext = (dim.y & 1);
    int l10_dim = 4 + ext;

    if (levels > 2)
    {
        for (int l = levels - 1; l > 1; l--)
        {
            int level_dim = 1 << (l + 1);
            int write_dim = (level_dim >> 1) + ext;
            int write_patches = ceil(DTYPE(write_dim) / write_len);

#pragma unroll
            for (int i = 0; i < write_patches; i++)
            {
                int data_idx3_y = i * write_len + threadIdx.x - l_halo;
                int a_idx_y = data_idx3_y * 2;
                int d_idx_y = data_idx3_y * 2 + 1;
                DTYPE sign_a;
                DTYPE sign_d;

                idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
                d_idx_y = max(0, d_idx_y);
                if (data_idx3_y < write_dim + r_halo)
                {
                    //printf("a_idx_y: %d, d_idx_y: %d, a_val: %f, d_val: %f\n", a_idx_y, d_idx_y, a_val, d_val);
                    d_val = sm[threadIdx.y * dim.y + d_idx_y];
                    a_val = sm[threadIdx.y * dim.y + a_idx_y];
                }

                update_ad_func(a_val, d_val, sign_a, sign_d);
                //printf("a_idx_y: %d\t data_idx_xz: %d, a_val, d_val: %f, %f, sign_a, sign_d: %f, %f\n", a_idx_y, threadIdx.y * dim.y, a_val, d_val, sign_a, sign_d);
                unsigned mask = __activemask();
                a_idx_y = data_idx3_y;
                if (threadIdx.x - l_halo >= 0 && threadIdx.x - l_halo < write_len && a_idx_y < write_dim)
                {
                    //printf("a_idx_y: %d\t data_idx_xz: %d, a_val, d_val: %f, %f\n", a_idx_y, threadIdx.y * dim.y, a_val, d_val);
                    sm[threadIdx.y * dim.y + a_idx_y] = a_val;
                    if (a_idx_y < write_dim - ext) sm_res[threadIdx.y * dim.y + a_idx_y + write_dim] = d_val;
                }
                __syncwarp(mask);
            }
        }

        a_val = __shfl_down_sync(-1, a_val, l_halo);
    }
    else
    {
        if (threadIdx.x < 4 + ext)
        {
            a_val = sm[threadIdx.y * dim.y + threadIdx.x];
        }
    }

    if (threadIdx.x < l10_dim)
    {
        sm_res[threadIdx.y * dim.y + threadIdx.x] = update_l1_func(a_val, threadIdx.x);
        //printf("a_val: %f, val: %f\n", a_val, sm_res[threadIdx.y * dim.y + threadIdx.x]);
    }

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        int idx_sm = threadIdx.x + l * blockDim.x;
        int idx_x = thread_start_idx3.y + threadIdx.y;
        if (idx_sm < dim.y && idx_x < dim.x)
        {
            int idx_gm = idx_sm + (threadIdx.y + thread_start_idx3.y) * dim_dst.y + thread_start_idx3.z * slice_xy;
            idx_sm += threadIdx.y * dim.y;
            dst[idx_gm] = sm_res[idx_sm];
            //printf("idx_gm: %d, idx_sm: %d, val: %f, dim: (%d, %d), dim_dst: (%d, %d)\n", idx_gm, idx_sm, sm_res[idx_sm], dim.x, dim.y, dim_dst.x, dim_dst.y);
        }
    }

}

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_l1_func>
__global__ void cuWHHDForward_x_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[CONFLICT_OFFSET(dim.x * blockDim.x)];
    int write_len = blockDim.y - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.y * blockDim.y, blockIdx.x * blockDim.x,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;

    int lls = dim.x > 32 ? (dim.x >> 5) + (dim.x & 1) : 1;
    //int idx_sm_block = threadIdx.x + (threadIdx.y << ty_level);

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim.x && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            sm[CONFLICT_OFFSET(idx_sm)] = src[idx_gm];
            //printf("idx_x, idx_y_thread: %d, %d, idx_sm: %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_sm, idx_gm);
        }
    }

    __syncthreads();

    int idx_sm_block = threadIdx.x + (threadIdx.y * blockDim.x);
    int idx_thread_y = idx_sm_block >> tx_level;
    int idx_thread_x = idx_sm_block - (idx_thread_y << tx_level);
    DTYPE d_val;
    DTYPE a_val;
    int ext = (dim.x & 1);
    int l10_dim = 4 + ext;
    if (levels > 2)
    {
#pragma unroll
        for (int l = levels - 1; l > 1; l--)
        {
            int level_dim = 1 << (l + 1);
            int write_dim = (level_dim >> 1) + ext;
            int write_patches = ceil(DTYPE(write_dim) / write_len);

#pragma unroll
            for (int i = 0; i < write_patches; i++)
            {
                int data_idx3_y = i * write_len + idx_thread_x - l_halo;
                int a_idx_y = data_idx3_y * 2;
                int d_idx_y = data_idx3_y * 2 + 1;
                DTYPE sign_a = 1.;
                DTYPE sign_d = 1.;

                idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
                d_idx_y = max(0, d_idx_y);
                if (data_idx3_y < write_dim + r_halo)
                {
                    int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                    int idx_d_val = idx_thread_y + d_idx_y * blockDim.x;
                    d_val = sm[CONFLICT_OFFSET(idx_d_val)];
                    a_val = sm[CONFLICT_OFFSET(idx_a_val)];

                    //printf("a_idx_y: %d\t d_idx_y: %d\t idx_a_va, idx_d_val: %d, %d, a_val, d_val, %f, %f\n", a_idx_y, d_idx_y, idx_a_val, idx_d_val, a_val, d_val);
                }

                update_ad_func(a_val, d_val, sign_a, sign_d);

                unsigned mask = __activemask();
                a_idx_y = data_idx3_y;
                if (idx_thread_x - l_halo >= 0 && idx_thread_x - l_halo < write_len && a_idx_y < write_dim)
                {
                    //unsigned mask = __activemask();
                    int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                    int idx_d_val = idx_thread_y + (a_idx_y + write_dim) * blockDim.x;

                    sm[CONFLICT_OFFSET(idx_a_val)] = a_val;
                    if (a_idx_y < write_dim - ext) sm_res[CONFLICT_OFFSET(idx_d_val)] = d_val;
                    //printf("a_idx_y: %d\t idx_d_val: %d, data_idx_xz: %d, a_val, d_val: %f, %f, l: %d\n", a_idx_y, idx_d_val, threadIdx.y * dim.y, a_val, d_val, l);
                }
                __syncwarp(mask);
            }
        }

        a_val = __shfl_down_sync(-1, a_val, l_halo);
    }
    else if (idx_thread_x < l10_dim)
    {
        int idx_a_val = idx_thread_y + idx_thread_x * blockDim.x;
        a_val = sm[CONFLICT_OFFSET(idx_a_val)];
    }

    int idx_res = idx_thread_y + idx_thread_x * blockDim.x;
    int idx_coef = idx_thread_x >= l10_dim ? 0 : idx_thread_x;
    DTYPE res_l10 = update_l1_func(a_val, idx_coef);
    if (idx_thread_x < l10_dim)
    {
        sm_res[CONFLICT_OFFSET(idx_res)] = res_l10;
    }
    __syncthreads();

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim.x && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            //printf("idx_x, idx_y_thread: %d, %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_gm);
            dst[idx_gm] = sm_res[CONFLICT_OFFSET(idx_sm)];
        }
    }
}

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_l1_func>
__global__ void cuWHHDForward_z_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[CONFLICT_OFFSET(dim.z * blockDim.x)];
    int write_len = blockDim.y - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.y * blockDim.y, blockIdx.x * blockDim.x,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;

    int lls = dim.z > 32 ? (dim.z >> 5) + (dim.z & 1) : 1;
    //int idx_sm_block = threadIdx.x + (threadIdx.y << ty_level);

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_z = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_z * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_z < dim.z && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_z * slice_xy + thread_start_idx3.z * dim_dst.y;
            sm[CONFLICT_OFFSET(idx_sm)] = src[idx_gm];
            //printf("idx_x, idx_y_thread: %d, %d, idx_sm: %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_sm, idx_gm);
        }
    }

    __syncthreads();

    int idx_sm_block = threadIdx.x + (threadIdx.y * blockDim.x);
    int idx_thread_y = idx_sm_block >> tx_level;
    int idx_thread_x = idx_sm_block - (idx_thread_y << tx_level);
    DTYPE d_val;
    DTYPE a_val;
    int ext = (dim.z & 1);
    int l10_dim = 4 + ext;

    if (levels > 2)
    {
#pragma unroll
        for (int l = levels - 1; l > 1; l--)
        {
            int level_dim = 1 << (l + 1);
            int write_dim = (level_dim >> 1) + ext;
            int write_patches = ceil(DTYPE(write_dim) / write_len);

#pragma unroll
            for (int i = 0; i < write_patches; i++)
            {
                int data_idx3_y = i * write_len + idx_thread_x - l_halo;
                int a_idx_y = data_idx3_y * 2;
                int d_idx_y = data_idx3_y * 2 + 1;
                DTYPE sign_a = 1.;
                DTYPE sign_d = 1.;

                idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
                d_idx_y = max(0, d_idx_y);

                if (data_idx3_y < write_dim + r_halo)
                {
                    int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                    int idx_d_val = idx_thread_y + d_idx_y * blockDim.x;
                    d_val = sm[CONFLICT_OFFSET(idx_d_val)];
                    a_val = sm[CONFLICT_OFFSET(idx_a_val)];

                    //printf("a_idx_y: %d\t d_idx_y: %d\t idx_a_va, idx_d_val: %d, %d, a_val, d_val, %f, %f\n", a_idx_y, d_idx_y, idx_a_val, idx_d_val, a_val, d_val);
                }

                update_ad_func(a_val, d_val, sign_a, sign_d);

                unsigned mask = __activemask();
                a_idx_y = data_idx3_y;
                if (idx_thread_x - l_halo >= 0 && idx_thread_x - l_halo < write_len && a_idx_y < write_dim)
                {
                    //unsigned mask = __activemask();
                    int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                    int idx_d_val = idx_thread_y + (a_idx_y + write_dim) * blockDim.x;

                    sm[CONFLICT_OFFSET(idx_a_val)] = a_val;
                    if (a_idx_y < write_dim - ext) sm_res[CONFLICT_OFFSET(idx_d_val)] = d_val;
                    //printf("a_idx_y: %d\t idx_d_val: %d, data_idx_xz: %d, a_val, d_val: %f, %f, l: %d\n", a_idx_y, idx_d_val, threadIdx.y * dim.y, a_val, d_val, l);
                }
                __syncwarp(mask);
            }
        }

        a_val = __shfl_down_sync(-1, a_val, l_halo);
    }
    else if (idx_thread_x < l10_dim)
    {
        int idx_a_val = idx_thread_y + idx_thread_x * blockDim.x;
        a_val = sm[CONFLICT_OFFSET(idx_a_val)];
    }

    int idx_coef = idx_thread_x >= l10_dim ? 0 : idx_thread_x;
    int idx_res = idx_thread_y + idx_thread_x * blockDim.x;
    DTYPE res_l10 = update_l1_func(a_val, idx_coef);
    if (idx_thread_x < l10_dim)
    {
        sm_res[CONFLICT_OFFSET(idx_res)] = res_l10;
    }
    __syncthreads();

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_z = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_z * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_z < dim.z && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_z * slice_xy + thread_start_idx3.z * dim_dst.y;
            //printf("idx_x, idx_y_thread: %d, %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_gm);
            dst[idx_gm] = sm_res[CONFLICT_OFFSET(idx_sm)];
        }
    }
}

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_il10_func>
__global__ void cuWHHDBackward_y_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[dim.y * blockDim.y];
    int write_len = blockDim.x - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(0, blockIdx.y * blockDim.y,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;

    DTYPE* sm_dst = sm_res;
    DTYPE* sm_src = sm;
    if (levels & 1)
    {
        sm_dst = sm;
        sm_src = sm_res;
    }

    int lls = dim.y > 32 ? (dim.y >> 5) + (dim.y & 1) : 1;

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        int idx_sm = threadIdx.x + l * blockDim.x;
        int idx_x = thread_start_idx3.y + threadIdx.y;
        if (idx_sm < dim.y && idx_x < dim.x)
        {
            int idx_gm = idx_sm + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            idx_sm += threadIdx.y * dim.y;
            sm[idx_sm] = src[idx_gm];
            //printf("threadIdx.y: %d, idx_sm: %d, idx_gm: %d, val: %f\n", threadIdx.y, idx_sm, idx_gm, sm[idx_sm]);
        }
    }


    DTYPE a_val;
    DTYPE d_val;
    int ext = dim.y & 1;

    int l10_dim = 4 + ext;
    int idx_coef = threadIdx.x >= l10_dim ? 0 : threadIdx.x;
    if (threadIdx.x < l10_dim)
    {
        a_val = sm[threadIdx.y * dim.y + threadIdx.x];
    }
    DTYPE res_l1 = update_il10_func(a_val, idx_coef);
    if (threadIdx.x < l10_dim)
    {
        //printf("threadidx_x: %d, a_val: %f, res_l10: %f\n", threadIdx.x, a_val, res_l1);
        sm_dst[threadIdx.y * dim.y + threadIdx.x] = res_l1;
    }

    __syncwarp(-1);
    for (int l = 2; l < levels; l++)
    {
        //unsigned mask = __activemask();
        //__syncwarp(-1);

        int level_dim = 1 << l;
        int write_dim = level_dim + ext;
        int write_patches = ceil(DTYPE(write_dim) / write_len);

        swap_pointer(sm_dst, sm_src);

#pragma unroll
        for (int i = 0; i < write_patches; i++)
        {
            int data_idx3_y = i * write_len + threadIdx.x - l_halo;

            int a_idx_y = data_idx3_y;
            int d_idx_y = data_idx3_y;

            DTYPE sign_a = 1.f;
            DTYPE sign_d = 1.f;

            idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
           
            if (data_idx3_y < level_dim + r_halo)
            {
                d_val = sm[threadIdx.y * dim.y + d_idx_y + level_dim + ext];
                a_val = sm_src[threadIdx.y * dim.y + a_idx_y];
                //printf("threadIdx.y: %d, ad_idx_y: %d, a_val: %f\n", threadIdx.y, ad_idx_y, a_val);
            }
            //printf("0_threadIdx.y: %d, ad_idx_y: %d, a_val: %f, d_val: %f, sign_a: %f\n", threadIdx.y, a_idx_y, a_val, d_val, sign_a);
            update_ad_func(a_val, d_val, sign_a, sign_d);

            a_idx_y = data_idx3_y * 2;
            if (threadIdx.x - l_halo >= 0 && threadIdx.x - r_halo < write_len)
            {
                if (data_idx3_y < level_dim + ext)
                {
                    sm_dst[threadIdx.y * dim.y + a_idx_y] = a_val;

                    //printf("threadIdx.y: %d, ad_idx_y: %d, a_val: %f, d_val: %f, level_dim: %d\n", threadIdx.y, a_idx_y, a_val, d_val, level_dim);
                }
                if (data_idx3_y < level_dim)
                    sm_dst[threadIdx.y * dim.y + a_idx_y + 1] = d_val;
            }
        }
    }

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        int idx_sm = threadIdx.x + l * blockDim.x;
        int idx_x = thread_start_idx3.y + threadIdx.y;
        if (idx_sm < dim.y && idx_x < dim.x)
        {
            int idx_gm = idx_sm + (threadIdx.y + thread_start_idx3.y) * dim_dst.y + thread_start_idx3.z * slice_xy;
            idx_sm += threadIdx.y * dim.y;
            dst[idx_gm] = sm_dst[idx_sm];
            //printf("idx_sm: %d, idx_gm: %d, val: %f\n", idx_sm, idx_gm, sm_dst[idx_sm]);
        }
    }

}

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_il10_func>
__global__ void cuWHHDBackward_x_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[CONFLICT_OFFSET(dim.x * blockDim.x)];
    int write_len = blockDim.y - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.y * blockDim.y, blockIdx.x * blockDim.x,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;
    int sm_in_block = blockDim.x * blockDim.y;

    int lls = dim.x > 32 ? (dim.x >> 5) + (dim.x & 1) : 1;

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim.x && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            //printf("idx_x, idx_y_thread: %d, %d, idx_sm: %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_sm, idx_gm);
            sm[CONFLICT_OFFSET(idx_sm)] = src[idx_gm];
        }
    }

    __syncthreads();

    DTYPE* sm_dst = sm_res;
    DTYPE* sm_src = sm;

    if (levels & 1)
    {
        sm_dst = sm;
        sm_src = sm_res;
    }

    //int level_dim;
    //DTYPE d_val_pre = 0.f;
    //DTYPE a_val_pre = 0.f;
    int idx_sm_block = threadIdx.x + (threadIdx.y * blockDim.x);
    int idx_thread_y = idx_sm_block >> tx_level;
    int idx_thread_x = idx_sm_block - (idx_thread_y << tx_level);

    DTYPE d_val;
    DTYPE a_val;

    int ext = dim.x & 1;

    int l10_dim = 4 + ext;
    int idx_coef = idx_thread_x >= l10_dim ? 0 : idx_thread_x;
    int idx_a_val = idx_thread_y + idx_thread_x * blockDim.x;

    if (idx_thread_x < l10_dim)
    {
        a_val = sm[CONFLICT_OFFSET(idx_a_val)];
    }

    DTYPE res_l1 = update_il10_func(a_val, idx_coef);

    if (idx_thread_x < l10_dim)
    {
        //printf("threadidx_x: %d, a_val: %f, res_l10: %f\n", idx_thread_x, a_val, res_l1);
        sm_dst[CONFLICT_OFFSET(idx_a_val)] = res_l1;
    }

    __syncwarp(-1);

#pragma unroll
    for (int l = 2; l < levels; l++)
    {
        int level_dim = 1 << l;
        int write_patches = ceil(DTYPE(level_dim + ext) / write_len);

        swap_pointer(sm_dst, sm_src);

#pragma unroll
        for (int i = 0; i < write_patches; i++)
        {
            int data_idx3_y = i * write_len + idx_thread_x - l_halo;

            int a_idx_y = data_idx3_y;
            int d_idx_y = data_idx3_y;

            DTYPE sign_a = 1.f;
            DTYPE sign_d = 1.f;

            idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
            if (data_idx3_y < level_dim + r_halo)
            {
                int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                int idx_d_val = idx_thread_y + (d_idx_y + level_dim + ext) * blockDim.x;

                d_val = sm[CONFLICT_OFFSET(idx_d_val)];
                a_val = sm_src[CONFLICT_OFFSET(idx_a_val)];
            }

            update_ad_func(a_val, d_val, sign_a, sign_d);
            a_idx_y = data_idx3_y * 2;

            if (idx_thread_x - l_halo >= 0 && idx_thread_x - l_halo < write_len)
            {
                //printf("a_idx_y: %d\t data_idx_xz: %d\n", a_idx_y, data_idx_xz);
                int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                int idx_d_val = idx_thread_y + (a_idx_y + 1) * blockDim.x;

                //printf("a_idx_y: %d\t idx_a_val, idx_d_val: %d, %d, a_val, d_val, %f, %f\n", a_idx_y, idx_a_val, idx_d_val, a_val, d_val);
                if (data_idx3_y < level_dim + ext)
                    sm_dst[CONFLICT_OFFSET(idx_a_val)] = a_val;
                if (data_idx3_y < level_dim)
                    sm_dst[CONFLICT_OFFSET(idx_d_val)] = d_val;
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_x = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_x * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_x < dim.x && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_x * dim_dst.y + thread_start_idx3.z * slice_xy;
            //printf("idx_x, idx_y_thread: %d, %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_gm);
            dst[idx_gm] = sm_res[CONFLICT_OFFSET(idx_sm)];
        }
    }


}

template <pidx_ad_func idx_ad_func, pupdate_ad_func update_ad_func, pupdate_l10_func update_il10_func>
__global__ void cuWHHDBackward_z_ml(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int levels, int tx_level, int l_halo, int r_halo)
{
    extern __shared__ DTYPE sm[];
    DTYPE* sm_res = &sm[CONFLICT_OFFSET(dim.z * blockDim.x)];
    int write_len = blockDim.y - l_halo - r_halo;
    int3 thread_start_idx3 = make_int3(blockIdx.y * blockDim.y, blockIdx.x * blockDim.x,
        blockDim.z * blockIdx.z);
    int slice_xy = dim_dst.x * dim_dst.y;
    int sm_in_block = blockDim.x * blockDim.y;

    int lls = dim.z > 32 ? (dim.z >> 5) + (dim.z & 1) : 1;

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_z = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_z * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_z < dim.z && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_z * slice_xy + thread_start_idx3.z * dim_dst.y;
            //printf("idx_x, idx_y_thread: %d, %d, idx_sm: %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_sm, idx_gm);
            sm[CONFLICT_OFFSET(idx_sm)] = src[idx_gm];
        }
    }

    __syncthreads();

    DTYPE* sm_dst = sm_res;
    DTYPE* sm_src = sm;

    if (levels & 1)
    {
        sm_dst = sm;
        sm_src = sm_res;
    }

    //int level_dim;
    //DTYPE d_val_pre = 0.f;
    //DTYPE a_val_pre = 0.f;
    int idx_sm_block = threadIdx.x + (threadIdx.y * blockDim.x);
    int idx_thread_y = idx_sm_block >> tx_level;
    int idx_thread_x = idx_sm_block - (idx_thread_y << tx_level);

    DTYPE d_val;
    DTYPE a_val;

    int ext = dim.z & 1;

    int l10_dim = 4 + ext;
    int idx_coef = idx_thread_x >= l10_dim ? 0 : idx_thread_x;
    int idx_a_val = idx_thread_y + idx_thread_x * blockDim.x;

    if (idx_thread_x < l10_dim)
    {
        a_val = sm[CONFLICT_OFFSET(idx_a_val)];
    }

    DTYPE res_l1 = update_il10_func(a_val, idx_coef);

    if (idx_thread_x < l10_dim)
    {
        //printf("threadidx_x: %d, a_val: %f, res_l10: %f\n", idx_thread_x, a_val, res_l1);
        sm_dst[CONFLICT_OFFSET(idx_a_val)] = res_l1;
    }

    __syncwarp(-1);

#pragma unroll
    for (int l = 2; l < levels; l++)
    {
        int level_dim = 1 << l;
        int write_patches = ceil(DTYPE(level_dim + ext) / write_len);

        swap_pointer(sm_dst, sm_src);

#pragma unroll
        for (int i = 0; i < write_patches; i++)
        {
            int data_idx3_y = i * write_len + idx_thread_x - l_halo;

            int a_idx_y = data_idx3_y;
            int d_idx_y = data_idx3_y;

            DTYPE sign_a = 1.f;
            DTYPE sign_d = 1.f;

            idx_ad_func(a_idx_y, d_idx_y, sign_a, sign_d, level_dim);
            if (data_idx3_y < level_dim + r_halo)
            {
                int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                int idx_d_val = idx_thread_y + (d_idx_y + level_dim + ext) * blockDim.x;

                d_val = sm[CONFLICT_OFFSET(idx_d_val)];
                a_val = sm_src[CONFLICT_OFFSET(idx_a_val)];
            }

            update_ad_func(a_val, d_val, sign_a, sign_d);
            a_idx_y = data_idx3_y * 2;

            if (idx_thread_x - l_halo >= 0 && idx_thread_x - l_halo < write_len)
            {
                //printf("a_idx_y: %d\t data_idx_xz: %d\n", a_idx_y, data_idx_xz);
                int idx_a_val = idx_thread_y + a_idx_y * blockDim.x;
                int idx_d_val = idx_thread_y + (a_idx_y + 1) * blockDim.x;

                //printf("a_idx_y: %d\t idx_a_val, idx_d_val: %d, %d, a_val, d_val, %f, %f\n", a_idx_y, idx_a_val, idx_d_val, a_val, d_val);
                if (data_idx3_y < level_dim + ext)
                    sm_dst[CONFLICT_OFFSET(idx_a_val)] = a_val;
                if (data_idx3_y < level_dim)
                    sm_dst[CONFLICT_OFFSET(idx_d_val)] = d_val;
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (int l = 0; l < lls; l++)
    {
        //int idx_sm = threadIdx.x;
        int idx_z = l * blockDim.y + threadIdx.y;
        int idx_y_sm = threadIdx.x;
        int idx_sm = idx_y_sm + (idx_z * blockDim.x);
        int idx_y = thread_start_idx3.y + idx_y_sm;
        if (idx_z < dim.z && idx_y < dim.y)
        {
            int idx_gm = idx_y + idx_z * slice_xy + thread_start_idx3.z * dim_dst.y;
            //printf("idx_x, idx_y_thread: %d, %d, idx_gm: %d\n", idx_x, idx_y_sm, idx_gm);
            dst[idx_gm] = sm_res[CONFLICT_OFFSET(idx_sm)];
        }
    }


}

CuWHHDForward::CuWHHDForward()
{
    // 35 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_n, cuUpdateCdf35_ad, cuUpdateCdf35_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_d, cuUpdateCdf35_ad, cuUpdateCdf35_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_n, cuUpdateCdf35_ad, cuUpdateCdf35_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_d, cuUpdateCdf35_ad, cuUpdateCdf35_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_n, cuUpdateCdf35_ad, cuUpdateCdf35_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_d, cuUpdateCdf35_ad, cuUpdateCdf35_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'y', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'y', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'x', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'x', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'z', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'z', true)];

    // 35 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_n_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_d_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_n_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_d_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_n_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_d_i, cuUpdateCdf35_ad_i, cuUpdateCdf35_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'y', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'y', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'x', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'x', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'z', 'z', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_5, 'd', 'z', false)];

    // 26 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_n, cuUpdateCdf26_ad, cuUpdateCdf26_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_d, cuUpdateCdf26_ad, cuUpdateCdf26_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_n, cuUpdateCdf26_ad, cuUpdateCdf26_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_d, cuUpdateCdf26_ad, cuUpdateCdf26_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_n, cuUpdateCdf26_ad, cuUpdateCdf26_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_d, cuUpdateCdf26_ad, cuUpdateCdf26_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_z, cuUpdateCdf26_ad, cuUpdateCdf26_l10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_z, cuUpdateCdf26_ad, cuUpdateCdf26_l10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_z, cuUpdateCdf26_ad, cuUpdateCdf26_l10_z>;

    // 26 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_n_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_d_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_n_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_d_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_n_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_d_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_z_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_z_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_6, 'z', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_z_i, cuUpdateCdf26_ad_i, cuUpdateCdf26_il10_z>;

    // 44 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF4_4_n, cuUpdateCdf44_ad, cuUpdateCdf44_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF4_4_d, cuUpdateCdf44_ad, cuUpdateCdf44_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF4_4_n, cuUpdateCdf44_ad, cuUpdateCdf44_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF4_4_d, cuUpdateCdf44_ad, cuUpdateCdf44_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF4_4_n, cuUpdateCdf44_ad, cuUpdateCdf44_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF4_4_d, cuUpdateCdf44_ad, cuUpdateCdf44_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'y', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'y', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'x', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'x', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'z', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'z', true)];

    // 44 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF4_4_n_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF4_4_d_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF4_4_n_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF4_4_d_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF4_4_n_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF4_4_d_i, cuUpdateCdf44_ad_i, cuUpdateCdf44_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'y', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'y', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'x', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'x', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'z', 'z', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_4, 'n', 'z', false)];

    // 37 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_n, cuUpdateCdf37_ad, cuUpdateCdf37_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_d, cuUpdateCdf37_ad, cuUpdateCdf37_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_n, cuUpdateCdf37_ad, cuUpdateCdf37_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_d, cuUpdateCdf37_ad, cuUpdateCdf37_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_n, cuUpdateCdf37_ad, cuUpdateCdf37_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_d, cuUpdateCdf37_ad, cuUpdateCdf37_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'y', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'y', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'x', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'x', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'z', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'z', true)];

    // 37 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_n_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_d_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_n_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_d_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_n_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_d_i, cuUpdateCdf37_ad_i, cuUpdateCdf37_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'y', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'y', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'x', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'x', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'z', 'z', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF3_7, 'd', 'z', false)];

    // 28 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_n, cuUpdateCdf28_ad, cuUpdateCdf28_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_d, cuUpdateCdf28_ad, cuUpdateCdf28_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_n, cuUpdateCdf28_ad, cuUpdateCdf28_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_d, cuUpdateCdf28_ad, cuUpdateCdf28_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_n, cuUpdateCdf28_ad, cuUpdateCdf28_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_d, cuUpdateCdf28_ad, cuUpdateCdf28_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_z, cuUpdateCdf28_ad, cuUpdateCdf28_l10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_z, cuUpdateCdf28_ad, cuUpdateCdf28_l10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_z, cuUpdateCdf28_ad, cuUpdateCdf28_l10_z>;

    // 28 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_n_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_d_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_n_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_d_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_n_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_d_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_z_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_z_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_z>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_8, 'z', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_z_i, cuUpdateCdf28_ad_i, cuUpdateCdf28_il10_z>;

    // 46 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF4_4_n, cuUpdateCdf46_ad, cuUpdateCdf46_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF4_4_d, cuUpdateCdf46_ad, cuUpdateCdf46_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF4_4_n, cuUpdateCdf46_ad, cuUpdateCdf46_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF4_4_d, cuUpdateCdf46_ad, cuUpdateCdf46_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF4_4_n, cuUpdateCdf46_ad, cuUpdateCdf46_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF4_4_d, cuUpdateCdf46_ad, cuUpdateCdf46_l10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'y', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'y', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'x', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'x', true)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'z', true)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'z', true)];

    // 46 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF4_4_n_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF4_4_d_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF4_4_n_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF4_4_d_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF4_4_n_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF4_4_d_i, cuUpdateCdf46_ad_i, cuUpdateCdf46_il10_d>;

    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'y', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'y', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'x', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'x', false)];
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'z', 'z', false)] =
        whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF4_6, 'n', 'z', false)];

    // 20 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_n, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF2_6_d, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_n, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF2_6_d, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_n, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF2_6_d, cuUpdateCdf20_ad, cuUpdateCdf20_l10>;

    // 20 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_n_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF2_6_d_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_n_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF2_6_d_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_n_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF2_0, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF2_6_d_i, cuUpdateCdf20_ad_i, cuUpdateCdf20_il10>;

    // 11 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_n, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_d, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_n, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_d, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_n, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_d, cuUpdateCdf11_ad, cuUpdateCdf11_l10>;

    // 11 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_n_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_d_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_n_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_d_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_n_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_1, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_d_i, cuUpdateCdf11_ad_i, cuUpdateCdf11_il10>;

    // 17 forward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_n, cuUpdateCdf17_ad, cuUpdateCdf17_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'y', true)] =
        cuWHHDForward_y_ml < cuWT_CDF3_5_d, cuUpdateCdf17_ad, cuUpdateCdf17_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_n, cuUpdateCdf17_ad, cuUpdateCdf17_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'x', true)] =
        cuWHHDForward_x_ml < cuWT_CDF3_5_d, cuUpdateCdf17_ad, cuUpdateCdf17_l10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_n, cuUpdateCdf17_ad, cuUpdateCdf17_l10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'z', true)] =
        cuWHHDForward_z_ml < cuWT_CDF3_5_d, cuUpdateCdf17_ad, cuUpdateCdf17_l10_d>;

    // 17 backward
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_n_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'y', false)] =
        cuWHHDBackward_y_ml < cuWT_CDF3_5_d_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_n_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'x', false)] =
        cuWHHDBackward_x_ml < cuWT_CDF3_5_d_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_d>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'n', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_n_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_n>;
    whhd_forward_funcs_[switch_pair(WaveletType::WT_CDF1_7, 'd', 'z', false)] =
        cuWHHDBackward_z_ml < cuWT_CDF3_5_d_i, cuUpdateCdf17_ad_i, cuUpdateCdf17_il10_d>;

    // halo map
    whhd_halo_map_[WaveletType::WT_CDF1_7] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF2_6] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF4_4] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF3_5] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF3_7] = { 4, 4 };
    whhd_halo_map_[WaveletType::WT_CDF4_6] = { 4, 4 };
    whhd_halo_map_[WaveletType::WT_CDF2_8] = { 4, 4 };

    whhd_halo_map_[WaveletType::WT_CDF3_11] = { 6, 6 };
    whhd_halo_map_[WaveletType::WT_CDF4_10] = { 6, 6 };
    whhd_halo_map_[WaveletType::WT_CDF2_12] = { 6, 6 };

    whhd_halo_map_[WaveletType::WT_CDF26_44] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF44_26] = { 3, 3 };
    whhd_halo_map_[WaveletType::WT_CDF28_46] = { 4, 4 };
    whhd_halo_map_[WaveletType::WT_CDF46_28] = { 4, 4 };

    whhd_halo_map_[WaveletType::WT_CDF2_0] = { 1, 1 };
    whhd_halo_map_[WaveletType::WT_CDF1_1] = { 0, 0 };
}

CuWHHDForward::~CuWHHDForward() = default;

std::auto_ptr<CuWHHDForward> CuWHHDForward::instance_;

CuWHHDForward* CuWHHDForward::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuWHHDForward>(new CuWHHDForward); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void CuWHHDForward::Solve2D(DTYPE* dst, DTYPE* src, int3 dim, int3 levels, char2 bcs, 
    WaveletType rsw_type_x, WaveletType rsw_type_y, bool is_forward)
{
    SolveYSide(dst, src, dim, dim, levels.y, bcs.y, rsw_type_y, is_forward);
    SolveXSide(dst, dst, dim, dim, levels.x, bcs.x, rsw_type_x, is_forward);
}

void CuWHHDForward::Solve1D(DTYPE* dst, DTYPE* src, int3 dim, int level, char direction,
    char bc, WaveletType rsw_type, bool is_forward)
{
    if (src == nullptr)
    {
        src = dst;
    }
    Solve1D(dst, src, dim, dim, level, direction, bc, rsw_type, is_forward);
}

void CuWHHDForward::Solve1D(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char direction,
    char bc, WaveletType rsw_type, bool is_forward)
{
    if (src == nullptr)
    {
        src = dst;
    }
    switch (direction)
    {
    case 'x':
    case 'X':
        SolveXSide(dst, src, dim_dst, dim, level, bc, rsw_type, is_forward);
        break;
    case 'z':
    case 'Z':
        SolveZSide(dst, src, dim_dst, dim, level, bc, rsw_type, is_forward);
        break;
    default:
        SolveYSide(dst, src, dim_dst, dim, level, bc, rsw_type, is_forward);
        break;
    }
}

void CuWHHDForward::SolveYSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc,
    WaveletType rsw_type, bool is_forward)
{
    auto rsw_bc_type_y = switch_pair(rsw_type, bc, 'y', is_forward);
    int l_halo = whhd_halo_map_[rsw_type].x;
    int r_halo = whhd_halo_map_[rsw_type].y;

    int2 ext = make_int2(0, dim.y & 1);

    const int threads_y = 32;
    //const int write_dim = 32 - l_halo - r_halo;
    const int max_thread_x = std::max(MAX_EXPECT_SM_SIZE >> (level + 1), 1);
    int threads_x = std::min(8, max_thread_x);
    threads_x = dim.x > threads_x ? threads_x : dim.x;

    int blocks_x = std::ceil(double(dim.x) / threads_x);

    dim3 grid(1, blocks_x, dim.z);
    dim3 block(threads_y, threads_x, 1);

    int sm_size = sizeof(DTYPE) * dim.y * threads_x * 2;

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    whhd_forward_funcs_[rsw_bc_type_y] << <grid, block, sm_size >> > (dst, src, dim_dst, dim, level, 0, l_halo, r_halo);
    checkCudaErrors(cudaGetLastError());
}

void CuWHHDForward::SolveXSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc,
    WaveletType rsw_type, bool is_forward)
{
    auto rsw_bc_type_x = switch_pair(rsw_type, bc, 'x', is_forward);
    int l_halo = whhd_halo_map_[rsw_type].x;
    int r_halo = whhd_halo_map_[rsw_type].y;

    //int2 ext = make_int2(dim.x & 1, 0);
    //int2 dim_dst = make_int2((dim.x >> 1) + ext.x, dim.y);

    const int max_thread_y = MAX_EXPECT_SM_SIZE >> (level);
    const int expected_thread_y = std::min(max_thread_y, 8);
    const int tx_level = 5;
    const int threads_x = 1 << tx_level;
    const int threads_y = dim.y > expected_thread_y ? expected_thread_y : dim.y;

    int blocks_y = std::ceil(double(dim.y) / threads_y);

    dim3 grid(blocks_y, 1, dim.z);
    dim3 block(threads_y, threads_x, 1);

    int sm_loop_mask = 0x7F;
    int sm_out_loop_mask = 0x3F;

    int sm_size = CONFLICT_OFFSET((dim.x * threads_y)) * sizeof(DTYPE) * 2;

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    whhd_forward_funcs_[rsw_bc_type_x] << <grid, block, sm_size >> > (dst, src, dim_dst, dim, level, tx_level, l_halo, r_halo);
    checkCudaErrors(cudaGetLastError());
}

void CuWHHDForward::SolveZSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc,
    WaveletType rsw_type, bool is_forward)
{
    auto rsw_bc_type_z = switch_pair(rsw_type, bc, 'z', is_forward);
    int l_halo = whhd_halo_map_[rsw_type].x;
    int r_halo = whhd_halo_map_[rsw_type].y;

    //int2 ext = make_int2(dim.x & 1, 0);
    //int2 dim_dst = make_int2((dim.x >> 1) + ext.x, dim.y);

    const int max_thread_y = MAX_EXPECT_SM_SIZE >> (level);
    const int expected_thread_y = std::min(max_thread_y, 16);
    const int tx_level = 5;
    const int threads_x = 1 << tx_level;
    const int threads_y = dim.y > expected_thread_y ? expected_thread_y : dim.y;

    int blocks_y = std::ceil(double(dim.y) / threads_y);

    dim3 grid(blocks_y, 1, dim.x);
    dim3 block(threads_y, threads_x, 1);

    int sm_loop_mask = 0x7F;
    int sm_out_loop_mask = 0x3F;

    int sm_size = CONFLICT_OFFSET((dim.z * threads_y)) * sizeof(DTYPE) * 2;

#ifdef PRINTF_THREADS
    printf("threads: %d, %d, %d, sm: %d\n", block.x, block.y, block.z, sm_size);
    printf("blocks: %d, %d, %d\n", grid.x, grid.y, grid.z);
#endif

    whhd_forward_funcs_[rsw_bc_type_z] << <grid, block, sm_size >> > (dst, src, dim_dst, dim, level, tx_level, l_halo, r_halo);
    checkCudaErrors(cudaGetLastError());
}