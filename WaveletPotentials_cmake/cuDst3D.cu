#include "cuDst3D.cuh"

__global__ void cuFlipup_x(cufftDtypeComplex* dst, DTYPE* src, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int dim_fft_x = 2 * (dim.x + 1);

        int idx_dst = idx_y + idx_x * dim.y + idx_z * dim.y * dim_fft_x;

        if (idx_x < dim.x)
        {
            dst[idx_dst + dim.y].x = src[idx];
            dst[idx_dst + (2 * (dim.x - idx_x) + 1) * dim.y].x = -src[idx];
        }
        //else
        //{
        //    dst[idx_y].x = 0.f;
        //    dst[idx_y + dim.x * (dim.y + 1)].x = 0.f;
        //}
    }
}

__global__ void cuFlipup_x_transpose(cufftDtypeComplex* dst, DTYPE* src, int3 dim)
{
    __shared__ DTYPE sm[THREAD_DIM_2D_16 * THREAD_DIM_2D_16];

    int idx_x = threadIdx.y + blockDim.y * blockIdx.y;
    int idx_y = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_z = blockIdx.z;

    if (idx_x < dim.x && idx_y < dim.y)
    {
        int idx_sm = threadIdx.x + threadIdx.y * blockDim.x;
        int idx = idx_y + idx_x * dim.y + idx_z * dim.x * dim.y;
        sm[CONFLICT_OFFSET(idx_sm)] = src[idx];
        //printf("idx_sm: %d, idx: %d, val: %f\n", idx_sm, idx, src[idx]);
    }

    __syncthreads();

    idx_x = threadIdx.y + blockDim.x * blockIdx.x;
    idx_y = threadIdx.x + blockDim.y * blockIdx.y;
    if (idx_x < dim.y && idx_y < dim.x)
    {
        int dim_fft_x = 2 * (dim.x + 1);
        int idx_fft = (idx_y + 1) + idx_x * dim_fft_x + idx_z * dim_fft_x * dim.y;
        int idx_sm = threadIdx.y + threadIdx.x * blockDim.x;

        //printf("after idx_sm: %d, idx: %d(%d, %d), val: %f\n", idx_sm, idx_fft, idx_x, idx_y, sm[CONFLICT_OFFSET(idx_sm)]);

        dst[idx_fft].x = sm[CONFLICT_OFFSET(idx_sm)];
        idx_fft = (dim_fft_x - idx_y - 1) + idx_x * dim_fft_x + idx_z * dim_fft_x * dim.y;
        dst[idx_fft].x = -sm[CONFLICT_OFFSET(idx_sm)];
    }
}

__global__ void cuFlipup_y(cufftDtypeComplex* dst, DTYPE* src, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;

        int dim_fft_y = 2 * (dim.y + 1);
        int idx_dst = idx_y + idx_x * dim_fft_y + idx_z * dim_fft_y * dim.x;

        dst[idx_dst + 1].x = src[idx];
        dst[idx_dst + 1].y = 0.f;
        dst[idx_dst + (2 * (dim.y - idx_y) + 1)].x = -src[idx];
        dst[idx_dst + (2 * (dim.y - idx_y) + 1)].y = 0.f;

        if (idx_y == 0)
        {
            dst[idx_dst].x = 0.f;
            dst[idx_dst].y = 0.f;
        }
        if (idx_y == dim.y - 1)
        {
            dst[idx_dst + 2].x = 0.f;
            dst[idx_dst + 2].y = 0.f;
        }
    }
}

__global__ void cuFlipup_z(cufftDtypeComplex* dst, DTYPE* src, int3 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;
        int slice = dim.x * dim.y;

        //int dim_fft_z = 2 * (dim.z + 1);
        int& idx_dst = idx;

        dst[idx_dst + slice].x = src[idx];
        dst[idx_dst + slice].y = 0.f;
        dst[idx_dst + (2 * (dim.z - idx_z) + 1) * slice].x = -src[idx];
        dst[idx_dst + (2 * (dim.z - idx_z) + 1) * slice].y = 0.f;

        if (idx_z == 0)
        {
            dst[idx_dst].x = 0.f;
            dst[idx_dst].y = 0.f;
        }
        if (idx_z == dim.z - 1)
        {
            dst[idx_dst + 2 * slice].x = 0.f;
            dst[idx_dst + 2 * slice].y = 0.f;
        }
    }
}

__global__ void cuScaleCopy(DTYPE* dst, cufftDtypeComplex* src, int3 dim, int3 dim_src, int3 off, DTYPE scale, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;

        int idx_src = idx_y + off.y + (idx_x + off.x) * dim_src.y + (idx_z + off.z) * dim_src.x * dim_src.y;

        dst[idx] = src[idx_src].y * scale;
    }
}

__global__ void cuScaleCopy_x_transpose(DTYPE* dst, cufftDtypeComplex* src, int3 dim, int3 dim_src, int3 off, DTYPE scale)
{
    __shared__ DTYPE sm[THREAD_DIM_2D_16 * THREAD_DIM_2D_16];

    int idx_x = threadIdx.y + blockDim.y * blockIdx.y;
    int idx_y = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_z = blockIdx.z;

    if (idx_x < dim.y && idx_y < dim.x)
    {
        int idx_sm = threadIdx.x + threadIdx.y * blockDim.x;
        int idx_src = idx_y + off.y + (idx_x + off.x) * dim_src.y + (idx_z + off.z) * dim_src.x * dim_src.y;

        sm[CONFLICT_OFFSET(idx_sm)] = src[idx_src].y * scale;
    }

    __syncthreads();

    idx_x = threadIdx.y + blockDim.x * blockIdx.x;
    idx_y = threadIdx.x + blockDim.y * blockIdx.y;

    if (idx_x < dim.x && idx_y < dim.y)
    {
        int idx = idx_y + idx_x * dim.y + idx_z * dim.x * dim.y;
        int idx_sm = threadIdx.y + threadIdx.x * blockDim.x;
        dst[idx] = sm[CONFLICT_OFFSET(idx_sm)];
    }

}

__global__ void cuDstProj_dx(cufftDtypeComplex* dst, cufftDtypeComplex* src, int3 dim_fft, int3 dim, DTYPE3 dx, DTYPE scale, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim_fft.y;
        int idx_y = idx - idx_xz * dim_fft.y;
        int idx_z = idx_xz / dim_fft.x;
        int idx_x = idx_xz - dim_fft.x * idx_z;

        int idx_mu = idx_x + 1;
        int idx_nu = idx_y + 1;
        int idx_ku = idx_z;
        if (idx_z > dim.z)
        {
            idx_ku = dim_fft.z - idx_z;
        }

        //DTYPE mu = (idx_mu)*_M_PI<DTYPE> *len_inv.x;
        //DTYPE nu = (idx_nu)*_M_PI<DTYPE> *len_inv.y;
        //dst[idx].x = src[idx].y * 0.5f / (mu * mu + nu * nu);

        DTYPE mu = (2.f - 2.f * cos((idx_mu)*_M_PI<DTYPE> * dx.x));
        DTYPE nu = (2.f - 2.f * cos((idx_nu)*_M_PI<DTYPE> * dx.y));
        DTYPE ku = (2.f - 2.f * cos((idx_ku)*_M_PI<DTYPE> * dx.z));
        dst[idx].x = src[idx].y * scale / (mu + nu + ku);
        dst[idx].y = 0.f;
    }
}

CuDst3D::CuDst3D(int3 dim, DTYPE3 dx) : dim_(dim), dx_(dx)
{
    dim_fft_ = { 2 * (dim.x + 1), 2 * (dim.y + 1), 2 * (dim.z + 1) };
    dim_fft_x_ = { dim_fft_.x, dim_.y, dim_.z };
    dim_fft_y_ = { dim_.x, dim_fft_.y, dim_.z };
    dim_fft_z_ = { dim_.x, dim_.y, dim_fft_.z };
    size_fft_x_ = dim_fft_.x * dim_.y * dim_.z;
    size_fft_y_ = dim_.x * dim_fft_.y * dim_.z;
    size_fft_y_ = dim_.x * dim_.y * dim_fft_.z;
    int max_size = std::max(size_fft_x_, std::max(size_fft_y_, size_fft_z_));
    cudaCheckError(cudaMalloc((void**)&buf_, sizeof(cufftDtypeComplex) * max_size));

    int inembed_y[1] = { 0 };
#ifdef USE_DOUBLE
    cufftResult r = cufftPlanMany(&plan_y_, 1, &(dim_fft_.y), inembed_y, 1, dim_fft_.y, inembed_y, 1, dim_fft_.y, CUFFT_Z2Z, dim_.x * dim_.z);
#else
    cufftResult r = cufftPlanMany(&plan_y_, 1, &(dim_fft_.y), inembed_y, 1, dim_fft_.y, inembed_y, 1, dim_fft_.y, CUFFT_C2C, dim_.x * dim_.z);
#endif
    if (r != 0) {
        printf("CUFFT FAILED! ERROR CODE: %d\n", r);
        exit(0);
    }

    len_ = { dx.x * (dim_.x + 1), dx.y * (dim_.y + 1), dx.z * (dim_.z + 1) };

    int inembed_x[1] = { 1 };
#ifdef USE_DOUBLE
    r = cufftPlanMany(&plan_x_, 1, &(dim_fft_.x), inembed_x, 1, dim_fft_.x, inembed_x, 1, dim_fft_.x, CUFFT_Z2Z, dim_.y * dim_.z);
#else
    r = cufftPlanMany(&plan_x_, 1, &(dim_fft_.x), inembed_x, 1, dim_fft_.x, inembed_x, 1, dim_fft_.x, CUFFT_C2C, dim_.y * dim_.z);
#endif
    //r = cufftPlan1d(&plan_x_, dim_fft_.x, CUFFT_C2C, 1);
    if (r != 0) {
        printf("CUFFT FAILED! ERROR CODE: %d\n", r);
        exit(0);
    }

    int inembed_z[1] = { 0 };
    int slice = dim_.x * dim_.y;
#ifdef USE_DOUBLE
    r = cufftPlanMany(&plan_z_, 1, &(dim_fft_.z), inembed_z, slice, 1, inembed_z, slice, 1, CUFFT_Z2Z, dim_.x * dim_.y);
#else
    r = cufftPlanMany(&plan_z_, 1, &(dim_fft_.z), inembed_z, slice, 1, inembed_z, slice, 1, CUFFT_C2C, dim_.x * dim_.y);
#endif
    //r = cufftPlan1d(&plan_x_, dim_fft_.x, CUFFT_C2C, 1);
    if (r != 0) {
        printf("CUFFT FAILED! ERROR CODE: %d\n", r);
        exit(0);
    }

    block_x_ = { THREAD_DIM_2D_16, THREAD_DIM_2D_16, 1 };
    grid_x_ = { unsigned int(std::ceil(DTYPE(dim_.y) / THREAD_DIM_2D_16)), unsigned int(std::ceil(DTYPE(dim_.x) / THREAD_DIM_2D_16)), unsigned int(dim_.z) };
   
}

CuDst3D::~CuDst3D()
{
    cudaFree(buf_);
    cufftDestroy(plan_x_);
    cufftDestroy(plan_y_);
    cufftDestroy(plan_z_);
}

void CuDst3D::Solve(DTYPE* dst, DTYPE* f)
{
    DTYPE s_x = std::sqrt(2.0f / (dim_.x + 1));
    DTYPE s_y = std::sqrt(2.0f / (dim_.y + 1));
    DTYPE s_z = std::sqrt(2.0f / (dim_.z + 1));
    // x direction
    int threads = dim_.x * dim_.y * dim_.z;
    //CudaPrintfMat(f, dim_);
    cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * size_fft_x_));
    //cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, f, dim_, threads);
    cuFlipup_x_transpose << <grid_x_, block_x_ >> > (buf_, f, dim_);
    //PirntComplex((DTYPE*)buf_, make_int3(dim_.y, dim_fft_.x, dim_.z));
    CuDst::CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int3(dim_.y, dim_fft_.x, dim_.z));
    //threads = dim_.x * dim_.y;
    DTYPE scale = -0.5f * s_x;
    //DTYPE scale = 1.f;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_x_, make_int3(1, 0, 0), scale, threads);
    dim3 grid_x_t(grid_x_.y, grid_x_.x, grid_x_.z);
    cuScaleCopy_x_transpose<<<grid_x_t, block_x_>>>(dst, buf_, dim_, make_int3(dim_.y, dim_fft_.x, dim_.z), make_int3(0, 1, 0), scale);
    //CudaPrintfMat(dst, dim_);

    // y direction
    //threads = (dim_.x) * (dim_.y);
    //cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_.x * dim_fft_.y))
    cuFlipup_y << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    //PirntComplex((DTYPE*)buf_, dim_fft_y_);
    CuDst::CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    scale = -0.5f * s_y;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_y_, make_int3(0, 1, 0), scale, threads);
    //PirntComplex((DTYPE*)buf_, dim_fft_y_);
    //CudaPrintfMat(dst, dim_);

    // z direction
    cuFlipup_z << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    //PirntComplex((DTYPE*)buf_, dim_fft_z_);
    CuDst::CufftC2CExec1D(buf_, buf_, plan_z_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, dim_fft_z_);
    //scale = -0.5f * s_z;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_z_, make_int2(0, 0, 1), scale, threads);

    //DTYPE2 len_inv = { 1.f / len_.x, 1.f / len_.y };
    //CudaPrintfMat(dst, dim_);
    threads = dim_.x * dim_.y * dim_fft_.z;
    //int2 test_dim = make_int2(dim_.x, dim_fft_.y);
    //cuDstProj<< < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, test_dim, dim_.y, len_inv, threads);
    scale = -0.5f * s_z;
    DTYPE3 dxx = { 1.f / (dim_.x + 1), 1.f / (dim_.y + 1), 1.f / (dim_.z + 1) };
    cuDstProj_dx << < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, dim_fft_z_, dim_, dxx, scale * dx_.y * dx_.x , threads);
    //PirntComplex((DTYPE*)buf_, dim_fft_z_);

    // inv z
    CuDst::CufftC2CExec1D(buf_, buf_, plan_z_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    //DTYPE scale = -1.f / (dim_.y + 1);
    threads = dim_.x * dim_.y * dim_.z;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_z_, make_int3(0, 0, 1), scale, threads);

    // inv y
    cuFlipup_y << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    CuDst::CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    scale = -0.5f * s_y;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_y_, make_int3(0, 1, 0), scale, threads);
    //CudaPrintfMat(dst, dim_);

    // inv x
    cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * size_fft_x_));
    //cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    cuFlipup_x_transpose << <grid_x_, block_x_ >> > (buf_, dst, dim_);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    CuDst::CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    scale = -0.5f * s_x;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_x_, make_int3(1, 0, 0), scale, threads);
    cuScaleCopy_x_transpose << <grid_x_t, block_x_ >> > (dst, buf_, dim_, make_int3(dim_.y, dim_fft_.x, dim_.z), make_int3(0, 1, 0), scale);
}

void CuDst3D::PirntComplex(DTYPE* val, int3 dim)
{
    DTYPE* tmp = new DTYPE[dim.x * dim.y * dim.z * 2];
    cudaCheckError(cudaMemcpy(tmp, val, sizeof(DTYPE) * 2 * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                int idx = (i + j * dim.y + k * dim.x * dim.y) * 2;
                printf("%.4f+%.4fi ", tmp[idx], tmp[idx + 1]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}