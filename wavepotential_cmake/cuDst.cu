#include "cuDst.cuh"
#include "cuMemoryManager.cuh"

CuDst::CuDst(int2 dim, DTYPE2 dx): dim_(dim), dx_(dx)
{
    dim_fft_ = { 2 * (dim.x + 1), 2 * (dim.y + 1) };
    size_fft_x_ = dim_fft_.x * dim_.y;
    size_fft_y_ = dim_.x * dim_fft_.y;
    int max_size = std::max(size_fft_x_, size_fft_y_);
    cudaCheckError(cudaMalloc((void**)&buf_, sizeof(cufftDtypeComplex) * max_size));

    int inembed_y[2] = { dim_fft_.y, dim_.x };
#ifdef USE_DOUBLE
    cufftResult r = cufftPlanMany(&plan_y_, 1, &(dim_fft_.y), inembed_y, 1, dim_fft_.y, inembed_y, 1, dim_fft_.y, CUFFT_Z2Z, dim_.x);
#else
    cufftResult r = cufftPlanMany(&plan_y_, 1, &(dim_fft_.y), inembed_y, 1, dim_fft_.y, inembed_y, 1, dim_fft_.y, CUFFT_C2C, dim_.x);
#endif
    if (r != 0) {
        printf("CUFFT FAILED! ERROR CODE: %d\n", r);
        exit(0);
    }

    len_ = { dx.x * (dim_.x + 1), dx.y * (dim_.y + 1) };

    int inembed_x[2] = { dim_.y, dim_fft_.x };
#ifdef USE_DOUBLE
    r = cufftPlanMany(&plan_x_, 1, &(dim_fft_.x), inembed_x, dim_.y, 1, inembed_x, dim_.y, 1, CUFFT_Z2Z, dim_.y);
#else
    r = cufftPlanMany(&plan_x_, 1, &(dim_fft_.x), inembed_x, dim_.y, 1, inembed_x, dim_.y, 1, CUFFT_C2C, dim_.y);
#endif
    //r = cufftPlan1d(&plan_x_, dim_fft_.x, CUFFT_C2C, 1);
    if (r != 0) {
        printf("CUFFT FAILED! ERROR CODE: %d\n", r);
        exit(0);
    }
}

CuDst::~CuDst()
{
    cudaFree(buf_);
}

__global__ void cuFlipup_x(cufftDtypeComplex* dst, DTYPE* src, int2 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        if (idx_x < dim.x)
        {
            dst[idx + dim.y].x = src[idx];
            dst[idx + (2 * (dim.x - idx_x) + 1) * dim.y].x = -src[idx];
        }
        //else
        //{
        //    dst[idx_y].x = 0.f;
        //    dst[idx_y + dim.x * (dim.y + 1)].x = 0.f;
        //}
    }
}

__global__ void cuFlipup_y(cufftDtypeComplex* dst, DTYPE* src, int2 dim, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;
        int dim_fft_y = 2 * (dim.y + 1);
        int idx_dst = idx_y + idx_x * dim_fft_y;

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

__global__ void cuScaleCopy(DTYPE* dst, cufftDtypeComplex* src, int2 dim, int dim_src_y, int2 off, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_src = idx_y + off.y + (idx_x + off.x) * dim_src_y;

        dst[idx] = src[idx_src].y * -0.5f;
    }
}

__global__ void cuScaleCopy(DTYPE* dst, cufftDtypeComplex* src, int2 dim, int dim_src_y, int2 off, DTYPE scale, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_src = idx_y + off.y + (idx_x + off.x) * dim_src_y;

        dst[idx] = src[idx_src].y * scale;
    }
}


__global__ void cuDstProj(DTYPE* dst, cufftDtypeComplex* src, int2 dim, int dim_src_y, int2 off, DTYPE2 len_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_src = idx_y + off.y + (idx_x + off.x) * dim_src_y;

        DTYPE mu = (idx_x + 1) * _M_PI<DTYPE> * len_inv.x;
        DTYPE nu = (idx_y + 1) * _M_PI<DTYPE> * len_inv.y;
        dst[idx] = src[idx_src].y * 0.5f / (mu * mu + nu * nu);
    }
}

__global__ void cuDstProj(cufftDtypeComplex* dst, cufftDtypeComplex* src, int2 dim_fft, int dim_y, DTYPE2 len_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim_fft.y;
        int idx_y = idx - idx_x * dim_fft.y;

        int idx_mu = idx_x + 1;
        int idx_nu = idx_y;
        if (idx_y > dim_y)
        {
            idx_nu = dim_fft.y - idx_y;
        }

        DTYPE mu = (idx_mu ) * _M_PI<DTYPE> *len_inv.x;
        DTYPE nu = (idx_nu ) * _M_PI<DTYPE> *len_inv.y;
        dst[idx].x = src[idx].y * 0.5f / (mu * mu + nu * nu);
        dst[idx].y = 0.f;
    }
}

__global__ void cuDstProj_dx(cufftDtypeComplex* dst, cufftDtypeComplex* src, int2 dim_fft, int dim_y, DTYPE2 dx, DTYPE scale, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim_fft.y;
        int idx_y = idx - idx_x * dim_fft.y;

        int idx_mu = idx_x + 1;
        int idx_nu = idx_y;
        if (idx_y > dim_y)
        {
            idx_nu = dim_fft.y - idx_y;
        }

        //DTYPE mu = (idx_mu)*_M_PI<DTYPE> *len_inv.x;
        //DTYPE nu = (idx_nu)*_M_PI<DTYPE> *len_inv.y;
        //dst[idx].x = src[idx].y * 0.5f / (mu * mu + nu * nu);

        DTYPE mu = (2.f - 2.f * cos((idx_mu) * _M_PI<DTYPE> * dx.x)) ;
        DTYPE nu = (2.f - 2.f * cos((idx_nu) * _M_PI<DTYPE> * dx.y));
        dst[idx].x = src[idx].y * scale / (mu + nu);
        dst[idx].y = 0.f;
    }
}

cufftResult CuDst::CufftC2CExec1D(DTYPE* dst, DTYPE* src, const cufftHandle& plan, int direction)
{
    cufftResult result;
#ifdef USE_DOUBLE
    result = cufftExecZ2Z(plan, (cufftDoubleComplex*)src, (cufftDoubleComplex*)dst, direction);
#else
    result = cufftExecC2C(plan, (cufftComplex*)src, (cufftComplex*)dst, direction);
#endif
    return result;
}

cufftResult CuDst::CufftC2CExec1D(cufftDtypeComplex* dst, cufftDtypeComplex* src, const cufftHandle& plan, int direction)
{
    cufftResult result;
#ifdef USE_DOUBLE
    result = cufftExecZ2Z(plan, (cufftDoubleComplex*)src, (cufftDoubleComplex*)dst, direction);
#else
    result = cufftExecC2C(plan, (cufftComplex*)src, (cufftComplex*)dst, direction);
#endif
    return result;
}

void CuDst::Solve(DTYPE* dst, DTYPE* f)
{
    //// x direction
    //int threads = (dim_.x + 1) * dim_.y;
    ////CudaPrintfMat(f, dim_);
    //cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_fft_.x * dim_.y));
    //cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, f, dim_, threads);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    //CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    //threads = dim_.x * dim_.y;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_.y, make_int2(1, 0), threads);
    ////CudaPrintfMat(dst, dim_);

    //// y direction
    //threads = (dim_.x) * (dim_.y);
    ////cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_.x * dim_fft_.y))
    //cuFlipup_y << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    //CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));

    //DTYPE2 len_inv = { 1.f / len_.x, 1.f / len_.y };
    ////cuDstProj << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_.y, make_int2(0, 1), len_inv, threads);
    ////CudaPrintfMat(dst, dim_);
    //threads = dim_.x * dim_fft_.y;
    //int2 test_dim = make_int2(dim_.x, dim_fft_.y);
    //cuDstProj<< < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, test_dim, dim_.y, len_inv, threads);
    ////cuDstProj_dx<< < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, test_dim, dim_.y, dx_, threads);  
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));

    //// inv y
    //CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    //DTYPE scale = -1.f / (dim_.y + 1);
    //threads = dim_.x * dim_.y;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_.y, make_int2(0, 1), scale, threads);
    ////CudaPrintfMat(dst, dim_);

    //// inv x
    //cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_fft_.x * dim_.y));
    //cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    //CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    ////PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    //threads = dim_.x * dim_.y;
    //scale = 1.f / (dim_.x + 1);
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_.y, make_int2(1, 0), scale, threads);
    ////CudaPrintfMat(dst, dim_);

    DTYPE s_x = std::sqrt(2.0f / (dim_.x + 1));
    DTYPE s_y = std::sqrt(2.0f / (dim_.y + 1));
    // x direction
    int threads = (dim_.x + 1) * dim_.y;
    //CudaPrintfMat(f, dim_);
    cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_fft_.x * dim_.y));
    cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, f, dim_, threads);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    threads = dim_.x * dim_.y;
    DTYPE scale = -0.5f * s_x ;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_.y, make_int2(1, 0), scale, threads);
    //CudaPrintfMat(dst, dim_);

    // y direction
    threads = (dim_.x) * (dim_.y);
    //cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_.x * dim_fft_.y))
    cuFlipup_y << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    //scale = -0.5f * s_y * dx_.y;
    //cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_.y, make_int2(0, 1), scale, threads);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    //CudaPrintfMat(dst, dim_);

    //DTYPE2 len_inv = { 1.f / len_.x, 1.f / len_.y };
    //CudaPrintfMat(dst, dim_);
    threads = dim_.x * dim_fft_.y;
    int2 test_dim = make_int2(dim_.x, dim_fft_.y);
    //cuDstProj<< < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, test_dim, dim_.y, len_inv, threads);
    scale = -0.5f * s_y;
    DTYPE2 dxx = { 1.f / (dim_.x + 1), 1.f / (dim_.y + 1) };
    cuDstProj_dx<< < BLOCKS(threads), THREADS(threads) >> > (buf_, buf_, test_dim, dim_.y, dxx, scale * dx_.y * dx_.x, threads);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));

    // inv y
    CufftC2CExec1D(buf_, buf_, plan_y_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_.x, dim_fft_.y));
    //DTYPE scale = -1.f / (dim_.y + 1);
    threads = dim_.x * dim_.y;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_fft_.y, make_int2(0, 1), scale, threads);
    //CudaPrintfMat(dst, dim_);

    // inv x
    cudaCheckError(cudaMemset(buf_, 0, sizeof(cufftDtypeComplex) * dim_fft_.x * dim_.y));
    cuFlipup_x << <BLOCKS(threads), THREADS(threads) >> > (buf_, dst, dim_, threads);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    CufftC2CExec1D(buf_, buf_, plan_x_, CUFFT_FORWARD);
    //PirntComplex((DTYPE*)buf_, make_int2(dim_fft_.x, dim_.y));
    threads = dim_.x * dim_.y;
    scale = -0.5f * s_x;
    cuScaleCopy << <BLOCKS(threads), THREADS(threads) >> > (dst, buf_, dim_, dim_.y, make_int2(1, 0), scale, threads);
    //CudaPrintfMat(dst, dim_);
}

void CuDst::PirntComplex(DTYPE* val, int2 dim)
{
    DTYPE* tmp = new DTYPE[dim.x * dim.y * 2];
    cudaCheckError(cudaMemcpy(tmp, val, sizeof(DTYPE) * 2 * dim.x * dim.y, cudaMemcpyDeviceToHost));
    for (int j = 0; j < dim.x; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            int idx = (i + j * dim.y) * 2;
            printf("%.4f+%.4fi ", tmp[idx], tmp[idx + 1]);
        }
        printf("\n");
    }
    printf("\n");

    delete[] tmp;
}