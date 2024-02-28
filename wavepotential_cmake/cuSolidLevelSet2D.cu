#include "cuSolidLevelSet2D.cuh"
#include "cuMemoryManager.cuh"

__device__ DTYPE cuComputeFrac2D(DTYPE dx, DTYPE ls_l, DTYPE ls_r)
{
    DTYPE ls_dif = ls_r - ls_l;
    DTYPE d = sqrtf(dx * dx - ls_dif * ls_dif);
    return (1.f - clamp((1.f - (ls_r + ls_l) / d) * 0.5f, 0.f, 1.f));
}

__global__ void cuGetAx(DTYPE* ax, DTYPE* ls, int2 dim, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ax_x = dim.x + 1;
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_ls_r = idx_y + min(idx_x, dim.x - 1) * dim.y;
        int idx_ls_l = idx_y + max(idx_x - 1, 0) * dim.y;

        ax[idx] = cuComputeFrac2D(dx, ls[idx_ls_l], ls[idx_ls_r]);
    }
}

__global__ void cuGetAy(DTYPE* ay, DTYPE* ls, int2 dim, DTYPE dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int dim_ay_y = dim.y + 1;
        int idx_x = idx / dim_ay_y;
        int idx_y = idx - idx_x * dim_ay_y;

        int idx_ls_r = min(idx_y, dim.y - 1) + idx_x * dim.y;
        int idx_ls_l = max(idx_y - 1, 0) + idx_x * dim.y;

        ay[idx] = cuComputeFrac2D(dx, ls[idx_ls_l], ls[idx_ls_r]);
    }
}

__global__ void cuInitLsSphere(DTYPE* dst, int2 dim, DTYPE2 center, DTYPE radius, DTYPE2 dx, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        DTYPE2 pos = make_DTYPE2((idx_x + 0.5f) * dx.x, (idx_y + 0.5f) * dx.y);
        DTYPE dis = length(pos - center);
        dst[idx] = dis - radius;
    }
}

CuSolidLevelSet2D::CuSolidLevelSet2D(int2 dim, DTYPE2 dx) : dim_(dim), dx_(dx)
{
    dim_ax_ = { dim.x + 1, dim.y };
    dim_ay_ = { dim.x, dim.y + 1 };
    size_ = dim.x * dim.y;
    ls_ = CuMemoryManager::GetInstance()->GetData("solidls", size_);

    size_ax_ = getsize(dim_ax_);
    size_ay_ = getsize(dim_ay_);
}

CuSolidLevelSet2D::~CuSolidLevelSet2D()
{
}

void CuSolidLevelSet2D::GetFrac(DTYPE* ax, DTYPE* ay)
{
    cuGetAx << <BLOCKS(size_ax_), THREADS(size_ax_) >> > (ax, ls_, dim_, dx_.x, size_ax_);
    cuGetAy << <BLOCKS(size_ay_), THREADS(size_ay_) >> > (ay, ls_, dim_, dx_.y, size_ay_);
}

void CuSolidLevelSet2D::InitLsSphere(DTYPE2 center, DTYPE radius)
{
    cuInitLsSphere << <BLOCKS(size_), THREADS(size_) >> > (ls_, dim_, center, radius, dx_, size_);
}

void CuSolidLevelSet2D::GetFracHost(DTYPE* ax, DTYPE* ay)
{
    DTYPE* dev_ax = CuMemoryManager::GetInstance()->GetData("tmp", size_ax_);
    DTYPE* dev_ay = CuMemoryManager::GetInstance()->GetData("tmp1", size_ay_);

    cuGetAx << <BLOCKS(size_ax_), THREADS(size_ax_) >> > (dev_ax, ls_, dim_, dx_.x, size_ax_);
    cuGetAy << <BLOCKS(size_ay_), THREADS(size_ay_) >> > (dev_ay, ls_, dim_, dx_.y, size_ay_);

    //CudaPrintfMat(ls_, dim_);

    cudaCheckError(cudaMemcpy(ax, dev_ax, sizeof(DTYPE) * size_ax_, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(ay, dev_ay, sizeof(DTYPE) * size_ay_, cudaMemcpyDeviceToHost));
}