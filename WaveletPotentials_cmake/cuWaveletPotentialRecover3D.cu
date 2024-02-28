#include "cuWaveletPotentialRecover3D.cuh"
#include "cuWHHDForward.cuh"
#include "CuDivLocalProject3D.cuh"
#include "cuMemoryManager.cuh"

CuWaveletPotentialRecover3D::CuWaveletPotentialRecover3D(int3 dim, DTYPE3 dx, char bc, WaveletType type_w_curl_, WaveletType type_w_div_)
    :dim_(dim), dim_ext_(dim + make_int3(1, 1, 1)), size_(getsize(dim)), size_ext_(getsize(dim_ext_)), dx_(dx), bc_(bc), bc_div_(proj_bc(bc)),
    cudlp_(dim + make_int3(1, 1, 1), make_int3(log2(dim.x), log2(dim.y), log2(dim.z)), dx, type_w_div_, proj_bc(bc)),
    type_w_curl_(type_w_curl_), type_w_div_(type_w_div_), levels_(make_int3(log2(dim.x), log2(dim.y), log2(dim.z)))
{

}

CuWaveletPotentialRecover3D::~CuWaveletPotentialRecover3D() = default;

void CuWaveletPotentialRecover3D::Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv, DTYPE* yv, DTYPE* zv)
{
    int3 dim_xv = dim_ + make_int3(1, 0, 0);
    int3 dim_yv = dim_ + make_int3(0, 1, 0);
    int3 dim_zv = dim_ + make_int3(0, 0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);
    int3 dim_qx = { dim_.x, dim_.y + 1, dim_.z + 1 };
    int3 dim_qy = { dim_.x + 1, dim_.y, dim_.z + 1 };
    int3 dim_qz = { dim_.x + 1, dim_.y + 1, dim_.z };

    DTYPE* xv_tmp = CuMemoryManager::GetInstance()->GetData("tmp", size_xv);
    DTYPE* yv_tmp = CuMemoryManager::GetInstance()->GetData("tmp1", size_yv);
    DTYPE* zv_tmp = CuMemoryManager::GetInstance()->GetData("tmp2", size_zv);

    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv, dim_xv, levels_.x, 'x', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.y, 'y', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.z, 'z', bc_, type_w_curl_, true);

    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv, dim_yv, levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.y, 'y', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.z, 'z', bc_, type_w_curl_, true);

    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv, dim_zv, levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.y, 'y', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.z, 'z', bc_div_, type_w_div_, true);

    //CudaPrintfMat(xv_tmp, dim_xv);

    cudlp_.ProjectLocal_q_ccc(qx, qy, qz, xv_tmp, yv_tmp, zv_tmp, levels_);
    //CudaPrintfMat(qx, dim_qx);
    //CudaPrintfMat(qy, dim_qy);

    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.x, 'x', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.y, 'y', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.z, 'z', bc_, type_w_curl_, false);
}

void CuWaveletPotentialRecover3D::Solve(DTYPE* qx, DTYPE* qy, DTYPE* qz, DTYPE* xv_out, DTYPE* yv_out, DTYPE* zv_out, 
    DTYPE* xv, DTYPE* yv, DTYPE* zv)
{
    int3 dim_xv = dim_ + make_int3(1, 0, 0);
    int3 dim_yv = dim_ + make_int3(0, 1, 0);
    int3 dim_zv = dim_ + make_int3(0, 0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);
    int3 dim_qx = { dim_.x, dim_.y + 1, dim_.z + 1 };
    int3 dim_qy = { dim_.x + 1, dim_.y, dim_.z + 1 };
    int3 dim_qz = { dim_.x + 1, dim_.y + 1, dim_.z };

    DTYPE* xv_tmp = xv_out;
    DTYPE* yv_tmp = yv_out;
    DTYPE* zv_tmp = zv_out;

    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv, dim_xv, levels_.x, 'x', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.y, 'y', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.z, 'z', bc_, type_w_curl_, true);

    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv, dim_yv, levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.y, 'y', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.z, 'z', bc_, type_w_curl_, true);

    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv, dim_zv, levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.y, 'y', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.z, 'z', bc_div_, type_w_div_, true);

    //CudaPrintfMat(xv_tmp, dim_xv);

    cudlp_.ProjectLocal_q_ccc(qx, qy, qz, xv_tmp, yv_tmp, zv_tmp, xv_tmp, yv_tmp, zv_tmp, levels_);
    //CudaPrintfMat(qx, dim_qx);
    //CudaPrintfMat(qy, dim_qy);

    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.x, 'x', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx, qx, dim_qx, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.y, 'y', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy, qy, dim_qy, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz, qz, dim_qz, levels_.z, 'z', bc_, type_w_curl_, false);

    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.y, 'y', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, dim_xv, levels_.z, 'z', bc_, type_w_curl_, false);

    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.x, 'x', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, dim_yv, levels_.z, 'z', bc_, type_w_curl_, false);

    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.x, 'x', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.y, 'y', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(zv_tmp, zv_tmp, dim_zv, levels_.z, 'z', bc_div_, type_w_div_, false);
}

__global__ void cuCutCoefs(DTYPE* qx_out, DTYPE* qy_out, DTYPE* qz_out, int3 dim, int3 levels, 
    int level_cut, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - idx_z * dim.x;

        int3 dim_qx = dim - make_int3(1, 0, 0);
        int3 dim_qy = dim - make_int3(0, 1, 0);
        int3 dim_qz = dim - make_int3(0, 0, 1);

        int slice_qz = dim_qz.x * dim_qz.y;
        int slice_qx = dim_qx.x * dim_qx.y;
        int slice_qy = dim_qy.x * dim_qy.y;

        int idx_qx = INDEX3D(idx_y, idx_x - 1, idx_z, dim_qx);
        int idx_qy = INDEX3D(idx_y - 1, idx_x, idx_z, dim_qy);
        int idx_qz = INDEX3D(idx_y, idx_x, idx_z - 1, dim_qz);

        int jy = levels.y - (32 - __clz(max(idx_y - 1, 0)));
        int jx = levels.x - (32 - __clz(max(idx_x - 1, 0)));
        int jz = levels.z - (32 - __clz(max(idx_z - 1, 0)));

        if (jx < level_cut)
        {
            qx_out[idx_qx] = 0.f;
           

            if (idx_y > 0)
            {
                qy_out[idx_qy] = 0.f;
            }

            if (idx_z > 0)
            {
                qz_out[idx_qz] = 0.f;
            }
        }

        if (jy < level_cut)
        {
            qy_out[idx_qy] = 0.f;

            if (idx_x > 0)
            {
                qx_out[idx_qx] = 0.f;
            }

            if (idx_z > 0)
            {
                qz_out[idx_qz] = 0.f;
            }
        }

        if (jz < level_cut)
        {
            qz_out[idx_qz] = 0.f;

            if (idx_x > 0)
            {
                qx_out[idx_qx] = 0.f;
            }

            if (idx_y > 0)
            {
                qy_out[idx_qy] = 0.f;
            }
        }
    }
}

void CuWaveletPotentialRecover3D::CutCoefficients(DTYPE* qx_out, DTYPE* qy_out, DTYPE* qz_out, DTYPE* qx, DTYPE* qy, DTYPE* qz, int level)
{
    int size = getsize(dim_);
    int3 dim_qx = dim_ + make_int3(0, 1, 1);
    int3 dim_qy = dim_ + make_int3(1, 0, 1);
    int3 dim_qz = dim_ + make_int3(1, 1, 0);
    //checkCudaErrors(cudaMemcpy(qx_out, qx, sizeof(DTYPE) * getsize(dim_qx), cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemcpy(qy_out, qy, sizeof(DTYPE) * getsize(dim_qy), cudaMemcpyDeviceToDevice));
    //checkCudaErrors(cudaMemcpy(qz_out, qz, sizeof(DTYPE) * getsize(dim_qz), cudaMemcpyDeviceToDevice));

    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx, dim_qx, levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx_out, dim_qx, levels_.y, 'y', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx_out, dim_qx, levels_.z, 'z', bc_div_, type_w_div_, true);

    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy, dim_qy, levels_.x, 'x', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy_out, dim_qy, levels_.y, 'y', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy_out, dim_qy, levels_.z, 'z', bc_div_, type_w_div_, true);

    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz, dim_qz, levels_.x, 'x', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz_out, dim_qz, levels_.y, 'y', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz_out, dim_qz, levels_.z, 'z', bc_, type_w_curl_, true);

    int size_ext = getsize(dim_ext_);
    cuCutCoefs << <BLOCKS(size_ext), THREADS(size_ext) >> > (qx_out, qy_out, qz_out, dim_ext_, levels_, level, size_ext);

    //CudaPrintfMat(qx_out, dim_qx);
    //CudaPrintfMat(qy_out, dim_qy);
    //CudaPrintfMat(qz_out, dim_qz);

    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx_out, dim_qx, levels_.x, 'x', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx_out, dim_qx, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qx_out, qx_out, dim_qx, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy_out, dim_qy, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy_out, dim_qy, levels_.y, 'y', bc_, type_w_curl_, false);
    CuWHHDForward::GetInstance()->Solve1D(qy_out, qy_out, dim_qy, levels_.z, 'z', bc_div_, type_w_div_, false);

    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz_out, dim_qz, levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz_out, dim_qz, levels_.y, 'y', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz_out, qz_out, dim_qz, levels_.z, 'z', bc_, type_w_curl_, false);
}