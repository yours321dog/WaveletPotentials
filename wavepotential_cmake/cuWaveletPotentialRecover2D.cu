#include "cuWaveletPotentialRecover2D.cuh"
#include "cuWHHDForward.cuh"
#include "CuDivLocalProject3D.cuh"
#include "cuMemoryManager.cuh"

CuWaveletPotentialRecover2D::CuWaveletPotentialRecover2D(int2 dim, DTYPE2 dx, char bc, WaveletType type_w_curl_, WaveletType type_w_div_)
    :dim_(dim), dim_ext_(dim + make_int2(1, 1)), size_(getsize(dim)), size_ext_(getsize(dim_ext_)), dx_(dx), bc_(bc), bc_div_(proj_bc(bc)),
    cudlp_(dim + make_int2(1, 1), make_int2(log2(dim.x), log2(dim.y)), dx, type_w_div_, proj_bc(bc)),
    type_w_curl_(type_w_curl_), type_w_div_(type_w_div_), levels_(make_int2(log2(dim.x), log2(dim.y)))
{

}

CuWaveletPotentialRecover2D::~CuWaveletPotentialRecover2D() = default;

void CuWaveletPotentialRecover2D::Solve(DTYPE * qz, DTYPE * xv, DTYPE * yv)
{
    int2 dim_xv = dim_ + make_int2(1, 0);
    int2 dim_yv = dim_ + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int2 dim_qz = { dim_.x + 1, dim_.y + 1 };

    DTYPE* xv_tmp = CuMemoryManager::GetInstance()->GetData("tmp", size_xv);
    DTYPE* yv_tmp = CuMemoryManager::GetInstance()->GetData("tmp1", size_yv);

    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv, make_int3(dim_xv, 1), levels_.x, 'x', bc_div_, type_w_div_, true);
    CuWHHDForward::GetInstance()->Solve1D(xv_tmp, xv_tmp, make_int3(dim_xv, 1), levels_.y, 'y', bc_, type_w_curl_, true);

    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv, make_int3(dim_yv, 1), levels_.x, 'x', bc_, type_w_curl_, true);
    CuWHHDForward::GetInstance()->Solve1D(yv_tmp, yv_tmp, make_int3(dim_yv, 1), levels_.y, 'y', bc_div_, type_w_div_, true);

    cudlp_.ProjectSingle(qz, xv_tmp, yv_tmp);

    CuWHHDForward::GetInstance()->Solve1D(qz, qz, make_int3(dim_qz, 1), levels_.x, 'x', bc_div_, type_w_div_, false);
    CuWHHDForward::GetInstance()->Solve1D(qz, qz, make_int3(dim_qz, 1), levels_.y, 'y', bc_div_, type_w_div_, false);
}