#ifndef __CUDAMATH_CUH__
#define __CUDAMATH_CUH__

#include "cudaGlobal.cuh"
#include "Utils.h"
#include "cuWHHD_Export.h"

//__device__ int cuIsPosition(int a);
//
//__device__ int cuIsNegative(int a);
//
//__device__ int cuSign(int a);
//
//__device__ int cuAbs(int a);
//
//__device__ int cuIsNotZero(int a);

__device__ inline int cuIsPosition(int a)
{
    return 1 - ((a >> 31) & 1);
}

__device__ inline int cuIsNegative(int a)
{
    return (a >> 31) & 1;
}

__device__ inline int cuSign(int a)
{
    return 1 - (((a >> 31) & 1) << 1);
}

__device__ inline int cuAbs(int a)
{
    int mask = (a >> 31);
    return (mask ^ a) - mask;
}

__device__ inline int cuIsNotZero(int a)
{
    return ((a | (~a + 1)) >> 31) & 1;
}

__device__ inline int clamp_periodic(int idx, int max_dim)
{
    if (idx < 0)
    {
        return max_dim + idx;
    }
    else if (idx >= max_dim)
    {
        return idx - max_dim;
    }
    return idx;
}

template <typename T>
__device__ __forceinline__ bool cuInRange(T val, T l_val, T r_val)
{
    return (val >= l_val) && (val <= r_val);
}

__device__ __host__ __forceinline__ DTYPE my_pow2(DTYPE val)
{
    return val * val;
}

template <typename T> __device__ void inline swap(T& a, T& b)
{
    T c(a); a = b; b = c;
}

//__device__ inline DTYPE clamp(DTYPE val, DTYPE min_val, DTYPE max_val)
//{
//    if (val < min_val)
//    {
//        return min_val;
//    }
//    if (val > max_val)
//    {
//        return max_val;
//    }
//    return val;
//}

__device__ inline void swap_pointer(DTYPE*& ptr1, DTYPE*& ptr2)
{
    DTYPE* temp = ptr1;
    ptr1 = ptr2;
    ptr2 = temp;
}

__device__ inline void cuInverseSymMatrix(DTYPE& a11, DTYPE& a12, DTYPE& a13, DTYPE& a22, DTYPE& a23, DTYPE& a33,
    DTYPE& m11, DTYPE& m12, DTYPE& m13, DTYPE& m22, DTYPE& m23, DTYPE& m33)
{
    a11 = m33 * m22 - m23 * m23;
    a12 = m13 * m23 - m33 * m12;
    a13 = m12 * m23 - m13 * m22;

    a22 = m33 * m11 - m13 * m13;
    a23 = m12 * m13 - m11 * m23;

    a33 = m11 * m22 - m12 * m12;

    DTYPE D_inv = 1.f / ((m11 * a11) + (m12 * a12) + (m13 * a13) + eps<DTYPE>);
    a11 *= D_inv;
    a12 *= D_inv;
    a13 *= D_inv;
    a22 *= D_inv;
    a23 *= D_inv;
    a33 *= D_inv;
}

__device__ inline void cuInverse2x2Matrix(DTYPE& m00, DTYPE& m10, DTYPE& m01, DTYPE& m11)
{
    DTYPE proj_mat_bot_inv = 1.f / (m00 * m11 - m10 * m01 + eps<DTYPE>);
    m00 *= proj_mat_bot_inv;
    m10 *= -proj_mat_bot_inv;
    m01 *= -proj_mat_bot_inv;
    m11 *= proj_mat_bot_inv;
    proj_mat_bot_inv = m00;
    m00 = m11;
    m11 = proj_mat_bot_inv;
}


CUWHHD_API DTYPE CudaArraySum(DTYPE* src, int size);

CUWHHD_API DTYPE CudaArrayAbsSum(DTYPE* src, int size);

CUWHHD_API DTYPE CudaArrayNormSum(DTYPE* src, int size);

CUWHHD_API void CudaMatAdd(DTYPE* dst, DTYPE* src, int size, DTYPE scale = 1.);

CUWHHD_API void CudaMatAdd(DTYPE* dst, DTYPE* lsrc, DTYPE lscale, DTYPE* rsrc, DTYPE rscale, int size);

CUWHHD_API void CudaMatSub(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int size);

CUWHHD_API void CudaMatMul(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int size);

CUWHHD_API void CudaMatAddEle(DTYPE* dst, DTYPE ele, int size);

CUWHHD_API void CudaMatAddEleInv(DTYPE* dst, DTYPE ele, int size);

CUWHHD_API void CudaMatAbs(DTYPE* dst, DTYPE* scale, int size);

CUWHHD_API void CudaMatNorm(DTYPE* dst, DTYPE* scale, int size);

CUWHHD_API void CudaMatScale(DTYPE* dst, DTYPE scale, int size);

CUWHHD_API void CudaMatScale(DTYPE* dst, DTYPE* scale, int size);

CUWHHD_API void CudaMatScaleInv(DTYPE* dst, DTYPE* scale, int size);

CUWHHD_API void CudaGetPGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int colN, int rowN, char boundaryCondition);

CUWHHD_API void CudaGetQGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int colN, int rowN);

//void CudaAddPGradient(DTYPE *p, DTYPE *xV, DTYPE *yV, int colN, int rowN, char boundaryCondition, DTYPE scale = 1.0);
//
//void CudaAddQGradient(DTYPE *p, DTYPE *xV, DTYPE *yV, int colN, int rowN, DTYPE scale = 1.0);

CUWHHD_API void CudaExpandCentraToX(DTYPE* dst, DTYPE* src, int* size);

CUWHHD_API void CudaExpandCentraToY(DTYPE* dst, DTYPE* src, int* size);

CUWHHD_API DTYPE CudaTolerance(DTYPE* p, DTYPE* delP, int colN, int rowN);

CUWHHD_API DTYPE CudaAbsSum(DTYPE* res, int colN, int rowN);

CUWHHD_API void CudaCopy(DTYPE* dst, DTYPE* src, int size);

CUWHHD_API void CudaCopyByJ(DTYPE* dst, DTYPE* src, int* j, int* pDims, char direction);

CUWHHD_API void CudaSetZero(DTYPE* dst, int size);

CUWHHD_API void CudaSetZero(int* dst, int size);

CUWHHD_API void CudaSetValue(DTYPE* dst, int size, DTYPE ele);

CUWHHD_API void CudaSetValue(int* dst, int size, int ele);

CUWHHD_API void CudaMultiplySquare(DTYPE* dst, DTYPE* left, DTYPE* right, int dim);

CUWHHD_API void CudaMatrixTranspose(DTYPE* dst, DTYPE* src, int2 dim);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, int2 dim, DTYPE* dx, DTYPE* dy);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz);

CUWHHD_API DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx);

CUWHHD_API void CudaSetLastX(DTYPE* dst, int3 dim, DTYPE val);


CUWHHD_API DTYPE CudaFindMaxValue(DTYPE* dst, int size);

CUWHHD_API void CudaCopyValOff(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, int3 offset);

CUWHHD_API void CudaOverwriteValOff(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, int3 offset);

CUWHHD_API DTYPE CudaCalculateError(DTYPE* dst, DTYPE* src, int3 dim_dst);

#endif