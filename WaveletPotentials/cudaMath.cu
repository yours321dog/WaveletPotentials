#include "cudaMath.cuh"
#include "cuMemoryManager.cuh"
#include "cudaGlobal.cuh"
#include <vector>
#include <algorithm>
//__device__ int cuIsPosition(int a)
//{
//    return 1 - ((a >> 31) & 1);
//}
//
//__device__ int cuIsNegative(int a)
//{
//    return (a >> 31) & 1;
//}
//
//__device__ int cuSign(int a)
//{
//    return 1 - (((a >> 31) & 1) << 1);
//}
//
//__device__ int cuAbs(int a)
//{
//    int mask = (a >> 31);
//    return (mask ^ a) - mask;
//}
//
//__device__ int cuIsNotZero(int a)
//{
//    return ((a | (~a + 1)) >> 31) & 1;
//}

static const int warpSize = 32;
static const int blockSize = 1024;

__device__ int sumCommSingleWarp(volatile DTYPE* shArr) {
    int idx = threadIdx.x % warpSize; //the lane index in the warp
    if (idx < 16) {
        shArr[idx] += shArr[idx + 16]; __syncwarp(0xFFFF);
        shArr[idx] += shArr[idx + 8]; __syncwarp(0xFFFF);
        shArr[idx] += shArr[idx + 4]; __syncwarp(0xFFFF);
        shArr[idx] += shArr[idx + 2]; __syncwarp(0xFFFF);
        shArr[idx] += shArr[idx + 1]; 
    }
    return shArr[0];
}

__global__ void sumCommSingleBlockWithWarps(DTYPE* out, const DTYPE* a, int max_size) {
    int idx = threadIdx.x;
    DTYPE sum = 0;
    for (int i = idx; i < max_size; i += blockSize)
        sum += a[i];
    __shared__ DTYPE r[blockSize];
    r[idx] = sum;
    sumCommSingleWarp(&r[idx & ~(warpSize - 1)]);
    __syncthreads();
    if (idx < warpSize) { //first warp only
        r[idx] = idx * warpSize < blockSize ? r[idx * warpSize] : 0;
        sumCommSingleWarp(r);
        if (idx == 0)
            *out = r[0];
    }
}

__device__ int sumAbsCommSingleWarp(volatile DTYPE* shArr) {
    int idx = threadIdx.x % warpSize; //the lane index in the warp
    if (idx < 16) {
        shArr[idx] += abs(shArr[idx + 16]);
        shArr[idx] += abs(shArr[idx + 8]);
        shArr[idx] += abs(shArr[idx + 4]);
        shArr[idx] += abs(shArr[idx + 2]);
        shArr[idx] += abs(shArr[idx + 1]);
    }
    return shArr[0];
}

__global__ void sumAbsCommSingleBlockWithWarps(DTYPE* out, const DTYPE* a, int max_size) {
    int idx = threadIdx.x;
    DTYPE sum = 0;
    for (int i = idx; i < max_size; i += blockSize)
        sum += abs(a[i]);
    __shared__ DTYPE r[blockSize];
    r[idx] = sum;
    sumAbsCommSingleWarp(&r[idx & ~(warpSize - 1)]);
    __syncthreads();
    if (idx < warpSize) { //first warp only
        r[idx] = idx * warpSize < blockSize ? r[idx * warpSize] : 0;
        sumAbsCommSingleWarp(r);
        if (idx == 0)
            *out = r[0];
    }
}

__global__ void sumNormCommSingleBlockWithWarps(DTYPE* out, const DTYPE* a, int max_size) {
    int idx = threadIdx.x;
    DTYPE sum = 0;
    for (int i = idx; i < max_size; i += blockSize)
        sum += a[i] * a[i];
    __shared__ DTYPE r[blockSize];
    r[idx] = sum;
    sumCommSingleWarp(&r[idx & ~(warpSize - 1)]);
    __syncthreads();
    if (idx < warpSize) { //first warp only
        r[idx] = idx * warpSize < blockSize ? r[idx * warpSize] : 0;
        sumCommSingleWarp(r);
        if (idx == 0)
            *out = r[0];
    }
}

__global__ void cuMatAdd(DTYPE* dst, DTYPE* src, int maxSize, DTYPE scale)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] += src[idx] * scale;
    }
}

__global__ void cuMatAddEle(DTYPE* dst, int maxSize, DTYPE ele)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] += ele;
    }
}

__global__ void cuMatAddEleInv(DTYPE* dst, int maxSize, DTYPE ele)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] /= 1 + ele * dst[idx];
    }
}

__global__ void cuMatAdd(DTYPE* dst, DTYPE* lsrc, DTYPE lscale, DTYPE* rsrc, DTYPE rscale, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        dst[idx] = lsrc[idx] * lscale + rsrc[idx] * rscale;
    }
}

__global__ void cuMatSub(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        dst[idx] = lsrc[idx] - rsrc[idx];
    }
}

__global__ void cuMatMul(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        dst[idx] = lsrc[idx] * rsrc[idx];
    }
}

__global__ void cuMatScaleOneEle(DTYPE* dst, DTYPE scale, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] *= scale;
    }
}

__global__ void cuMatScale(DTYPE* dst, DTYPE* scale, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] *= scale[idx];
    }
}

__global__ void cuMatScaleInv(DTYPE* dst, DTYPE* scale, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] /= scale[idx];
    }
}

__global__ void cuMatSetZero(DTYPE* dst, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = 0.;
    }
}

__global__ void cuMatSetZero(int* dst, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = 0;
    }
}

__global__ void cuMatSetValue(DTYPE* dst, int maxSize, DTYPE value)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = value;
    }
}

__global__ void cuMatSetValue(int* dst, int maxSize, int value)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = value;
    }
}

__global__ void cuGetPGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int maxSize, int colN, int rowN, char boundaryCondition)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        //int n = colN < rowN ? colN : rowN;
        DTYPE dxInv = rowN;
        DTYPE dyInv = colN;

        dxInv = dxInv < dyInv ? dxInv : dyInv;
        dyInv = dxInv;

        int idxCol = idx % colN;
        int idxRow = idx / colN;

        if (idxCol > 0)
        {
            yV[idxCol + idxRow * (colN + 1)] = (p[idx] - p[idx - 1]) * dyInv;
        }
        else if (idxCol == 0)
        {
            int yCalIdx = idxRow * (colN + 1);
            if (boundaryCondition == 'n')
            {
                yV[yCalIdx] = 0;
                yV[yCalIdx + colN] = 0;
            }
            else
            {
                yV[yCalIdx] = 2 * p[idxRow * colN] * dyInv;
                yV[yCalIdx + colN] = -2 * p[idxRow * colN + colN - 1] * dyInv;
            }
        }

        if (idxRow > 0)
        {
            xV[idx] = (p[idx] - p[idx - colN]) * dxInv;
        }
        else if (idxRow == 0)
        {
            if (boundaryCondition == 'n')
            {
                xV[idx] = 0;
                xV[idx + maxSize] = 0;
            }
            else
            {
                xV[idx] = 2 * p[idx] * dxInv;
                xV[idx + maxSize] = -2 * p[idx + maxSize - colN] * dxInv;
            }
        }
    }
}

__global__ void cuGetQGradient(DTYPE* q, DTYPE* xV, DTYPE* yV, int maxSize, int colN, int rowN)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int colXN = colN - 1;
        int idxCol = idx % colN;
        int idxRow = idx / colN;

        //int n = colN < rowN ? colN : rowN;
        DTYPE dxInv = rowN - 1;
        DTYPE dyInv = colN - 1;

        dxInv = dxInv < dyInv ? dxInv : dyInv;
        dyInv = dxInv;

        if (idxCol < colXN)
        {
            xV[idxCol + colXN * idxRow] = (q[idx + 1] - q[idx]) * dyInv;
        }

        if (idx < maxSize - colN)
        {
            yV[idx] = -(q[idx + colN] - q[idx]) * dxInv;
        }
    }
}

__global__ void cuAddPGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int maxSize, int colN, int rowN, char boundaryCondition, DTYPE scale)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int idxCol = idx % colN;
        int idxRow = idx / colN;

        if (idxCol > 0)
        {
            yV[idxCol + idxRow * (colN + 1)] += scale * (p[idx] - p[idx - 1]) * colN;
        }
        else if (idxCol == 0)
        {
            int yCalIdx = idxRow * (colN + 1);
            if (boundaryCondition == 'n')
            {
                yV[yCalIdx] = 0;
                yV[yCalIdx + colN] = 0;
            }
            else
            {
                yV[yCalIdx] += scale * 2 * p[idxRow * colN] * colN;
                yV[yCalIdx + colN] += scale * -2 * p[idxRow * colN + colN - 1] * colN;
            }
        }

        if (idxRow > 0)
        {
            xV[idx] += scale * (p[idx] - p[idx - colN]) * rowN;
        }
        else if (idxRow == 0)
        {
            if (boundaryCondition == 'n')
            {
                xV[idx] = 0;
                xV[idx + maxSize] = 0;
            }
            else
            {
                xV[idx] += scale * 2 * p[idx] * rowN;
                xV[idx + maxSize] += scale * -2 * p[idx + maxSize - colN] * rowN;
            }
        }
    }
}

__global__ void cuAddQGradient(DTYPE* q, DTYPE* xV, DTYPE* yV, int maxSize, int colN, int rowN, DTYPE scale)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int colXN = colN - 1;
        int idxCol = idx % colN;
        int idxRow = idx / colN;


        if (idxCol < colXN)
        {
            xV[idxCol + colXN * idxRow] += scale * (q[idx + 1] - q[idx]) * (colN - 1);
        }

        if (idx < maxSize - colN)
        {
            yV[idx] += scale * -(q[idx + colN] - q[idx]) * (rowN - 1);
        }
    }
}

__global__ void cuReduceAbsSum(DTYPE* dst, DTYPE* src, int maxSize, int levelPow2, bool isOdd)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int calIdx = 2 * idx * levelPow2;

        dst[calIdx] = abs(src[calIdx]) + abs(src[calIdx + levelPow2]);
        if (idx == maxSize - 1 && isOdd)
        {
            dst[calIdx] += abs(src[calIdx + 2 * levelPow2]);
        }
    }
}

__global__ void cuMatAdsDiff(DTYPE* dst, DTYPE* src, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = abs(src[idx] - dst[idx]);
    }
}

__global__ void cuMatCopy(DTYPE* dst, DTYPE* src, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = src[idx];
    }
}

__global__ void cuCopyByJ_X(DTYPE* dst, DTYPE* src, int* j, int maxSize, int colN)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int rowIdx = idx / colN;

        //bool isOdd = (j[rowIdx] % 2 == 1);

        int srcCoe = j[rowIdx] % 2;

        //int srcCoe = j[rowIdx] & 1;
        /*int dstCoe = ~j[rowIdx] & 1;*/

        //if (isOdd)
        //{
            /*dst[idx] = j[rowIdx] % 2 == 1 ? src[idx] : dst[idx];*/
        //}

        dst[idx] = srcCoe * src[idx] + (1 - srcCoe) * dst[idx];
        src[idx] = dst[idx];
    }
}

__global__ void cuCopyByJ_Y(DTYPE* dst, DTYPE* src, int* j, int maxSize, int colN)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int colIdx = idx % colN;

        int srcCoe = j[colIdx] & 1;
        /*int dstCoe = ~j[rowIdx] & 1;*/

        //if (isOdd)
        //{
        /*dst[idx] = j[rowIdx] % 2 == 1 ? src[idx] : dst[idx];*/
        //}

        dst[idx] = srcCoe * src[idx] + (1 - srcCoe) * dst[idx];
        src[idx] = dst[idx];
    }
}

__global__ void cuExpandToX(DTYPE* dst, DTYPE* src, int maxSize, int colN, int rowN)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int idxRow = idx / colN;

        if (idxRow > 0 && idxRow < rowN - 1)
        {
            dst[idx] = 0.5 * (src[idx] + src[idx - colN]);
        }
        else if (idxRow == 0)
        {
            dst[idx] = src[idx];
        }
        else
        {
            dst[idx] = src[idx - colN];
        }
    }
}

__global__ void cuExpandToY(DTYPE* dst, DTYPE* src, int maxSize, int colN)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        int idxRow = idx / colN;
        int idxCol = idx - colN * idxRow;

        int idxCen = idxRow * (colN - 1) + idxCol;

        if (idxCol > 0 && idxCol < colN)
        {
            dst[idx] = 0.5 * (src[idxCen] + src[idxCen - 1]);
        }
        else if (idxCol == 0)
        {
            dst[idx] = src[idxCen];
        }
        else
        {
            dst[idx] = src[idxCen - 1];
        }

    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, int2 dim, DTYPE* dx, DTYPE* dy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y ;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        dst[idx] = (xv[idx_xv + dim.y] - xv[idx_xv]) / dx[idx_x] + (yv[idx_yv + 1] - yv[idx_yv]) / dy[idx_y];
    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_x = idx / dim.y;
        int idx_y = idx - idx_x * dim.y;

        int idx_xv = idx_y + idx_x * dim.y;
        int idx_yv = idx_y + idx_x * (dim.y + 1);

        dst[idx] = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) / dx[idx_x]
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dy[idx_y];
    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
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

        dst[idx] = (xv[idx_xv + dim.y] - xv[idx_xv]) / dx[idx_x] + (yv[idx_yv + 1] - yv[idx_yv]) / dy[idx_y]
            + (zv[idx_zv + slice_xy] - zv[idx_zv]) / dz[idx_z];
    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx, int max_size)
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

        dst[idx] = (xv[idx_xv + dim.y] - xv[idx_xv]) / dx.x + (yv[idx_yv + 1] - yv[idx_yv]) / dx.y
            + (zv[idx_zv + slice_xy] - zv[idx_zv]) / dx.z;
    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
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

        dst[idx] = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) / dx[idx_x]
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dy[idx_y]
            + (zv[idx_zv + slice_xy] * az[idx_zv + slice_xy] - zv[idx_zv] * az[idx_zv]) / dz[idx_z];
    }
}

__global__ void cuCalConvergence(DTYPE* dst, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE3 dx, int max_size)
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

        dst[idx] = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) / dx.x
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dx.y
            + (zv[idx_zv + slice_xy] * az[idx_zv + slice_xy] - zv[idx_zv] * az[idx_zv]) / dx.z;
    }
}

__global__ void cuMatAbs(DTYPE* dst, DTYPE* src, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = abs(src[idx]);
    }
}

__global__ void cuMatNorm(DTYPE* dst, DTYPE* src, int maxSize)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < maxSize)
    {
        dst[idx] = src[idx] * src[idx];
    }
}

// Kernel that executes on the CUDA device
/* left: left operand
 * right: right operand
 * res : result array
 * dim: M dimension of MxM matrix
 * Blok_size: defines block size
 *
 * this function divides the matrices to tiles and load those tiles to shared memory
 * After loading to shared memory it function multiplies with the corresponding tile of other matrix
 * After finishing multiplication of 1 row and 1 column by collecting results of different tiles
 * it stores the result in global memory
 * Function has coalesced access to the global memory and prevent bank conflict
 */

#define MPSQ_BLOCK_SIZE 16
__global__ void cuMultiplySquare(DTYPE* res, DTYPE* left, DTYPE* right, int dim) {

    int i, j;
    float temp = 0;

    __shared__ DTYPE Left_shared_t[MPSQ_BLOCK_SIZE][MPSQ_BLOCK_SIZE];
    __shared__ DTYPE Right_shared_t[MPSQ_BLOCK_SIZE][MPSQ_BLOCK_SIZE];

    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * MPSQ_BLOCK_SIZE + threadIdx.x;
        i = tileNUM * MPSQ_BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem

        if (row < dim && j < dim)
            Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
        else
            Left_shared_t[threadIdx.y][threadIdx.x] = 0;
        // Load right[i][j] to shared mem

        if (col < dim && i < dim)
            Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        else
            Right_shared_t[threadIdx.y][threadIdx.x] = 0;
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (int k = 0; k < MPSQ_BLOCK_SIZE; k++) {

            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
            //if (row == 0 && col ==0)
            //    printf("l: %f, r: %f, temp: %f\n", Left_shared_t[threadIdx.y][k], Right_shared_t[k][threadIdx.x], temp);
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    if (row < dim && col < dim)
    {
        res[row * dim + col] = temp;
        //printf("temp: %f\n", temp);
    }
}

#define TILE_DIM 64
__global__ void cuMatrixTranspose(DTYPE* odata, const DTYPE* idata, int2 dim)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += blockDim.y)
    {
        if (y + j < dim.x && x < dim.y)
        {
            tile[(threadIdx.y + j)][threadIdx.x] = idata[(y + j) * dim.y + x];
            //printf("x: %d, y: %d, val: %f\n", x, y, idata[(y + j) * dim.y + x]);
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += blockDim.y)
    {
        if (y + j < dim.y && x < dim.x)
            odata[(y + j) * dim.x + x] = tile[threadIdx.x][(threadIdx.y + j)];
    }
}

__global__ void cuSetLastX(DTYPE* dst, int3 dim, DTYPE val)
{
    int idx_z = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_y = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_z < dim.z && idx_y < dim.y)
    {
        int idx = idx_y + (dim.x - 1) * dim.y + idx_z * dim.x * dim.y;
        dst[idx] = val;
    }
}

__global__ void cuCopyValOff(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int3 offset, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;

        int idx_src = INDEX3D(idx_y + offset.y, idx_x + offset.x, idx_z + offset.z, dim_src);
        dst[idx] = src[idx_src];
    }

}

__global__ void cuOverwriteValOff(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int3 offset, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / dim.x;
        int idx_x = idx_xz - dim.x * idx_z;

        int idx_dst = INDEX3D(idx_y + offset.y, idx_x + offset.x, idx_z + offset.z, dim_dst);
        dst[idx_dst] = src[idx];
    }

}

bool isOdd(int num)
{
    return num % 2 == 1;
}

void CudaMatAdd(DTYPE* dst, DTYPE* src, int size, DTYPE scale)
{
    if (size < NUM_THREADS)
    {
        cuMatAdd << <1, size >> > (dst, src, size, scale);
    }
    else
    {
        cuMatAdd << <BLOCKS(size), NUM_THREADS >> > (dst, src, size, scale);
    }
}

void CudaMatAdd(DTYPE* dst, DTYPE* lsrc, DTYPE lscale, DTYPE* rsrc, DTYPE rscale, int size)
{
    cuMatAdd << <BLOCKS(size), THREADS(size) >> > (dst, lsrc, lscale, rsrc, rscale, size);
}

void CudaMatSub(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int size)
{
    cuMatSub << <BLOCKS(size), THREADS(size) >> > (dst, lsrc, rsrc, size);
}

void CudaMatMul(DTYPE* dst, DTYPE* lsrc, DTYPE* rsrc, int size)
{
    cuMatMul << <BLOCKS(size), THREADS(size) >> > (dst, lsrc, rsrc, size);
}

void CudaMatAddEle(DTYPE* dst, DTYPE ele, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatAddEle << <1, size >> > (dst, size, ele);
    }
    else
    {
        cuMatAddEle << <BLOCKS(size), NUM_THREADS >> > (dst, size, ele);
    }
}

void CudaMatAddEleInv(DTYPE* dst, DTYPE ele, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatAddEleInv << <1, size >> > (dst, size, ele);
    }
    else
    {
        cuMatAddEleInv << <BLOCKS(size), NUM_THREADS >> > (dst, size, ele);
    }
}

void CudaMatAbs(DTYPE* dst, DTYPE* src, int size)
{
    cuMatAbs << <BLOCKS(size), THREADS(size) >> > (dst, src, size);
}

void CudaMatNorm(DTYPE* dst, DTYPE* src, int size)
{
    cuMatNorm << <BLOCKS(size), THREADS(size) >> > (dst, src, size);
}

void CudaMatScale(DTYPE* dst, DTYPE scale, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatScaleOneEle << <1, size >> > (dst, scale, size);
    }
    else
    {
        cuMatScaleOneEle << <BLOCKS(size), NUM_THREADS >> > (dst, scale, size);
    }
}

void CudaMatScale(DTYPE* dst, DTYPE* scale, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatScale << <1, size >> > (dst, scale, size);
    }
    else
    {
        cuMatScale << <BLOCKS(size), NUM_THREADS >> > (dst, scale, size);
    }
}

void CudaMatScaleInv(DTYPE* dst, DTYPE* scale, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatScaleInv << <1, size >> > (dst, scale, size);
    }
    else
    {
        cuMatScaleInv << <BLOCKS(size), NUM_THREADS >> > (dst, scale, size);
    }
}

void CudaGetPGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int colN, int rowN, char boundaryCondition)
{
    int maxSize = colN * rowN;
    if (maxSize < NUM_THREADS)
    {
        cuGetPGradient << <1, maxSize >> > (p, xV, yV, maxSize, colN, rowN, boundaryCondition);
    }
    else
    {
        cuGetPGradient << <BLOCKS(maxSize), NUM_THREADS >> > (p, xV, yV, maxSize, colN, rowN, boundaryCondition);
    }
}

void CudaGetQGradient(DTYPE* q, DTYPE* xV, DTYPE* yV, int colN, int rowN)
{
    int maxSize = colN * rowN;
    if (maxSize < NUM_THREADS)
    {
        cuGetQGradient << <1, maxSize >> > (q, xV, yV, maxSize, colN, rowN);
    }
    else
    {
        cuGetQGradient << <BLOCKS(maxSize), NUM_THREADS >> > (q, xV, yV, maxSize, colN, rowN);
    }
}

void CudaAddPGradient(DTYPE* p, DTYPE* xV, DTYPE* yV, int colN, int rowN, char boundaryCondition, DTYPE scale)
{
    int maxSize = colN * rowN;
    if (maxSize < NUM_THREADS)
    {
        cuGetPGradient << <1, maxSize >> > (p, xV, yV, maxSize, colN, rowN, boundaryCondition);
    }
    else
    {
        cuGetPGradient << <BLOCKS(maxSize), NUM_THREADS >> > (p, xV, yV, maxSize, colN, rowN, boundaryCondition);
    }
}

void CudaAddQGradient(DTYPE* q, DTYPE* xV, DTYPE* yV, int colN, int rowN, DTYPE scale)
{
    int maxSize = colN * rowN;
    if (maxSize < NUM_THREADS)
    {
        cuGetQGradient << <1, maxSize >> > (q, xV, yV, maxSize, colN, rowN);
    }
    else
    {
        cuGetQGradient << <BLOCKS(maxSize), NUM_THREADS >> > (q, xV, yV, maxSize, colN, rowN);
    }
}

void CudaExpandCentraToX(DTYPE* dst, DTYPE* src, int* pDims)
{
    int size = pDims[0] * pDims[1];

    if (size < NUM_THREADS)
    {
        cuExpandToX << <1, size >> > (dst, src, size, pDims[0], pDims[1]);
    }
    else
    {
        cuExpandToX << <BLOCKS(size), NUM_THREADS >> > (dst, src, size, pDims[0], pDims[1]);
    }
}

void CudaExpandCentraToY(DTYPE* dst, DTYPE* src, int* pDims)
{
    int size = pDims[0] * pDims[1];

    if (size < NUM_THREADS)
    {
        cuExpandToY << <1, size >> > (dst, src, size, pDims[0]);
    }
    else
    {
        cuExpandToY << <BLOCKS(size), NUM_THREADS >> > (dst, src, size, pDims[0]);
    }
}

DTYPE CudaTolerance(DTYPE* p, DTYPE* delP, int colN, int rowN)
{
    DTYPE norm_abs_incrm, norm_abs_total;

    //DTYPE *tmpP = new DTYPE[colN * rowN];

    int maxSize = colN * rowN;
    /*if (maxSize < NUM_THREADS)
    {
        cuMatAdsDiff << <1, maxSize >> >(delP, p, maxSize);
    }
    else
    {
        cuMatAdsDiff << <BLOCKS(maxSize), NUM_THREADS >> >(delP, p, maxSize);
    }*/

    //cudaMemcpy(tmpP, delP, sizeof(DTYPE) * colN * rowN, cudaMemcpyDeviceToHost);

    //DTYPE sumP = 0.;
    //for (int i = 0; i < colN * rowN; i++)
    //{
    //    sumP += abs(tmpP[i]);
    //    //printf("%f\t", tmpP[i]);
    //}
    //printf("sumDelP: %f\n", sumP);

    //cudaMemcpy(tmpP, p, sizeof(DTYPE) * colN * rowN, cudaMemcpyDeviceToHost);

    //sumP = 0.;
    //for (int i = 0; i < colN * rowN; i++)
    //{
    //    sumP += abs(tmpP[i]);
    //    //printf("%f\t", tmpP[i]);
    //}
    //printf("sumP: %f\n", sumP);

    norm_abs_incrm = CudaAbsSum(delP, colN, rowN);

    CudaCopy(delP, p, maxSize);

    norm_abs_total = CudaAbsSum(delP, colN, rowN);

    printf("%f\t%f\n", norm_abs_incrm, norm_abs_total);

    //delete[] tmpP;

    return norm_abs_incrm / norm_abs_total;
}

DTYPE CudaAbsSum(DTYPE* res, int colN, int rowN)
{
    int levelPow2 = 1;
    int calSize = colN * rowN / 2;
    bool isCalSizeOdd = isOdd(colN * rowN);
    for (; calSize > 0; calSize /= 2)
    {
        if (calSize < NUM_THREADS)
        {
            cuReduceAbsSum << <1, calSize >> > (res, res, calSize, levelPow2, isCalSizeOdd);
        }
        else
        {
            cuReduceAbsSum << <BLOCKS(calSize), NUM_THREADS >> > (res, res, calSize, levelPow2, isCalSizeOdd);
        }
        levelPow2 *= 2;
        isCalSizeOdd = isOdd(calSize);
    }

    DTYPE sum;
    cudaCheckError(cudaMemcpy(&sum, res, sizeof(DTYPE), cudaMemcpyDeviceToHost));

    return sum;
}

CUWHHD_API void CudaCopy(DTYPE* dst, DTYPE* src, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatCopy << <1, size >> > (dst, src, size);
    }
    else
    {
        cuMatCopy << <BLOCKS(size), NUM_THREADS >> > (dst, src, size);
    }
}

void CudaCopyByJ(DTYPE* dst, DTYPE* src, int* j, int* pDims, char direction)
{
    int size = pDims[0] * pDims[1];

    switch (direction)
    {
    case 'X':
    case 'x':
        if (size < NUM_THREADS)
        {
            cuCopyByJ_X << <1, size >> > (dst, src, j, size, pDims[0]);
        }
        else
        {
            cuCopyByJ_X << <BLOCKS(size), NUM_THREADS >> > (dst, src, j, size, pDims[0]);
        }
        break;
    default:
        if (size < NUM_THREADS)
        {
            cuCopyByJ_Y << <1, size >> > (dst, src, j, size, pDims[0]);
        }
        else
        {
            cuCopyByJ_Y << <BLOCKS(size), NUM_THREADS >> > (dst, src, j, size, pDims[0]);
        }
        break;
    }
}

void CudaSetZero(DTYPE* dst, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatSetZero << <1, size >> > (dst, size);
    }
    else
    {
        cuMatSetZero << <BLOCKS(size), NUM_THREADS >> > (dst, size);
    }
}

void CudaSetZero(int* dst, int size)
{
    if (size < NUM_THREADS)
    {
        cuMatSetZero << <1, size >> > (dst, size);
    }
    else
    {
        cuMatSetZero << <BLOCKS(size), NUM_THREADS >> > (dst, size);
    }
}

void CudaSetValue(DTYPE* dst, int size, DTYPE ele)
{
    if (size < NUM_THREADS)
    {
        cuMatSetValue << <1, size >> > (dst, size, ele);
    }
    else
    {
        cuMatSetValue << <BLOCKS(size), NUM_THREADS >> > (dst, size, ele);
    }

    cudaCheckError(cudaGetLastError());
}

void CudaSetValue(int* dst, int size, int ele)
{
    if (size < NUM_THREADS)
    {
        cuMatSetValue << <1, size >> > (dst, size, ele);
    }
    else
    {
        cuMatSetValue << <BLOCKS(size), NUM_THREADS >> > (dst, size, ele);
    }
    cudaCheckError(cudaGetLastError());
}

void CudaMultiplySquare(DTYPE* dst, DTYPE* left, DTYPE* right, int dim)
{
    dim3 block(1, 1, 1);
    block.x = std::min(MPSQ_BLOCK_SIZE, dim);
    block.y = std::min(MPSQ_BLOCK_SIZE, dim);

    block.x = MPSQ_BLOCK_SIZE;
    block.y = MPSQ_BLOCK_SIZE;
    dim3 grid(1, 1, 1);
    grid.x = std::ceil(DTYPE(dim) / block.x);
    grid.y = std::ceil(DTYPE(dim) / block.y);

    cuMultiplySquare << <grid, block >> > (dst, right, left, dim);
}

void CudaMatrixTranspose(DTYPE* dst, DTYPE* src, int2 dim)
{
    dim3 block(TILE_DIM, NUM_THREADS / TILE_DIM, 1);
    dim3 grid(std::ceil(DTYPE(dim.y) / block.x), std::ceil(DTYPE(dim.x) / block.y), 1);
    cuMatrixTranspose << <grid, block >> > (dst, src, dim);
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, int2 dim, DTYPE* dx, DTYPE* dy)
{
    int max_size = dim.x * dim.y;
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("conv", max_size);

    cuCalConvergence << <BLOCKS(max_size), THREADS(max_size) >> > (dst, xv, yv, dim, dx, dy, max_size);
    //CudaPrintfMat(dst, dim);
    DTYPE dst_sum = CudaArrayAbsSum(dst, max_size);
    return dst_sum / max_size;
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE* dx, DTYPE* dy)
{
    int max_size = dim.x * dim.y;
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("conv", max_size);

    cuCalConvergence << <BLOCKS(max_size), THREADS(max_size) >> > (dst, xv, yv, ax, ay, dim, dx, dy, max_size);
    //CudaPrintfMat(dst, dim);
    DTYPE dst_sum = CudaArrayAbsSum(dst, max_size);
    return dst_sum / max_size;
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx)
{
    return 0.f;
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("conv", max_size);

    cuCalConvergence << <BLOCKS(max_size), THREADS(max_size) >> > (dst, xv, yv, zv, dim, dx, dy, dz, max_size);
    //CudaPrintfMat(dst, dim);
    DTYPE dst_sum = CudaArrayAbsSum(dst, max_size);
    return dst_sum / max_size;
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("conv", max_size);

    cuCalConvergence << <BLOCKS(max_size), THREADS(max_size) >> > (dst, xv, yv, zv, ax, ay, az, dim, dx, dy, dz, max_size);
    //CudaPrintfMat(dst, dim);
    DTYPE dst_sum = CudaArrayAbsSum(dst, max_size);
    return dst_sum / max_size;
}

DTYPE CudaConvergence(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("conv", max_size);

    cuCalConvergence << <BLOCKS(max_size), THREADS(max_size) >> > (dst, xv, yv, zv, ax, ay, az, dim, dx, max_size);
    //CudaPrintfMat(dst, dim);
    DTYPE dst_sum = CudaArrayAbsSum(dst, max_size);
    return dst_sum / max_size;
}

DTYPE CudaArraySum(DTYPE* src, int size)
{
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("arr_sum", size);
    sumCommSingleBlockWithWarps << <1, blockSize >> > (dst, src, size);
    DTYPE res;
    checkCudaErrors(cudaMemcpy(&res, dst, sizeof(DTYPE), cudaMemcpyDeviceToHost));
    return res;
}

DTYPE CudaArrayAbsSum(DTYPE* src, int size)
{
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("arr_sum", size);
    sumAbsCommSingleBlockWithWarps << <1, blockSize >> > (dst, src, size);
    cudaCheckError(cudaGetLastError());
    DTYPE res;
    checkCudaErrors(cudaMemcpy(&res, dst, sizeof(DTYPE), cudaMemcpyDeviceToHost));
    return res;
}

DTYPE CudaArrayNormSum(DTYPE* src, int size)
{
    DTYPE* dst = CuMemoryManager::GetInstance()->GetData("arr_sum", size);
    sumNormCommSingleBlockWithWarps << <1, blockSize >> > (dst, src, size);
    cudaCheckError(cudaGetLastError());
    DTYPE res;
    checkCudaErrors(cudaMemcpy(&res, dst, sizeof(DTYPE), cudaMemcpyDeviceToHost));
    return res;
}

void CudaSetLastX(DTYPE* dst, int3 dim, DTYPE val)
{
    dim3 blocks(MPSQ_BLOCK_SIZE, MPSQ_BLOCK_SIZE, 1);
    dim3 grid;
    grid.x = std::ceil(DTYPE(dim.y) / MPSQ_BLOCK_SIZE);
    grid.y = std::ceil(DTYPE(dim.z) / MPSQ_BLOCK_SIZE);

    cuSetLastX << <grid, blocks >> > (dst, dim, val);
}

DTYPE CudaFindMaxValue(DTYPE* dst, int size)
{
    std::vector<DTYPE> host_val(size, 0.f);
    checkCudaErrors(cudaMemcpy(host_val.data(), dst, sizeof(DTYPE) * size, cudaMemcpyDeviceToHost));

    // 使用 std::max_element 查找最大值的迭代器
    auto max_iter = std::max_element(host_val.begin(), host_val.end());
    auto min_iter = std::min_element(host_val.begin(), host_val.end());


    return std::max(std::abs(*max_iter),std::abs(*min_iter));
}

void CudaCopyValOff(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, int3 offset)
{
    int max_size = getsize(dim_dst);
    cuCopyValOff<<<BLOCKS(max_size), THREADS(max_size)>>>(dst, src, dim_dst, dim_src, offset, max_size);
}

void CudaOverwriteValOff(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim_src, int3 offset)
{
    int max_size = getsize(dim_src);
    cuOverwriteValOff << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim_src, offset, max_size);
}

DTYPE CudaCalculateError(DTYPE* dst, DTYPE* src, int3 dim_dst)
{
    int size = getsize(dim_dst);
    std::unique_ptr<DTYPE[]> host_dst(new DTYPE[size]);
    std::unique_ptr<DTYPE[]> host_src(new DTYPE[size]);
    checkCudaErrors(cudaMemcpy(host_dst.get(), dst, sizeof(DTYPE) * size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(host_src.get(), src, sizeof(DTYPE) * size, cudaMemcpyDeviceToHost));

    return CalculateError(host_dst.get(), host_src.get(), dim_dst);
}