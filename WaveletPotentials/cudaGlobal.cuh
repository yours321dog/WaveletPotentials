#ifndef _CUDAGLOBAL_CUH_
#define _CUDAGLOBAL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "Utils.h"

#include "cuWHHD_Export.h"

#include "helper_cuda.h"

#define MAX_THREADS 1024
#define NUM_THREADS 512
#define NUM_THREAD_DIM_2D   32
#define NUM_THREAD_DIM_3D   8
#define THREAD_DIM_2D_16 16

#ifdef USE_DOUBLE
//#define MAX_EXPECT_SM_SIZE 1024 
#define MAX_EXPECT_SM_SIZE  2048
#else
#define MAX_EXPECT_SM_SIZE  4096
#endif // USE_DOUBLE


#define CUDA_SQRT_2 1.41421356

#define NUM_BATCHES 4

//#define PRINTF_THREADS

#define BLOCKS(threads) (((threads) - 1) / NUM_THREADS + 1)
#define BLOCKS_PER_DIM(threads) (((threads) - 1) / NUM_THREAD_DIM_2D + 1)
#define THREADS(threads) (std::min(threads, NUM_THREADS))

#define BLOCKS_256(threads) (((threads) - 1) / 256 + 1)
#define THREADS_256(threads) (std::min(threads, 256))

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }

#define CONFLICT_OFFSET(idx)    ((idx) + ((idx) >> 5))
#define CONFLICT_OFFSET_64(idx)    ((idx) + ((idx) >> 6))
#define CONFLICT_OFFSET_NUM(idx, num) (idx + (idx >> num))

void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true);

void CalculateThreadDim(dim3& threads, dim3& blocks, dim3& pDims, int D = 2);

template<typename T, int D>
void CudaPrintfMat(T* devValue, const int* sizes)
{
    T* tmp;

    if (D == 2)
    {
        tmp = new DTYPE[sizes[0] * sizes[1]];
        cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(DTYPE) * sizes[0] * sizes[1], cudaMemcpyDeviceToHost));
        for (int j = 0; j < sizes[1]; j++)
            //for (int i = 0; i < sizes[0]; i++)
        {
            for (int i = 0; i < sizes[0]; i++)
                //for (int j = 0; j < sizes[1]; j++)
            {
                //hostYV[index] = DTYPE(rand()) / (DTYPE)RAND_MAX;
                //sum += abs(hostYV[index]);
                std::cout << tmp[i + j * sizes[0]] << "\t";
            }
            printf("\n");
        }
        printf("\n");
    }
    else
    {
        tmp = new DTYPE[sizes[0] * sizes[1] * sizes[2]];
        cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(DTYPE) * sizes[0] * sizes[1] * sizes[2], cudaMemcpyDeviceToHost));
        for (int k = 0; k < sizes[2]; k++)
        {
            for (int j = 0; j < sizes[1]; j++)
            {
                for (int i = 0; i < sizes[0]; i++)
                {
                    //hostYV[index] = DTYPE(rand()) / (DTYPE)RAND_MAX;
                    //sum += abs(hostYV[index]);
                    //printf("idx : %d, value:%f\n", i + j * sizes[0] + k * sizes[0] * sizes[1], tmp[i + j * sizes[0] + k * sizes[0] * sizes[1]]);

                    std::cout << "idx: " << i + j * sizes[0] + k * sizes[0] * sizes[1] << ":value: " << tmp[i + j * sizes[0]] << "\t";
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    delete[] tmp;
}

template<typename T>
void CudaPrintfMat(T* devValue, int size)
{
    T* tmp = new T[size];
    cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; i++)
    {
        printf("%f ", (DTYPE)tmp[i]);
    }
    printf("\n");

    delete[] tmp;
}

template<typename T>
void CudaPrintfMat(T* devValue, int2 dim)
{
    T* tmp = new T[dim.x * dim.y];
    cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(T) * dim.x * dim.y, cudaMemcpyDeviceToHost));
    for (int j = 0; j < dim.x; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            if (typeid(T) == typeid(int))
            {
                printf("%d ", tmp[i + j * dim.y]);
            }
            else
            {
                printf("%f ", tmp[i + j * dim.y]);
            }
        }
        printf("\n");
    }
    printf("\n");

    delete[] tmp;
}

template<typename T>
void CudaPrintfMat(T* devValue, int3 dim)
{
    T* tmp = new T[dim.x * dim.y * dim.z];
    cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(T) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                if (typeid(T) == typeid(int))
                {
                    printf("%d ",(tmp[i + j * dim.y + k * dim.x * dim.y]));
                }
                else
                {
                    printf("%f ", static_cast<float>(tmp[i + j * dim.y + k * dim.x * dim.y]));
                }
                //std::cout << tmp[i + j * dim.y + k * dim.x * dim.y] << ' ';
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    delete[] tmp;
}

template<typename T>
void CudaPrintfMat(T* devValue, int3 dim, DTYPE err)
{
    T* tmp = new T[dim.x * dim.y * dim.z];
    cudaCheckError(cudaMemcpy(tmp, devValue, sizeof(T) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                T val = tmp[i + j * dim.y + k * dim.x * dim.y];
                if (abs(val) > err)
                    printf("i, j, k: (%d, %d, %d), %f\n", i, j, k, tmp[i + j * dim.y + k * dim.x * dim.y]);
            }
        }
    }

    delete[] tmp;
}
#endif