#include "cudaGlobal.cuh"

void cudaAssert(cudaError_t code, const char* file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            system("pause");
            exit(code);
        }
    }
}

void CalculateThreadDim(dim3& threads, dim3& blocks, dim3& pDims, int D)
{
    int totalThreads = pDims.x * pDims.y * pDims.z;
    int tmpThreads[3] = { pDims.x, pDims.y, pDims.z };
    int tmpBlocks[3] = { 1, 1, 1 };
    int extraBlocks[3] = { tmpThreads[0] > 1 ? tmpThreads[0] % 2 : 0,
        tmpThreads[1] > 1 ? tmpThreads[1] % 2 : 0,
        tmpThreads[2] > 1 ? tmpThreads[2] % 2 : 0
    };


    int i = 2;
    while (totalThreads > NUM_THREADS)
    {
        int maxIndex = tmpThreads[0] >= tmpThreads[1] ? 0 : 1;
        maxIndex = tmpThreads[2] > tmpThreads[maxIndex] ? 2 : maxIndex;

        tmpThreads[maxIndex] /= 2;
        totalThreads /= 2;
        tmpBlocks[maxIndex] *= 2;
    }

    blocks.x = tmpBlocks[0] + extraBlocks[0];
    blocks.y = tmpBlocks[1] + extraBlocks[1];
    blocks.z = tmpBlocks[2] + extraBlocks[2];

    threads.x = tmpThreads[0];
    threads.y = tmpThreads[1];
    threads.z = tmpThreads[2];
}
