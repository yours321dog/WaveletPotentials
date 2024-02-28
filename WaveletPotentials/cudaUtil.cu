#include "cudaUtil.cuh"

#include "cudaGlobal.cuh"
#include "cuMemoryManager.cuh"

void ShowCudaVal(DTYPE* val, int size)
{
    //DTYPE* dev_val = CuMemoryManager::GetInstance()->GetData("cudaUtil", size);
    std::unique_ptr<DTYPE[]> data(new DTYPE[size]);
    checkCudaErrors(cudaMemcpy(data.get(), val, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; i++)
    {
        printf("%f ", data[i]);
    }
    printf("\n");
}