#include "cuMemoryManager.cuh"

CuMemoryManager::CuMemoryManager()
{}

CuMemoryManager::~CuMemoryManager()
{
	for (auto & x : datas_)
	{
		cudaFree(x.second.second);
		x.second.first = 0;
		x.second.second = nullptr;
	}
}

DTYPE* CuMemoryManager::GetData(std::string name, int size)
{
	if (datas_.find(name) == datas_.end() || datas_[name].first < size)
	{
		GenerateData(name, size);
	}
	
	return datas_[name].second;
}

void CuMemoryManager::FreeData(std::string name)
{
	if (datas_.find(name) != datas_.end())
	{
		cudaFree(datas_[name].second);
		datas_[name].first = 0;
		datas_[name].second = nullptr;
	}
}

void CuMemoryManager::GenerateData(std::string name, int size)
{
	DTYPE* data;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaError_t cudaStatus = cudaMalloc((void**)&data, size * sizeof(DTYPE));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed in GenerateData!");
		return;
	}
	if (datas_.find(name) != datas_.end())
	{
		cudaFree(datas_[name].second);
	}
	datas_[name].first = size;
	datas_[name].second = data;
}

void CuMemoryManager::GenerateData(std::string name, int3 dim)
{
	int size = dim.x * dim.y * dim.z;
	GenerateData(name, size);
}

std::auto_ptr<CuMemoryManager> CuMemoryManager::instance_;

CuMemoryManager* CuMemoryManager::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuMemoryManager>(new CuMemoryManager); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}
