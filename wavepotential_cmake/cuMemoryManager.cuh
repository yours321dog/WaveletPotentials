#ifndef __CUMEMORYMANAGER_CUH__
#define __CUMEMORYMANAGER_CUH__

#include <map>
#include "cudaGlobal.cuh"

class CuMemoryManager
{
public:
	CuMemoryManager();
	~CuMemoryManager();
	DTYPE* GetData(std::string name, int size);
	void GenerateData(std::string name, int size);
	void GenerateData(std::string name, int3 dim);
	void FreeData(std::string name);
	static CuMemoryManager* GetInstance();

private:
	std::map<std::string, std::pair<int, DTYPE*>> datas_;
	static std::auto_ptr<CuMemoryManager> instance_;
};

#endif