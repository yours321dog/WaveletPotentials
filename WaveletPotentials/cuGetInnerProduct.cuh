#ifndef __CUGETINNERPRODUCT_cuh__
#define __CUGETINNERPRODUCT_cuh__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"
#include <map>

class CuGetInnerProduct
{
public:
    CuGetInnerProduct();
    ~CuGetInnerProduct();
    static CuGetInnerProduct* GetInstance();

    DTYPE* GetIp(std::string str, int dim);
    DTYPE* GenerateIp(std::string str, int dim);

    DTYPE* GenerateProjCoef(int& length, char direction);
    DTYPE* GenerateProjQCoef(int& length, char direction);

    DTYPE* GetIp(WaveletType type, char bc, int dim);
    DTYPE* GenerateIp(WaveletType type, char bc, int dim);

private:
    static std::auto_ptr<CuGetInnerProduct> instance_;
    std::map<std::string, DTYPE*> ips_map_;
};

#endif