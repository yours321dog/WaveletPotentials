#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>
#include <fstream>
//#include "vector_types.h"
#include "HelperUtil.h"

#include "DoubleUseDefine.h"

#ifdef USE_DOUBLE //define data type
#define DTYPE double
#define DTYPE2 double2
#define DTYPE3 double3
#define DTYPE4 double4
#define make_DTYPE2 make_double2
#define make_DTYPE3 make_double3
#else
#define DTYPE float
#define DTYPE2 float2
#define DTYPE3 float3
#define DTYPE4 float4
#define make_DTYPE2 make_float2
#define make_DTYPE3 make_float3
#endif
#define C_DTYPE const DTYPE
#define C_INT   const int
#define C_CHAR  const char 
#define C_UINT  const unsigned int

//#define MAX_DATA_BLOCK 268435456       //1024 * 1024 * 256
#define MAX_DATA_BLOCK 1048576      //1024 * 1024
//#define UINT    unsigned int

//#define USING_MAXCLIP

#define SSTR(x) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define INDEX2D(i, j, dim) ((i) + (j) * (dim.y))
#define INDEX3D(i, j, k, dim) ((i) + (j) * (dim.y) + (k) * (dim.x) * (dim.y))

template<class T>
constexpr T sqrt2 = T(1.41421356237309504880);

template<class T>
constexpr T eps = T(1e-12L);


template<class T>
constexpr T _1_3 = T(0.33333333333333333333L);

template<class T>
constexpr T _M_PI = T(3.14159265358979323846L);

static const long long gi_pow2[] = {
    1, 2, 4, 8,
    16, 32, 64, 128,
    256, 512, 1024, 2048,
    4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728,
    268435456, 536870912, 1073741824, 2147483648,
    4294967296
};

void PrintMat(C_DTYPE* mat, int3 dim);
void PrintMat(C_DTYPE* mat, int3 dim_dst, int3 dim);

void PrintMat(C_DTYPE* mat, int2 dim);

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int size);

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

DTYPE CalculateErrorL2(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

DTYPE CalculateErrorMax(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

DTYPE CalculateErrorMaxL2(DTYPE2* value1, DTYPE2* value2, int2 size3);

DTYPE CalculateErrorMaxL2(DTYPE3* value1, DTYPE3* value2, int3 size3);

DTYPE CalculateErrorRelativeMax(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

std::pair<DTYPE, DTYPE> CalculateErrorPair(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

std::pair<DTYPE, DTYPE> CalculateErrorL2Pair(C_DTYPE* value1, C_DTYPE* value2, int3 size3);

DTYPE CalculateError(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3);

DTYPE CalculateErrorL2(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3);

DTYPE CalculateErrorMax(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3);

DTYPE CalculateErrorRelativeMax(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3);

DTYPE CalculateErrorL2(C_DTYPE* value1, C_DTYPE* value2, int2 size2);

std::pair<DTYPE, DTYPE> CalculateErrorL2Pair(C_DTYPE* value1, C_DTYPE* value2, int2 size2);

DTYPE CalculateErrorMax(C_DTYPE* value1, C_DTYPE* value2, int2 size2);

DTYPE CalculateErrorRelativeMax(C_DTYPE* value1, C_DTYPE* value2, int2 size2);

DTYPE CalculateErrorL2(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2);

DTYPE CalculateErrorMax(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2);

DTYPE CalculateErrorRelativeMax(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2);

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int3 dim, int3 dim_dst);

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, C_INT* size1, C_INT* size2, int extent = 0);

DTYPE CalculateErrorWithPrint(C_DTYPE* value1, C_DTYPE* value2, int size);

DTYPE CalculateErrorWithPrint(C_DTYPE* value1, C_DTYPE* value2, C_INT* size1, C_INT* size2, int extent = 0);

DTYPE CalculateErrorWithoutMean(C_DTYPE* value1, C_DTYPE* value2, int size);

DTYPE CalculateErrorNP2(C_DTYPE* value, C_DTYPE* np2Value, C_INT* pDims, C_INT* np2Dims, char direction, int extent = 0);

void SwapPointer(void** ptr1, void** ptr2);

bool fileExists(const std::string& filename);

void Array2Ddda(DTYPE* vel_ddda, DTYPE* vel, int level, int3 dim, char direction, int is_ext);
void Array2Ddda3D(DTYPE* vel_ddda, DTYPE* vel, int3 levels, int3 dim);

void ReadCsv(std::vector<DTYPE>& res, std::ifstream& fp, int* res_dim = nullptr);
void ReadCsv3D(std::vector<DTYPE>& res, std::ifstream& fp, int* res_dim = nullptr);

void WriteCsvYLine(std::ofstream& fp, int* data, int3 dim);
void WriteDataTxt(std::ofstream& fp, DTYPE* data, int3 dim);
void WriteDataTxt(const char* fp_name, DTYPE* data, int3 dim);
void WriteDataTxt_nonewline(const char* fp_name, DTYPE* data, int3 dim);
void WriteDataTxt_sparse(const char* fp_name, DTYPE* data, int3 dim);
void WriteVectorTxt_sparse(const char* fp_name, DTYPE3* data, int3 dim);
void WriteDataETxt(const char* fp_name, DTYPE* data, int3 dim);
void ReadsDataTxt(const char* fp_name, DTYPE* data, int3 dim);

void WriteDataTxt_sparse_2d(const char* fp_name, DTYPE* data, int2 dim);

void WriteDataBinary(const char* fp_name, DTYPE* data, int3 dim);

constexpr inline unsigned int switch_pair(int a, int b)
{
    return (a << 16) + b;
}

//random num range is min to max, template function
template <typename T> T cRandom(T min, T len)
{
    //srand(time(0));
    return (min + static_cast<T>(len * rand() / static_cast<T>(RAND_MAX + 1)));
}

// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x);

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x);


inline void log_times_errs(const char* fname, std::vector<double> times, std::vector<double> errs)
{
    std::ofstream f(fname);
    f << "iters, error, time" << std::endl;
    for (int i = 0; i < times.size(); i++)
    {
        f << i << ", " << errs[i] << ", " << times[i] << std::endl;
    }
}


inline void log_times_errs_L2_Linf(const char* fname, std::vector<double> times, std::vector<double> errs, std::vector<double> errs_inf)
{
    std::ofstream f(fname);
    f << "iters, error, time, error_inf" << std::endl;
    for (int i = 0; i < times.size(); i++)
    {
        f << i << ", " << errs[i] << ", " << times[i] << ", " << errs_inf[i] << std::endl;
    }
}

inline void log_times_errs(const char* fname_time, const char* fname_err, std::vector<std::vector<double>>& times, std::vector<std::vector<double>>& errs)
{
    std::ofstream f_time(fname_time);
    std::ofstream f_err(fname_err);
    for (int idx_dim = 0; idx_dim < times.size(); idx_dim++)
    {
        for (int i = 0; i < times[idx_dim].size() - 1; i++)
        {
            f_time << times[idx_dim][i] << ", ";
            f_err << errs[idx_dim][i] << ", ";
        }
        f_time << times[idx_dim][times[idx_dim].size() - 1] << std::endl;
        f_err << errs[idx_dim][errs[idx_dim].size() - 1] << std::endl;
    }
}


inline void log_times_errs(const char* fname, double times[], double errs[], int size)
{
    std::ofstream f(fname);
    f << "iters, error, time" << std::endl;
    for (int i = 0; i < size; i++)
    {
        f << i << ", " << errs[i] << ", " << times[i] << std::endl;
    }
}

inline void log_particles_vel(const char* fname, std::vector<double> px, std::vector<double> py, DTYPE2* pv, DTYPE2* pv_ana)
{
    std::ofstream f(fname);
    f << "iters,px,py,pvx,pvy,pvanax,pvanay" << std::endl;
    for (int i = 0; i < px.size(); i++)
    {
        f << i << "," << px[i] << "," << py[i] << "," << pv[i].x << "," << pv[i].y << "," << pv_ana[i].x << "," << pv_ana[i].y << std::endl;
    }
}


DTYPE GenerateRandomDtype(DTYPE min, DTYPE max);
#endif