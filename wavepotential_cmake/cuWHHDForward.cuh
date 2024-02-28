#ifndef __CUWHHDFORWARD_CUH__
#define __CUWHHDFORWARD_CUH__

#include "cudaGlobal.cuh"
#include "WaveletTypes.h"
#include <map>
#include <functional>

typedef void (*pwhhd_forward_func)(DTYPE*, DTYPE*, int3, int3, int, int, int, int);

class CuWHHDForward
{
public:
    CuWHHDForward();
    ~CuWHHDForward();

    void Solve2D(DTYPE* dst, DTYPE* src, int3 dim, int3 levels, char2 bc_x = make_char2('n', 'n'),
        WaveletType rsw_type_x = WaveletType::WT_CDF3_5, WaveletType rsw_type_y = WaveletType::WT_CDF3_5,
        bool is_forward = true);
    void Solve1D(DTYPE* dst, DTYPE* src, int3 dim, int level, char direction, char bc = 'n',
        WaveletType rsw_type = WaveletType::WT_CDF3_5, bool is_forward = true);
    void Solve1D(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char direction, char bc = 'n',
        WaveletType rsw_type = WaveletType::WT_CDF3_5, bool is_forward = true);

   static CuWHHDForward* GetInstance();
private:
    void SolveYSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc = 'n',
        WaveletType rsw_type = WaveletType::WT_CDF3_5, bool is_forward = true);
    void SolveXSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc = 'n',
        WaveletType rsw_type = WaveletType::WT_CDF3_5, bool is_forward = true);
    void SolveZSide(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, int level, char bc = 'n',
        WaveletType rsw_type = WaveletType::WT_CDF3_5, bool is_forward = true);

    static std::auto_ptr<CuWHHDForward> instance_;
    std::map<unsigned int, pwhhd_forward_func> whhd_forward_funcs_;
    std::map<WaveletType, int2> whhd_halo_map_;
};

#endif