#ifndef __CUBOUNDSELECTOR_CUH__
#define __CUBOUNDSELECTOR_CUH__

#include "cudaGlobal.cuh"

class CuBoundSelector
{
public:
    CuBoundSelector() = default;
    ~CuBoundSelector() = default;

    static void SelectByFrac(int* is_bound, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim);
    static void SelectAxyzByLevels(DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, int3 dim, DTYPE select_width);
    static void SelectToIndex(int* index, int* select_sum, int size);
    static void SelectFromWholeDomain(int* is_select, DTYPE* value, int size);
    static void ValueSelect(DTYPE* value, int* is_select, int size);
};

#endif