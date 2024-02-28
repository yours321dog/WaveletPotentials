#ifndef __STREAMINITIAL_H__
#define __STREAMINITIAL_H__

#include "Utils.h"

class StreamInitial
{
public:
    StreamInitial() = default;
    ~StreamInitial() = default;
    void InitialRandStream(DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, char bc);
    void GradientStream(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, DTYPE3 dx);
    void GradientStream(DTYPE* xv, DTYPE* yv, DTYPE* qz, int2 dim, DTYPE2 dx);

    static StreamInitial* GetInstance();

private:
    static std::auto_ptr<StreamInitial> instance_;
};

#endif