#include "TestLocalProjection.h"
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "cuMemoryManager.cuh"
#include "cuWHHDForward.cuh"
#include "cuGetInnerProduct.cuh"
#include "CuDivLocalProject.cuh"
#include "CuDivLocalProject3D.cuh"
#include "cuGradient.cuh"
#include <random>
#include "TimeCounter.h"
#include "StreamInitial.h"
#include "cudaMath.cuh"
#include "cuWeightedJacobi3D.cuh"
#include "cuWeightedJacobi.cuh"
#include "cuParallelSweep3D.cuh"
#include "cuConvergence.cuh"
#include "cuConvergence.cuh"
//#include "cuVelInterpolationFromQ.cuh"
#include "VelocitiesInit.h"
#include "cuMultigridFrac2D.cuh"
#include "cuWaveletPotentialRecover3D.cuh"
#include "happly.h"
#include "cuWaveletPotentialRecover2D.cuh"
#include "cuDst.cuh"
#include "cuSolidLevelSet3D.cuh"
#include "cuSolidLevelSet2D.cuh"

DTYPE cf_fraq = 3.f;
const int ramdom_size = 20000;

DTYPE CurlVelX(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    return  cos_y * sin_x - cos_z * sin_x;
}

DTYPE CurlVelY(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    return  cos_z * sin_y - cos_x * sin_y;
}

DTYPE CurlVelZ(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    return cos_x * sin_z - cos_y * sin_z;
}

void GetInitVelCurl_lp(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE dx, char bc, char type)
{
    int3 dim_xv = dim + make_int3(1, 0, 0);
    int3 dim_yv = dim + make_int3(0, 1, 0);
    int3 dim_zv = dim + make_int3(0, 0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);

    if (type == 'r')
    {
        std::default_random_engine dre;
        dre.seed(45);
        std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
        if (bc == 'n')
        {
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 1; j < dim_xv.x - 1; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        xv[idx] = u(dre);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 1; i < dim_yv.y - 1; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        yv[idx] = u(dre);
                    }

            for (int k = 1; k < dim_zv.z - 1; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        zv[idx] = u(dre);
                    }
        }
        else
        {
            for (int i = 0; i < size_xv; i++) xv[i] = u(dre);
            for (int i = 0; i < size_yv; i++) yv[i] = u(dre);
            for (int i = 0; i < size_zv; i++) zv[i] = u(dre);
        }
    }
    else
    {
        if (bc == 'n')
        {
            DTYPE cf_pi = cf_fraq * _M_PI<DTYPE>;
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 1; j < dim_xv.x - 1; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        DTYPE3 pos = make_DTYPE3(j, i + 0.5f, k + 0.5f) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        xv[idx] = CurlVelX(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 1; i < dim_yv.y - 1; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        DTYPE3 pos = make_DTYPE3(j + 0.5f, i, k + 0.5f) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        yv[idx] = CurlVelY(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }

            for (int k = 1; k < dim_zv.z - 1; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        DTYPE3 pos = make_DTYPE3(j + 0.5f, i + 0.5f, k) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        zv[idx] = CurlVelZ(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }
        }
        else
        {
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 0; j < dim_xv.x; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        xv[idx] = std::sin(j * dx);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 0; i < dim_yv.y; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        yv[idx] = std::sin(i * dx);
                    }

            for (int k = 0; k < dim_zv.z; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        zv[idx] = std::sin(k * dx);
                    }
        }
    }
}

void InitVel_lp(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE dx, char bc, char type)
{
    int3 dim_xv = dim + make_int3(1, 0, 0);
    int3 dim_yv = dim + make_int3(0, 1, 0);
    int3 dim_zv = dim + make_int3(0, 0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);

    if (type == 'r')
    {
        std::default_random_engine dre;
        dre.seed(45);
        std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
        if (bc == 'n')
        {
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 1; j < dim_xv.x - 1; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        xv[idx] = u(dre);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 1; i < dim_yv.y - 1; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        yv[idx] = u(dre);
                    }

            for (int k = 1; k < dim_zv.z - 1; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        zv[idx] = u(dre);
                    }
        }
        else
        {
            for (int i = 0; i < size_xv; i++) xv[i] = u(dre);
            for (int i = 0; i < size_yv; i++) yv[i] = u(dre);
            for (int i = 0; i < size_zv; i++) zv[i] = u(dre);
        }
    }
    else
    {
        if (bc == 'n')
        {
            DTYPE cf_pi = cf_fraq * _M_PI<DTYPE>;
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 1; j < dim_xv.x - 1; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        DTYPE3 pos = make_DTYPE3(j, i + 0.5f, k + 0.5f) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        xv[idx] = /*sin_x * cos_y * cos_z +*/ CurlVelX(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 1; i < dim_yv.y - 1; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        DTYPE3 pos = make_DTYPE3(j + 0.5f, i, k + 0.5f) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        yv[idx] = /*cos_x * sin_y * cos_z +*/ CurlVelY(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }

            for (int k = 1; k < dim_zv.z - 1; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        DTYPE3 pos = make_DTYPE3(j + 0.5f, i + 0.5f, k) * dx;
                        DTYPE sin_x = sin(cf_pi * pos.x);
                        DTYPE sin_y = sin(cf_pi * pos.y);
                        DTYPE sin_z = sin(cf_pi * pos.z);
                        DTYPE cos_x = cos(cf_pi * pos.x);
                        DTYPE cos_y = cos(cf_pi * pos.y);
                        DTYPE cos_z = cos(cf_pi * pos.z);
                        zv[idx] = /*cos_x * cos_y * sin_z +*/ CurlVelZ(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
                    }
        }
        else
        {
            for (int k = 0; k < dim_xv.z; k++)
                for (int j = 0; j < dim_xv.x; j++)
                    for (int i = 0; i < dim_xv.y; i++)
                    {
                        int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                        xv[idx] = std::sin(j * dx);
                    }

            for (int k = 0; k < dim_yv.z; k++)
                for (int j = 0; j < dim_yv.x; j++)
                    for (int i = 0; i < dim_yv.y; i++)
                    {
                        int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                        yv[idx] = std::sin(i * dx);
                    }

            for (int k = 0; k < dim_zv.z; k++)
                for (int j = 0; j < dim_zv.x; j++)
                    for (int i = 0; i < dim_zv.y; i++)
                    {
                        int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                        zv[idx] = std::sin(k * dx);
                    }
        }
    }
}

void InitB_lp(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc)
{
    int3 dim_xv = { dim.x + 1, dim.y, dim.z };
    int3 dim_yv = { dim.x, dim.y + 1, dim.z };
    int3 dim_zv = { dim.x, dim.y, dim.z + 1 };

    int size = dim.x * dim.y * dim.z;
    int size_xv = dim_xv.x * dim_xv.y * dim_xv.z;
    int size_yv = dim_yv.x * dim_yv.y * dim_yv.z;
    int size_zv = dim_zv.x * dim_zv.y * dim_zv.z;

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);
    memset(zv, 0, sizeof(DTYPE) * size_zv);

    std::default_random_engine dre;
    std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
    if (bc == 'n')
    {
        for (int k = 0; k < dim_xv.z; k++)
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                    xv[idx] = u(dre);
                }

        for (int k = 0; k < dim_yv.z; k++)
            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                    yv[idx] = u(dre);
                }

        for (int k = 1; k < dim_zv.z - 1; k++)
            for (int j = 0; j < dim_zv.x; j++)
                for (int i = 0; i < dim_zv.y; i++)
                {
                    int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                    zv[idx] = u(dre);
                }

        for (int k = 0; k < dim.z; k++)
            for (int j = 0; j < dim.x; j++)
                for (int i = 0; i < dim.y; i++)
                {
                    int idx = i + j * dim.y + k * dim.x * dim.y;

                    int idx_xv = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                    int idx_yv = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                    int idx_zv = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                    int slice_zv = dim_zv.x * dim_zv.y;

                    b[idx] = -((xv[idx_xv + dim_xv.y] - xv[idx_xv]) / dx[j] + (yv[idx_yv + 1] - yv[idx_yv]) / dy[i]
                        + (zv[idx_zv + slice_zv] - zv[idx_zv]) / dz[k]);
                }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            b[i] = u(dre);
            //printf("%f ", host_c[i]);
        }
    }
}

void InitB_lp(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc, char type = 'r')
{
    int3 dim_xv = { dim.x + 1, dim.y, dim.z };
    int3 dim_yv = { dim.x, dim.y + 1, dim.z };
    int3 dim_zv = { dim.x, dim.y, dim.z + 1 };

    int size = dim.x * dim.y * dim.z;
    int size_xv = dim_xv.x * dim_xv.y * dim_xv.z;
    int size_yv = dim_yv.x * dim_yv.y * dim_yv.z;
    int size_zv = dim_zv.x * dim_zv.y * dim_zv.z;

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);
    memset(zv, 0, sizeof(DTYPE) * size_zv);
    
    InitVel_lp(xv, yv, zv, dim, dx[0], bc, type);

    for (int k = 0; k < dim.z; k++)
        for (int j = 0; j < dim.x; j++)
            for (int i = 0; i < dim.y; i++)
            {
                int idx = i + j * dim.y + k * dim.x * dim.y;

                int idx_xv = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
                int idx_yv = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
                int idx_zv = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
                int slice_zv = dim_zv.x * dim_zv.y;

                b[idx] = -((xv[idx_xv + dim_xv.y] * ax[idx_xv + dim_xv.y] - xv[idx_xv] * ax[idx_xv]) / dx[j]
                    + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dy[i]
                    + (zv[idx_zv + slice_zv] * az[idx_zv + slice_zv] - zv[idx_zv] * az[idx_zv]) / dz[k]);
            }
}

void InitAxyz_lp(DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, char type = 'x')
{
    int3 dim_xv = { dim.x + 1, dim.y, dim.z };
    int3 dim_yv = { dim.x, dim.y + 1, dim.z };
    int3 dim_zv = { dim.x, dim.y, dim.z + 1 };

    int size = dim.x * dim.y * dim.z;
    int size_xv = dim_xv.x * dim_xv.y * dim_xv.z;
    int size_yv = dim_yv.x * dim_yv.y * dim_yv.z;
    int size_zv = dim_zv.x * dim_zv.y * dim_zv.z;

    std::default_random_engine dre;
    std::uniform_real_distribution<DTYPE> u(0.f, 1.f);
    switch (type)
    {
    case '1':
        for (int i = 0; i < size_xv; i++) ax[i] = 1.f;
        for (int i = 0; i < size_yv; i++) ay[i] = 1.f;
        for (int i = 0; i < size_zv; i++) az[i] = 1.f;
        break;
    default:
        for (int i = 0; i < size_xv; i++) ax[i] = u(dre);
        for (int i = 0; i < size_yv; i++) ay[i] = u(dre);
        for (int i = 0; i < size_zv; i++) az[i] = u(dre);
    }
}

int TestWHHDForward1D()
{
    std::ifstream input_init("Data\\whhdforward1d_init.txt");

    if (input_init.fail())
    {
        printf("TestWHHDForward1D: Cannot find \"Data\\whhdforward1d_init.txt\"");
        return 1;
    }

    std::string type_x;
    std::string type_y;
    std::string type_z;
    int pDims[3] = { 0 };
    int levels[3] = { 0 };
    char direction;
    bool is_forward;

    input_init >> type_x >> type_y >> type_z >> levels[0] >> levels[1] >> levels[2] >> pDims[0] >> pDims[1]
        >> pDims[2] >> direction >> is_forward;

    std::map<char, int> levels_map;
    levels_map['x'] = levels[0];
    levels_map['X'] = levels[0];
    levels_map['y'] = levels[1];
    levels_map['Y'] = levels[1];
    levels_map['z'] = levels[2];
    levels_map['Z'] = levels[2];

    std::map<char, char> bcs_map;
    bcs_map['x'] = type_x[0];
    bcs_map['X'] = type_x[0];
    bcs_map['y'] = type_y[0];
    bcs_map['Y'] = type_y[0];
    bcs_map['z'] = type_z[0];
    bcs_map['Z'] = type_z[0];

    WaveletType type_in = type_from_string(type_y);
    switch (direction)
    {
    case'x':
    case'X':
        type_in = type_from_string(type_x);
        break;
    case'z':
    case'Z':
        type_in = type_from_string(type_z);
        break;
    default:
        break;
    }
    //std::ifstream input_src("Data\\input.csv");
    printf("%s %s %s %d %d %d %d %d %d %c %d\n", type_x.c_str(), type_y.c_str(), type_z.c_str(), levels[0], levels[1],
        levels[2], pDims[0], pDims[1], pDims[2], direction, is_forward);

    std::ifstream fp_input("Data\\whhdforward1d_input.csv");
    std::vector<DTYPE> input_arr;
    ReadCsv3D(input_arr, fp_input);

    std::ifstream fp_output("Data\\whhdforward1d_output.csv");
    std::vector<DTYPE> output_arr;
    ReadCsv3D(output_arr, fp_output);

    int3 dim = make_int3(pDims[0], pDims[1], pDims[2]);
    int size = dim.x * dim.y * dim.z;

    auto dim_rs_from_dim = [](int dim_in, int level_in)
    {
        return std::min((1 << level_in) + (dim_in & 1), dim_in);
    };
    //int2 dim_rs = make_int2(dim_rs_from_dim(dim.x), dim_rs_from_dim(dim.y));
    int3 dim_whhd = make_int3(dim_rs_from_dim(dim.x, levels[0]), dim_rs_from_dim(dim.y, levels[1]), dim_rs_from_dim(dim.z, levels[2]));

    std::vector<DTYPE> host_r_arr;
    host_r_arr.resize(size);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    DTYPE* dev_src = CuMemoryManager::GetInstance()->GetData("src", size);
    DTYPE* dev_dst = CuMemoryManager::GetInstance()->GetData("dst", size);


    checkCudaErrors(cudaMemcpy(dev_dst, input_arr.data(), size * sizeof(DTYPE),
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(host_r_arr.data(), dev_dst, size * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    DTYPE err = CalculateError(output_arr.data(), host_r_arr.data(), size);
    printf("error: %e\n", err);

    return 0;
}

int TestLocalProjectionQ3D_div_total(char axyz_type)
{
    int3 dim = { 32, 32, 32 };
    DTYPE3 dx = { DTYPE(1) / dim.x, DTYPE(1) / dim.y, DTYPE(1) / dim.z };
    dx.x = std::max(dx.x, std::max(dx.y, dx.z));
    dx.y = dx.x;
    dx.z = dx.x;
    char bc = 'n';
    printf("%d %d %d\n", dim.x, dim.y, dim.z);

    int3 dim_xv = { dim.x + 1, dim.y, dim.z };
    int3 dim_yv = { dim.x, dim.y + 1, dim.z };
    int3 dim_zv = { dim.x, dim.y, dim.z + 1 };

    int3 dim_qx = { dim.x, dim.y + 1, dim.z + 1 };
    int3 dim_qy = { dim.x + 1, dim.y, dim.z + 1 };
    int3 dim_qz = { dim.x + 1, dim.y + 1, dim.z };

    int size = dim.x * dim.y * dim.z;
    int size_xv = dim_xv.x * dim_xv.y * dim_xv.z;
    int size_yv = dim_yv.x * dim_yv.y * dim_yv.z;
    int size_zv = dim_zv.x * dim_zv.y * dim_zv.z;
    int size_qx = getsize(dim_qx);
    int size_qy = getsize(dim_qy);
    int size_qz = getsize(dim_qz);

    std::unique_ptr<DTYPE[]> host_c(new DTYPE[size]);
    std::unique_ptr<DTYPE[]> host_xv(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> host_yv(new DTYPE[size_yv]);
    std::unique_ptr<DTYPE[]> host_zv(new DTYPE[size_zv]);

    std::unique_ptr<DTYPE[]> xv_out(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> yv_out(new DTYPE[size_yv]);
    std::unique_ptr<DTYPE[]> zv_out(new DTYPE[size_zv]);

    std::unique_ptr<DTYPE[]> xv(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> yv(new DTYPE[size_yv]);
    std::unique_ptr<DTYPE[]> zv(new DTYPE[size_zv]);

    std::unique_ptr<DTYPE[]> host_dx(new DTYPE[dim.x]);
    std::unique_ptr<DTYPE[]> host_dy(new DTYPE[dim.y]);
    std::unique_ptr<DTYPE[]> host_dz(new DTYPE[dim.z]);

    std::unique_ptr<DTYPE[]> host_ax(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> host_ay(new DTYPE[size_yv]);
    std::unique_ptr<DTYPE[]> host_az(new DTYPE[size_zv]);

    std::unique_ptr<DTYPE[]> qx_out(new DTYPE[size_qx]);
    std::unique_ptr<DTYPE[]> qy_out(new DTYPE[size_qy]);
    std::unique_ptr<DTYPE[]> qz_out(new DTYPE[size_qz]);

    std::default_random_engine dre;
    std::uniform_real_distribution<DTYPE> u(0.3f, 1.f);
    for (int i = 0; i < dim.x; i++) host_dx[i] = dx.x;
    for (int i = 0; i < dim.y; i++) host_dy[i] = dx.y;
    for (int i = 0; i < dim.z; i++) host_dz[i] = dx.z;
    //for (int i = 0; i < dim.x; i++) host_dx[i] = u(dre);
    //for (int i = 0; i < dim.y; i++) host_dy[i] = u(dre);
    //for (int i = 0; i < dim.z; i++) host_dz[i] = u(dre);
    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));


    DTYPE* dev_src = CuMemoryManager::GetInstance()->GetData("src", size);
    DTYPE* dev_f = CuMemoryManager::GetInstance()->GetData("f", size);
    DTYPE* dev_dst = CuMemoryManager::GetInstance()->GetData("dst", size);
    DTYPE* dev_dx = CuMemoryManager::GetInstance()->GetData("dx", dim.x);
    DTYPE* dev_dy = CuMemoryManager::GetInstance()->GetData("dy", dim.y);
    DTYPE* dev_dz = CuMemoryManager::GetInstance()->GetData("dz", dim.z);
    DTYPE* dev_ax = CuMemoryManager::GetInstance()->GetData("ax", size_xv);
    DTYPE* dev_ay = CuMemoryManager::GetInstance()->GetData("ay", size_yv);
    DTYPE* dev_az = CuMemoryManager::GetInstance()->GetData("az", size_zv);

    DTYPE* dev_xv_cf = CuMemoryManager::GetInstance()->GetData("xv_cf", size_xv);
    DTYPE* dev_yv_cf = CuMemoryManager::GetInstance()->GetData("yv_cf", size_yv);
    DTYPE* dev_zv_cf = CuMemoryManager::GetInstance()->GetData("zv_cf", size_zv);

    DTYPE* dev_xv = CuMemoryManager::GetInstance()->GetData("xv", size_xv);
    DTYPE* dev_yv = CuMemoryManager::GetInstance()->GetData("yv", size_yv);
    DTYPE* dev_zv = CuMemoryManager::GetInstance()->GetData("zv", size_zv);

    DTYPE* dev_xv_lwt = CuMemoryManager::GetInstance()->GetData("xv_lwt", size_xv);
    DTYPE* dev_yv_lwt = CuMemoryManager::GetInstance()->GetData("yv_lwt", size_yv);
    DTYPE* dev_zv_lwt = CuMemoryManager::GetInstance()->GetData("zv_lwt", size_zv);

    DTYPE* dev_xv_o = CuMemoryManager::GetInstance()->GetData("xv_o", size_xv);
    DTYPE* dev_yv_o = CuMemoryManager::GetInstance()->GetData("yv_o", size_yv);
    DTYPE* dev_zv_o = CuMemoryManager::GetInstance()->GetData("zv_o", size_zv);

    DTYPE* dev_qx = CuMemoryManager::GetInstance()->GetData("qx", size_qx);
    DTYPE* dev_qy = CuMemoryManager::GetInstance()->GetData("qy", size_qy);
    DTYPE* dev_qz = CuMemoryManager::GetInstance()->GetData("qz", size_qz);

    if (axyz_type == 's')
    {
        CuSolidLevelSet3D cusls3d(dim, dx, 1.f * dx.x);
        DTYPE3 sphere_mid = make_DTYPE3(0.5f, 0.5f, 0.5f) * make_DTYPE3(dim) * dx;
        DTYPE sphere_radius = 0.245f * dim.x * dx.x;
        cusls3d.InitLsSphere(sphere_mid, sphere_radius);
        cusls3d.GetFrac(dev_ax, dev_ay, dev_az);


        checkCudaErrors(cudaMemcpy(host_ax.get(), dev_ax, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(host_ay.get(), dev_ay, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(host_az.get(), dev_az, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    else
    {
        InitAxyz_lp(host_ax.get(), host_ay.get(), host_az.get(), dim, axyz_type);
    }
    InitB_lp(host_c.get(), host_xv.get(), host_yv.get(), host_zv.get(), host_ax.get(), host_ay.get(), host_az.get(),
        dim, host_dx.get(), host_dy.get(), host_dz.get(), bc);
    printf("\n");

    checkCudaErrors(cudaMemcpy(dev_xv_o, host_xv.get(), size_xv * sizeof(DTYPE),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv_o, host_yv.get(), size_yv * sizeof(DTYPE),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_zv_o, host_zv.get(), size_zv * sizeof(DTYPE),
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_f, host_c.get(), size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_dx, host_dx.get(), dim.x * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_dy, host_dy.get(), dim.y * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_dz, host_dz.get(), dim.z * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ax, host_ax.get(), size_xv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ay, host_ay.get(), size_yv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_az, host_az.get(), size_zv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_xv, host_xv.get(), size_xv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv, host_yv.get(), size_yv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_zv, host_zv.get(), size_zv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(dev_src, 0, size * sizeof(DTYPE)));
    checkCudaErrors(cudaMemset(dev_dst, 0, size * sizeof(DTYPE)));

    int n_iters = 10;
    int after_n_iter = 60000;
    DTYPE weight = 0.8f;

    char bc_mg = bc == 'd' ? 'z' : bc;
    DTYPE3 dx_e = bc == 'd' ? make_DTYPE3(0.f, 0.f, 0.f) : dx;
    CuMultigridFrac3D cumgf3d(dev_ax, dev_ay, dev_az, dim, dx, bc);
    cumgf3d.Solve(dev_dst, nullptr, dev_f, n_iters, 3, 3);

    DTYPE err_wj = CuWeightedJacobi3D::GetFracRhs(dev_dst, dev_f, dev_ax, dev_ay, dev_az, dim, dev_dx, dev_dy, dev_dz, dx_e, bc_mg);
    printf("err_wj: %e\n", err_wj);

    CuGradient::GetInstance()->Gradient3D_Frac_P(dev_xv_cf, dev_yv_cf, dev_zv_cf, dev_dst, dev_ax, dev_ay, dev_az, dim, dev_dx, dev_dy, dev_dz, dx_e, bc_mg);
    CudaMatAdd(dev_xv_o, dev_xv_cf, size_xv, -1.f);
    CudaMatAdd(dev_yv_o, dev_yv_cf, size_yv, -1.f);
    CudaMatAdd(dev_zv_o, dev_zv_cf, size_zv, -1.f);

    // Multiply frac
    CudaMatMul(dev_xv_o, dev_xv_o, dev_ax, size_xv);
    CudaMatMul(dev_yv_o, dev_yv_o, dev_ay, size_yv);
    CudaMatMul(dev_zv_o, dev_zv_o, dev_az, size_zv);

    checkCudaErrors(cudaMemcpy(xv_out.get(), dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yv_out.get(), dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(zv_out.get(), dev_zv_o, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    cumgf3d.Solve(dev_dst, nullptr, dev_f, 50, 5, 5);
    err_wj = CuWeightedJacobi3D::GetFracRhs(dev_dst, dev_f, dev_ax, dev_ay, dev_az, dim, dev_dx, dev_dy, dev_dz, dx_e, bc_mg);
    printf("after err_wj: %e\n", err_wj);

    CuGradient::GetInstance()->Gradient3D_Frac_P(dev_xv_cf, dev_yv_cf, dev_zv_cf, dev_dst, dev_ax, dev_ay, dev_az, dim, dev_dx, dev_dy, dev_dz, dx_e, bc_mg);
    CudaMatAdd(dev_xv, dev_xv_cf, size_xv, -1.f);
    CudaMatAdd(dev_yv, dev_yv_cf, size_yv, -1.f);
    CudaMatAdd(dev_zv, dev_zv_cf, size_zv, -1.f);

    CuConvergence::GetInstance()->GetB_3d_frac_vel(dev_f, dev_xv, dev_yv, dev_zv, dev_ax, dev_ay, dev_az, dim, dx, bc_mg);
    printf("max_convergence: %e\n", CudaFindMaxValue(dev_f, size));

    // Multiply frac
    CudaMatMul(dev_xv, dev_xv, dev_ax, size_xv);
    CudaMatMul(dev_yv, dev_yv, dev_ay, size_yv);
    CudaMatMul(dev_zv, dev_zv, dev_az, size_zv);

    checkCudaErrors(cudaMemcpy(xv.get(), dev_xv, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yv.get(), dev_yv, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(zv.get(), dev_zv, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    DTYPE err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("origin xv error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("origin yv error: %e\n", err);
    err = CalculateErrorL2(zv.get(), zv_out.get(), dim_zv);
    printf("origin zv error: %e\n", err);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv,
        zv.get(), zv_out.get(), dim_zv);
    printf("origin total error: %e\n", err);

    ///////8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    CuParallelSweep3D cups3d(dim, dx, bc);
    cups3d.Solve(dev_qx, dev_qy, dev_qz, dev_xv_o, dev_yv_o, dev_zv_o);
    cups3d.Project(dev_qx, dev_qy, dev_qz);
    checkCudaErrors(cudaMemcpy(qx_out.get(), dev_qx, size_qx * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(qy_out.get(), dev_qy, size_qy * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(qz_out.get(), dev_qz, size_qz * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));

    StreamInitial::GetInstance()->GradientStream(xv_out.get(), yv_out.get(), zv_out.get(), qx_out.get(), qy_out.get(), qz_out.get(), dim, dx);

    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("swp+dst xv error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("swp+dst yv error: %e\n", err);
    err = CalculateErrorL2(zv.get(), zv_out.get(), dim_zv);
    printf("swp+dst zv error: %e\n", err);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv,
        zv.get(), zv_out.get(), dim_zv);
    printf("swp+dst total error: %e\n", err);
    ///////8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

    WaveletType type_w_curl = WaveletType::WT_CDF3_7;
    WaveletType type_w_div = WaveletType::WT_CDF4_6;

    char bc_div = proj_bc(bc);
    int3 levels_t = make_int3(log2(dim.x), log2(dim.y), log2(dim.z));
    int3 dim_ext = { dim.x + 1, dim.y + 1, dim.z + 1 };
    CuDivLocalProject3D cudlp(dim_ext, levels_t, dx, type_w_div, bc_div);
    CuWaveletPotentialRecover3D cuwpr3d(dim, dx, bc, type_w_curl, type_w_div);

    checkCudaErrors(cudaMemcpy(dev_xv_lwt, dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv_lwt, dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_zv_lwt, dev_zv_o, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));

    char proj_type = 's';
    checkCudaErrors(cudaMemcpy(dev_xv_lwt, dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv_lwt, dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_zv_lwt, dev_zv_o, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(dev_qx, dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_qy, dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_qz, dev_zv_o, size_zv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    CudaSetValue(dev_qx, size_qx, 0.f);
    CudaSetValue(dev_qy, size_qy, 0.f);
    CudaSetValue(dev_qz, size_qz, 0.f);
    cuwpr3d.Solve(dev_qx, dev_qy, dev_qz, dev_xv_lwt, dev_yv_lwt, dev_zv_lwt);

    checkCudaErrors(cudaMemcpy(qx_out.get(), dev_qx, size_qx * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(qy_out.get(), dev_qy, size_qy * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(qz_out.get(), dev_qz, size_qz * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));

    StreamInitial::GetInstance()->GradientStream(xv_out.get(), yv_out.get(), zv_out.get(), qx_out.get(), qy_out.get(), qz_out.get(), dim, dx);

    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("wave xv error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("wave yv error: %e\n", err);
    err = CalculateErrorL2(zv.get(), zv_out.get(), dim_zv);
    printf("wave zv error: %e\n", err);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv,
        zv.get(), zv_out.get(), dim_zv);
    printf("wave total error: %e\n", err);

    return 0;
}

int TestLocalProjectionQ2D_div_total(char axyz_type)
{
    int dim_1d = 32;
    int2 dim = { dim_1d, dim_1d };
    DTYPE2 dx = { DTYPE(1) / dim.x, DTYPE(1) / dim.y };
    dx.x = std::max(dx.x, dx.y);
    dx.y = dx.x;
    char bc = 'n';
    //std::ifstream input_src("Data\\input.csv");
    printf("%d %d\n", dim.x, dim.y);

    int2 dim_xv = { dim.x + 1, dim.y };
    int2 dim_yv = { dim.x, dim.y + 1 };

    int2 dim_qz = { dim.x + 1, dim.y + 1 };

    int size = dim.x * dim.y;
    int size_xv = dim_xv.x * dim_xv.y;
    int size_yv = dim_yv.x * dim_yv.y;
    int size_qz = getsize(dim_qz);

    std::unique_ptr<DTYPE[]> host_c(new DTYPE[size]);
    std::unique_ptr<DTYPE[]> host_xv(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> host_yv(new DTYPE[size_yv]);

    std::unique_ptr<DTYPE[]> xv_out(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> yv_out(new DTYPE[size_yv]);

    std::unique_ptr<DTYPE[]> xv(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> yv(new DTYPE[size_yv]);

    std::unique_ptr<DTYPE[]> host_dx(new DTYPE[dim.x]);
    std::unique_ptr<DTYPE[]> host_dy(new DTYPE[dim.y]);

    std::unique_ptr<DTYPE[]> host_ax(new DTYPE[size_xv]);
    std::unique_ptr<DTYPE[]> host_ay(new DTYPE[size_yv]);

    std::unique_ptr<DTYPE[]> qz_out(new DTYPE[size_qz]);

    std::default_random_engine dre;
    std::uniform_real_distribution<DTYPE> u(0.3f, 1.f);
    for (int i = 0; i < dim.x; i++) host_dx[i] = dx.x;
    for (int i = 0; i < dim.y; i++) host_dy[i] = dx.y;
    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    DTYPE* dev_src = CuMemoryManager::GetInstance()->GetData("src", size);
    DTYPE* dev_f = CuMemoryManager::GetInstance()->GetData("f", size);
    DTYPE* dev_dst = CuMemoryManager::GetInstance()->GetData("dst", size);
    DTYPE* dev_dx = CuMemoryManager::GetInstance()->GetData("dx", dim.x);
    DTYPE* dev_dy = CuMemoryManager::GetInstance()->GetData("dy", dim.y);
    DTYPE* dev_ax = CuMemoryManager::GetInstance()->GetData("ax", size_xv);
    DTYPE* dev_ay = CuMemoryManager::GetInstance()->GetData("ay", size_yv);

    DTYPE* dev_xv_cf = CuMemoryManager::GetInstance()->GetData("xv_cf", size_xv);
    DTYPE* dev_yv_cf = CuMemoryManager::GetInstance()->GetData("yv_cf", size_yv);

    DTYPE* dev_xv = CuMemoryManager::GetInstance()->GetData("xv", size_xv);
    DTYPE* dev_yv = CuMemoryManager::GetInstance()->GetData("yv", size_yv);

    DTYPE* dev_xv_lwt = CuMemoryManager::GetInstance()->GetData("xv_lwt", size_xv);
    DTYPE* dev_yv_lwt = CuMemoryManager::GetInstance()->GetData("yv_lwt", size_yv);

    DTYPE* dev_xv_o = CuMemoryManager::GetInstance()->GetData("xv_o", size_xv);
    DTYPE* dev_yv_o = CuMemoryManager::GetInstance()->GetData("yv_o", size_yv);

    DTYPE* dev_qz = CuMemoryManager::GetInstance()->GetData("qz", size_qz);

    if (axyz_type == 's')
    {
        CuSolidLevelSet2D cusls3d(dim, dx);

        DTYPE2 sphere_mid = make_DTYPE2(0.5f, 0.5f) * make_DTYPE2(dim) * dx;
        DTYPE sphere_radius = 0.245f * dim.x * dx.x;
        cusls3d.InitLsSphere(sphere_mid, sphere_radius);
        cusls3d.GetFrac(dev_ax, dev_ay);

        checkCudaErrors(cudaMemcpy(host_ax.get(), dev_ax, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(host_ay.get(), dev_ay, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    else
    {
        InitAxy_global(host_ax.get(), host_ay.get(), dim, axyz_type);
    }
    InitB_global(host_c.get(), host_xv.get(), host_yv.get(), host_ax.get(), host_ay.get(),
        dim, host_dx.get(), host_dy.get(), bc);
    printf("\n");

    checkCudaErrors(cudaMemcpy(dev_xv_o, host_xv.get(), size_xv * sizeof(DTYPE),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv_o, host_yv.get(), size_yv * sizeof(DTYPE),
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_f, host_c.get(), size * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_dx, host_dx.get(), dim.x * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_dy, host_dy.get(), dim.y * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ax, host_ax.get(), size_xv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_ay, host_ay.get(), size_yv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_xv, host_xv.get(), size_xv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv, host_yv.get(), size_yv * sizeof(DTYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(dev_src, 0, size * sizeof(DTYPE)));
    checkCudaErrors(cudaMemset(dev_dst, 0, size * sizeof(DTYPE)));

    int n_iters = 50;
    int after_n_iter = 60000;
    DTYPE weight = 0.8f;

    char bc_mg = bc == 'd' ? 'z' : 'n';
    DTYPE2 dx_mg = bc == 'd' ? make_DTYPE2(0., 0.) : dx;
    CuMultigridFrac2D cumgf2d(dev_ax, dev_ay, dim, dx, bc);
    cumgf2d.Solve(dev_dst, nullptr, dev_f, 20, 5, 5);
    //CuWeightedJacobi3D::GetInstance()->SolveFrac(dev_dst, dev_src, dev_f, dev_ax, dev_ay, dev_az, dim, dev_dx, dev_dy, dev_dz, dx, n_iters, weight, bc);
    //CudaPrintfMat(dev_dst, dim);
    
    DTYPE err_wj = CuWeightedJacobi::GetFracRhs(dev_dst, dev_f, dev_ax, dev_ay, dim, dev_dx, dev_dy, dx_mg, bc_mg);
    printf("err_wj: %e\n", err_wj);

    CuGradient::GetInstance()->Gradient2D_Frac_P(dev_xv_cf, dev_yv_cf, dev_dst, dev_ax, dev_ay, dim, dev_dx, dev_dy, dx_mg, bc_mg);
    CudaMatAdd(dev_xv_o, dev_xv_cf, size_xv, -1.f);
    CudaMatAdd(dev_yv_o, dev_yv_cf, size_yv, -1.f);

    // Multiply frac
    CudaMatMul(dev_xv_o, dev_xv_o, dev_ax, size_xv);
    CudaMatMul(dev_yv_o, dev_yv_o, dev_ay, size_yv);

    checkCudaErrors(cudaMemcpy(xv_out.get(), dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yv_out.get(), dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    cumgf2d.Solve(dev_dst, nullptr, dev_f, 15, 5, 5);
    err_wj = CuWeightedJacobi::GetFracRhs(dev_dst, dev_f, dev_ax, dev_ay, dim, dev_dx, dev_dy, dx_mg, bc_mg);
    printf("after err_wj: %e\n", err_wj);

    CuGradient::GetInstance()->Gradient2D_Frac_P(dev_xv_cf, dev_yv_cf, dev_dst, dev_ax, dev_ay, dim, dev_dx, dev_dy, dx_mg, bc_mg);
    CudaMatAdd(dev_xv, dev_xv_cf, size_xv, -1.f);
    CudaMatAdd(dev_yv, dev_yv_cf, size_yv, -1.f);

    CuConvergence::GetInstance()->GetB_2d_frac_vel(dev_f, dev_xv, dev_yv, dev_ax, dev_ay, dim, dx, bc_mg);
    printf("max_convergence: %e\n", CudaFindMaxValue(dev_f, size));

    // Multiply frac
    CudaMatMul(dev_xv, dev_xv, dev_ax, size_xv);
    CudaMatMul(dev_yv, dev_yv, dev_ay, size_yv);

    checkCudaErrors(cudaMemcpy(xv.get(), dev_xv, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yv.get(), dev_yv, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToHost));

    DTYPE err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("error: %e\n", err);
    //PrintMat(yv.get(), dim_yv);
    //PrintMat(yv_out.get(), dim_yv);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv);
    printf("direct total error: %e\n", err);

    ///////8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
    CuParallelSweep3D cups3d(dim, dx, bc);
    //cups3d.Solve(dev_qx, dev_qy, dev_qz, dev_xv, dev_yv, dev_zv);
    cups3d.Solve(dev_qz, dev_xv_o, dev_yv_o);
    checkCudaErrors(cudaMemcpy(qz_out.get(), dev_qz, size_qz * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));

    StreamInitial::GetInstance()->GradientStream(xv_out.get(), yv_out.get(), qz_out.get(), dim, dx);

    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("error: %e\n", err);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv);
    printf("sweep total error: %e\n", err);
    ///////8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

    WaveletType type_w_curl = WaveletType::WT_CDF3_7;
    WaveletType type_w_div = WaveletType::WT_CDF4_6;

    char bc_div = proj_bc(bc);
    int2 levels_t = make_int2(log2(dim.x), log2(dim.y));
    int2 dim_ext = { dim.x + 1, dim.y + 1 };
    CuDivLocalProject cudlp(dim_ext, levels_t, dx, type_w_div, bc_div);

    checkCudaErrors(cudaMemcpy(dev_xv_lwt, dev_xv_o, size_xv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(dev_yv_lwt, dev_yv_o, size_yv * sizeof(DTYPE), cudaMemcpyDeviceToDevice));

    char proj_type = 's';
    CuWHHDForward::GetInstance()->Solve1D(dev_xv_lwt, dev_xv_lwt, make_int3(dim_xv, 1), levels_t.x, 'x', bc_div, type_w_div, true);
    CuWHHDForward::GetInstance()->Solve1D(dev_xv_lwt, dev_xv_lwt, make_int3(dim_xv, 1), levels_t.y, 'y', bc, type_w_curl, true);

    CuWHHDForward::GetInstance()->Solve1D(dev_yv_lwt, dev_yv_lwt, make_int3(dim_yv, 1), levels_t.x, 'x', bc, type_w_curl, true);
    CuWHHDForward::GetInstance()->Solve1D(dev_yv_lwt, dev_yv_lwt, make_int3(dim_yv, 1), levels_t.y, 'y', bc_div, type_w_div, true);

    cudlp.ProjectSingle(dev_qz, dev_xv_lwt, dev_yv_lwt);
    //cudlp.ProjectLocal(dev_qz, dev_xv_lwt, dev_yv_lwt, levels_t);

    CuWHHDForward::GetInstance()->Solve1D(dev_qz, dev_qz, make_int3(dim_qz, 1), levels_t.x, 'x', bc_div, type_w_div, false);
    CuWHHDForward::GetInstance()->Solve1D(dev_qz, dev_qz, make_int3(dim_qz, 1), levels_t.y, 'y', bc_div, type_w_div, false);

    checkCudaErrors(cudaMemcpy(qz_out.get(), dev_qz, size_qz * sizeof(DTYPE),
        cudaMemcpyDeviceToHost));

    StreamInitial::GetInstance()->GradientStream(xv_out.get(), yv_out.get(), qz_out.get(), dim, dx);

    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv);
    printf("error: %e\n", err);
    err = CalculateErrorL2(yv.get(), yv_out.get(), dim_yv);
    printf("error: %e\n", err);
    err = CalculateErrorL2(xv.get(), xv_out.get(), dim_xv,
        yv.get(), yv_out.get(), dim_yv);
    printf("wavelet total error: %e\n", err);

    return 0;
}