#ifndef __VELOCITIESINIT_H__
#define __VELOCITIESINIT_H__

#include "Utils.h"
#include <random>

constexpr DTYPE cf_fraq_global = 3.f;

__device__ __host__ inline DTYPE CurlVelX_global(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    //return  (cos_y * sin_z - sin_y * cos_z) * sin_x;
    return  cos_y * sin_x - cos_z * sin_x;
}

__device__ __host__ inline DTYPE CurlVelY_global(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    //return  (sin_x * cos_z - cos_x * sin_z) * sin_y;
    return  cos_z * sin_y - cos_x * sin_y;
}

__device__ __host__ inline DTYPE CurlVelZ_global(DTYPE sin_x, DTYPE sin_y, DTYPE sin_z, DTYPE cos_x, DTYPE cos_y, DTYPE cos_z)
{
    //return (cos_x * sin_y - sin_x * cos_y) * sin_z;
    return cos_x * sin_z - cos_y * sin_z;
}

inline void GetInitVelCurl_global(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE dx, char bc, char type)
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
            //for (int i = 0; i < size_xv; i++) xv[i] = 0.f;
            //for (int i = 0; i < size_yv; i++) yv[i] = 0.f;
            //for (int i = 0; i < size_zv; i++) zv[i] = 0.4f;
        }
    }
    else
    {
        if (bc == 'n')
        {
            //for (int k = 0; k < dim_xv.z; k++)
            //    for (int j = 1; j < dim_xv.x - 1; j++)
            //        for (int i = 0; i < dim_xv.y; i++)
            //        {
            //            int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
            //            DTYPE3 pos = make_DTYPE3(j, i + 0.5f, k + 0.5f) * dx[0];
            //            xv[idx] = (2.f * pos.x - 1.f) * ( pos.y * pos.y - pos.y) * (pos.z * pos.z - pos.z) + (2.f * pos.z - 1.f) * (pos.x * pos.x - pos.x)
            //                + (2.f * pos.y - 1.f) * (pos.x * pos.x - pos.x);
            //        }

            //for (int k = 0; k < dim_yv.z; k++)
            //    for (int j = 0; j < dim_yv.x; j++)
            //        for (int i = 1; i < dim_yv.y - 1; i++)
            //        {
            //            int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
            //            DTYPE3 pos = make_DTYPE3(j + 0.5f, i, k + 0.5f) * dx[0];
            //            yv[idx] = (2.f * pos.y - 1.f) * (pos.x * pos.x - pos.x) * (pos.z * pos.z - pos.z) - (2.f * pos.z - 1.f) * (pos.y * pos.y - pos.y)
            //                - (2.f * pos.x - 1.f) * (pos.y * pos.y - pos.y);
            //        }

            //for (int k = 1; k < dim_zv.z - 1; k++)
            //    for (int j = 0; j < dim_zv.x; j++)
            //        for (int i = 0; i < dim_zv.y; i++)
            //        {
            //            int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
            //            DTYPE3 pos = make_DTYPE3(j + 0.5f, i + 0.5f, k) * dx[0];
            //            zv[idx] = (2.f * pos.z - 1.f) * (pos.y * pos.y - pos.y) * (pos.x * pos.x - pos.x) + (2.f * pos.y - 1.f) * (pos.z * pos.z - pos.z) 
            //                - (2.f * pos.x - 1.f) * (pos.z * pos.z - pos.z);
            //        }

            DTYPE cf_pi = cf_fraq_global * _M_PI<DTYPE>;
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
                        xv[idx] = CurlVelX_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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
                        yv[idx] = CurlVelY_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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
                        zv[idx] = CurlVelZ_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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

inline void InitVel_global(DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE dx, char bc, char type)
{
    int3 dim_xv = dim + make_int3(1, 0, 0);
    int3 dim_yv = dim + make_int3(0, 1, 0);
    int3 dim_zv = dim + make_int3(0, 0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);
    memset(zv, 0, sizeof(DTYPE) * size_zv);

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
            //for (int i = 0; i < size_xv; i++) xv[i] = 0.f;
            //for (int i = 0; i < size_yv; i++) yv[i] = 0.f;
            //for (int i = 0; i < size_zv; i++) zv[i] = 0.4f;
        }
    }
    else
    {
        if (bc == 'n')
        {
            //for (int k = 0; k < dim_xv.z; k++)
            //    for (int j = 1; j < dim_xv.x - 1; j++)
            //        for (int i = 0; i < dim_xv.y; i++)
            //        {
            //            int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
            //            DTYPE3 pos = make_DTYPE3(j, i + 0.5f, k + 0.5f) * dx[0];
            //            xv[idx] = (2.f * pos.x - 1.f) * ( pos.y * pos.y - pos.y) * (pos.z * pos.z - pos.z) + (2.f * pos.z - 1.f) * (pos.x * pos.x - pos.x)
            //                + (2.f * pos.y - 1.f) * (pos.x * pos.x - pos.x);
            //        }

            //for (int k = 0; k < dim_yv.z; k++)
            //    for (int j = 0; j < dim_yv.x; j++)
            //        for (int i = 1; i < dim_yv.y - 1; i++)
            //        {
            //            int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
            //            DTYPE3 pos = make_DTYPE3(j + 0.5f, i, k + 0.5f) * dx[0];
            //            yv[idx] = (2.f * pos.y - 1.f) * (pos.x * pos.x - pos.x) * (pos.z * pos.z - pos.z) - (2.f * pos.z - 1.f) * (pos.y * pos.y - pos.y)
            //                - (2.f * pos.x - 1.f) * (pos.y * pos.y - pos.y);
            //        }

            //for (int k = 1; k < dim_zv.z - 1; k++)
            //    for (int j = 0; j < dim_zv.x; j++)
            //        for (int i = 0; i < dim_zv.y; i++)
            //        {
            //            int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
            //            DTYPE3 pos = make_DTYPE3(j + 0.5f, i + 0.5f, k) * dx[0];
            //            zv[idx] = (2.f * pos.z - 1.f) * (pos.y * pos.y - pos.y) * (pos.x * pos.x - pos.x) + (2.f * pos.y - 1.f) * (pos.z * pos.z - pos.z) 
            //                - (2.f * pos.x - 1.f) * (pos.z * pos.z - pos.z);
            //        }

            DTYPE cf_pi = cf_fraq_global * _M_PI<DTYPE>;
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
                        xv[idx] = sin_x * cos_y * cos_z + CurlVelX_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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
                        yv[idx] = cos_x * sin_y * cos_z + CurlVelY_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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
                        zv[idx] = cos_x * cos_y * sin_z + CurlVelZ_global(sin_x, sin_y, sin_z, cos_x, cos_y, cos_z);
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

inline void GetB_global(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim,
    DTYPE* dx, DTYPE* dy, DTYPE* dz)
{
    int3 dim_xv = { dim.x + 1, dim.y, dim.z };
    int3 dim_yv = { dim.x, dim.y + 1, dim.z };
    int3 dim_zv = { dim.x, dim.y, dim.z + 1 };
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);
    int size_zv = getsize(dim_zv);
    //memset(xv, 0, sizeof(DTYPE) * size_xv);
    //memset(yv, 0, sizeof(DTYPE) * size_yv);
    //memset(zv, 0, sizeof(DTYPE) * size_zv);

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

inline void InitB_global(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc)
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

inline void InitB_global(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az,
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

    InitVel_global(xv, yv, zv, dim, dx[0], bc, type);

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

inline void InitAxyz_global(DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, char type = 'x')
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

//----------------------------------------------2D--------------------------------------------------------------//
inline void InitAxy_global(DTYPE* ax, DTYPE* ay, int2 dim, char type = 'x')
{
    int2 dim_xv = { dim.x + 1, dim.y };
    int2 dim_yv = { dim.x, dim.y + 1 };

    int size = dim.x * dim.y;
    int size_xv = dim_xv.x * dim_xv.y;
    int size_yv = dim_yv.x * dim_yv.y;

    std::default_random_engine dre;
    std::uniform_real_distribution<DTYPE> u(0.f, 1.f);
    switch (type)
    {
    case '1':
        for (int i = 0; i < size_xv; i++) ax[i] = 1.f;
        for (int i = 0; i < size_yv; i++) ay[i] = 1.f;
        break;
    default:
        for (int i = 0; i < size_xv; i++) ax[i] = u(dre);
        for (int i = 0; i < size_yv; i++) ay[i] = u(dre);
    }
}

__device__ __host__ inline DTYPE CurlVelX_global(DTYPE sin_x, DTYPE sin_y, DTYPE cos_x, DTYPE cos_y)
{
    //return  (cos_y * sin_z - sin_y * cos_z) * sin_x;
    return  cos_y * sin_x;
}

__device__ __host__ inline DTYPE CurlVelY_global(DTYPE sin_x, DTYPE sin_y, DTYPE cos_x, DTYPE cos_y)
{
    //return  (sin_x * cos_z - cos_x * sin_z) * sin_y;
    return  -cos_x * sin_y;
}

__device__ __host__ inline double CurlVelX_global_double(double sin_x, double sin_y, double cos_x, double cos_y)
{
    //return  (cos_y * sin_z - sin_y * cos_z) * sin_x;
    return  cos_y * sin_x;
}

__device__ __host__ inline double CurlVelY_global_double(double sin_x, double sin_y, double cos_x, double cos_y)
{
    //return  (sin_x * cos_z - cos_x * sin_z) * sin_y;
    return  -cos_x * sin_y;
}

inline void GetInitVelCurl_global(DTYPE* xv, DTYPE* yv, int2 dim, DTYPE dx, char bc, char type)
{
    int2 dim_xv = dim + make_int2(1, 0);
    int2 dim_yv = dim + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);

    if (type == 'r')
    {
        std::default_random_engine dre;
        dre.seed(45);
        std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
        if (bc == 'n')
        {
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    xv[INDEX2D(i, j, dim_xv)] = u(dre);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    yv[INDEX2D(i, j, dim_yv)] = u(dre);
                }
        }
        else
        {
            for (int i = 0; i < size_xv; i++) xv[i] = u(dre);
            for (int i = 0; i < size_yv; i++) yv[i] = u(dre);
        }
    }
    else
    {
        if (bc == 'n')
        {
            double cf_pi = cf_fraq_global * _M_PI<double>;
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;

                    double2 pos = make_double2(j * dx, (i + 0.5f) * dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);

                    xv[idx] = CurlVelX_global_double(sin_x, sin_y, cos_x, cos_y);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    int idx = i + j * dim_yv.y;
                    double2 pos = make_double2((j + 0.5f) * dx, (i)*dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);
                    yv[idx] = CurlVelY_global_double(sin_x, sin_y, cos_x, cos_y);
                }
        }
        else
        {
            for (int j = 0; j < dim_xv.x; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;
                    xv[idx] = std::sin(j * dx);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 0; i < dim_yv.y; i++)
                {
                    int idx = i + j * dim_yv.y;
                    yv[idx] = std::sin(i * dx);
                }
        }
    }
}

inline void InitVel_global(DTYPE* xv, DTYPE* yv, int2 dim, DTYPE dx, char bc, char type)
{
    int2 dim_xv = dim + make_int2(1, 0);
    int2 dim_yv = dim + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);

    if (type == 'r')
    {
        std::default_random_engine dre;
        dre.seed(45);
        std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
        if (bc == 'n')
        {
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    xv[INDEX2D(i, j, dim_xv)] = u(dre);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    yv[INDEX2D(i, j, dim_yv)] = u(dre);
                }
        }
        else
        {
            for (int i = 0; i < size_xv; i++) xv[i] = u(dre);
            for (int i = 0; i < size_yv; i++) yv[i] = u(dre);
        }
    }
    else
    {
        if (bc == 'n')
        {
            //DTYPE cf_pi = cf_fraq_global * _M_PI<DTYPE>;
            double cf_pi = cf_fraq_global * _M_PI<double>;
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;
                    //DTYPE2 pos = make_DTYPE2(j, i + 0.5f) * dx;
                    //DTYPE sin_x = sin(cf_pi * pos.x);
                    //DTYPE sin_y = sin(cf_pi * pos.y);
                    //DTYPE cos_x = cos(cf_pi * pos.x);
                    //DTYPE cos_y = cos(cf_pi * pos.y);

                    //DTYPE sin_px = sin(cf_pi * 20 * pos.x);
                    //DTYPE sin_py = sin(cf_pi * 20 * pos.y);
                    //DTYPE cos_px = cos(cf_pi * 20 * pos.x);
                    //DTYPE cos_py = cos(cf_pi * 20 * pos.y);
                    //xv[idx] = /*sin_px * cos_py*/ /*(3.f * pos.x * pos.x - pos.x + 1) * (pos.y * pos.y * pos.y - 0.5f * pos.y * pos.y + pos.y)*/
                    //    + CurlVelX_global(sin_x, sin_y, cos_x, cos_y);

                    double2 pos = make_double2(j * dx, (i + 0.5f) * dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);

                    double sin_px = sin(cf_pi * 2000 * pos.x);
                    double sin_py = sin(cf_pi * 2000 * pos.y);
                    double cos_px = cos(cf_pi * 2000 * pos.x);
                    double cos_py = cos(cf_pi * 2000 * pos.y);
                    xv[idx] = /*sin_px * cos_py*/ /*(0.5f * pos.x * pos.x - pos.x + 1) * (pos.y * pos.y * pos.y - 0.5f * pos.y * pos.y + pos.y)*/
                        +CurlVelX_global_double(sin_x, sin_y, cos_x, cos_y);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    int idx = i + j * dim_yv.y;
                    //DTYPE2 pos = make_DTYPE2(j + 0.5f, i) * dx;
                    //DTYPE sin_x = sin(cf_pi * pos.x);
                    //DTYPE sin_y = sin(cf_pi * pos.y);
                    //DTYPE cos_x = cos(cf_pi * pos.x);
                    //DTYPE cos_y = cos(cf_pi * pos.y);

                    //DTYPE sin_px = sin(cf_pi * 20 * pos.x);
                    //DTYPE sin_py = sin(cf_pi * 20 * pos.y);
                    //DTYPE cos_px = cos(cf_pi * 20 * pos.x);
                    //DTYPE cos_py = cos(cf_pi * 20 * pos.y);

                    double2 pos = make_double2((j + 0.5f) * dx, (i) * dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);

                    double sin_px = sin(cf_pi * 2000 * pos.x);
                    double sin_py = sin(cf_pi * 2000 * pos.y);
                    double cos_px = cos(cf_pi * 2000 * pos.x);
                    double cos_py = cos(cf_pi * 2000 * pos.y);
                    yv[idx] = /*cos_px * sin_py*/   /*(3.f * pos.y * pos.y - pos.y + 1) * (pos.x * pos.x * pos.x - 0.5f * pos.x * pos.x + pos.x)*/
                        + CurlVelY_global_double(sin_x, sin_y, cos_x, cos_y);
                }
        }
        else
        {
            for (int j = 0; j < dim_xv.x; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;
                    xv[idx] = std::sin(j * dx);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 0; i < dim_yv.y; i++)
                {
                    int idx = i + j * dim_yv.y;
                    yv[idx] = std::sin(i * dx);
                }
        }
    }
}

inline void InitB_global(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, 
    int2 dim, DTYPE* dx, DTYPE* dy, char bc, char type = 'r')
{
    int2 dim_xv = dim + make_int2(1, 0);
    int2 dim_yv = dim + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);

    InitVel_global(xv, yv, dim, dx[0], bc, type);

    for (int j = 0; j < dim.x; j++)
        for (int i = 0; i < dim.y; i++)
        {
            int idx = i + j * dim.y;

            int idx_xv = i + j * dim_xv.y;
            int idx_yv = i + j * dim_yv.y;

            b[idx] = -((xv[idx_xv + dim_xv.y] * ax[idx_xv + dim_xv.y] - xv[idx_xv] * ax[idx_xv]) / dx[j]
                + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dy[i]);
        }
}

inline void InitVel_global_setseed(DTYPE* xv, DTYPE* yv, int2 dim, DTYPE dx, char bc, char type, int random_seed = 45)
{
    int2 dim_xv = dim + make_int2(1, 0);
    int2 dim_yv = dim + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);

    memset(xv, 0, sizeof(DTYPE) * size_xv);
    memset(yv, 0, sizeof(DTYPE) * size_yv);

    if (type == 'r')
    {
        std::default_random_engine dre;
        dre.seed(random_seed);
        std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);
        if (bc == 'n')
        {
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    xv[INDEX2D(i, j, dim_xv)] = u(dre);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    yv[INDEX2D(i, j, dim_yv)] = u(dre);
                }
        }
        else
        {
            for (int i = 0; i < size_xv; i++) xv[i] = u(dre);
            for (int i = 0; i < size_yv; i++) yv[i] = u(dre);
        }
    }
    else
    {
        if (bc == 'n')
        {
            //DTYPE cf_pi = cf_fraq_global * _M_PI<DTYPE>;
            double cf_pi = cf_fraq_global * _M_PI<double>;
            for (int j = 1; j < dim_xv.x - 1; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;
                    //DTYPE2 pos = make_DTYPE2(j, i + 0.5f) * dx;
                    //DTYPE sin_x = sin(cf_pi * pos.x);
                    //DTYPE sin_y = sin(cf_pi * pos.y);
                    //DTYPE cos_x = cos(cf_pi * pos.x);
                    //DTYPE cos_y = cos(cf_pi * pos.y);

                    //DTYPE sin_px = sin(cf_pi * 20 * pos.x);
                    //DTYPE sin_py = sin(cf_pi * 20 * pos.y);
                    //DTYPE cos_px = cos(cf_pi * 20 * pos.x);
                    //DTYPE cos_py = cos(cf_pi * 20 * pos.y);
                    //xv[idx] = /*sin_px * cos_py*/ /*(3.f * pos.x * pos.x - pos.x + 1) * (pos.y * pos.y * pos.y - 0.5f * pos.y * pos.y + pos.y)*/
                    //    + CurlVelX_global(sin_x, sin_y, cos_x, cos_y);

                    double2 pos = make_double2(j * dx, (i + 0.5f) * dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);

                    double sin_px = sin(cf_pi * 2000 * pos.x);
                    double sin_py = sin(cf_pi * 2000 * pos.y);
                    double cos_px = cos(cf_pi * 2000 * pos.x);
                    double cos_py = cos(cf_pi * 2000 * pos.y);
                    xv[idx] = /*sin_px * cos_py*/ /*(0.5f * pos.x * pos.x - pos.x + 1) * (pos.y * pos.y * pos.y - 0.5f * pos.y * pos.y + pos.y)*/
                        +CurlVelX_global_double(sin_x, sin_y, cos_x, cos_y);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 1; i < dim_yv.y - 1; i++)
                {
                    int idx = i + j * dim_yv.y;
                    //DTYPE2 pos = make_DTYPE2(j + 0.5f, i) * dx;
                    //DTYPE sin_x = sin(cf_pi * pos.x);
                    //DTYPE sin_y = sin(cf_pi * pos.y);
                    //DTYPE cos_x = cos(cf_pi * pos.x);
                    //DTYPE cos_y = cos(cf_pi * pos.y);

                    //DTYPE sin_px = sin(cf_pi * 20 * pos.x);
                    //DTYPE sin_py = sin(cf_pi * 20 * pos.y);
                    //DTYPE cos_px = cos(cf_pi * 20 * pos.x);
                    //DTYPE cos_py = cos(cf_pi * 20 * pos.y);

                    double2 pos = make_double2((j + 0.5f) * dx, (i)*dx);
                    double sin_x = sin(cf_pi * pos.x);
                    double sin_y = sin(cf_pi * pos.y);
                    double cos_x = cos(cf_pi * pos.x);
                    double cos_y = cos(cf_pi * pos.y);

                    double sin_px = sin(cf_pi * 2000 * pos.x);
                    double sin_py = sin(cf_pi * 2000 * pos.y);
                    double cos_px = cos(cf_pi * 2000 * pos.x);
                    double cos_py = cos(cf_pi * 2000 * pos.y);
                    yv[idx] = /*cos_px * sin_py*/   /*(3.f * pos.y * pos.y - pos.y + 1) * (pos.x * pos.x * pos.x - 0.5f * pos.x * pos.x + pos.x)*/
                        +CurlVelY_global_double(sin_x, sin_y, cos_x, cos_y);
                }
        }
        else
        {
            for (int j = 0; j < dim_xv.x; j++)
                for (int i = 0; i < dim_xv.y; i++)
                {
                    int idx = i + j * dim_xv.y;
                    xv[idx] = std::sin(j * dx);
                }

            for (int j = 0; j < dim_yv.x; j++)
                for (int i = 0; i < dim_yv.y; i++)
                {
                    int idx = i + j * dim_yv.y;
                    yv[idx] = std::sin(i * dx);
                }
        }
    }
}

inline void InitB_global_setseed(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay,
    int2 dim, DTYPE* dx, DTYPE* dy, char bc, char type = 'r', int random_seed = 45)
{
    int2 dim_xv = dim + make_int2(1, 0);
    int2 dim_yv = dim + make_int2(0, 1);
    int size_xv = getsize(dim_xv);
    int size_yv = getsize(dim_yv);

    InitVel_global_setseed(xv, yv, dim, dx[0], bc, type, random_seed);

    for (int j = 0; j < dim.x; j++)
        for (int i = 0; i < dim.y; i++)
        {
            int idx = i + j * dim.y;

            int idx_xv = i + j * dim_xv.y;
            int idx_yv = i + j * dim_yv.y;

            b[idx] = -((xv[idx_xv + dim_xv.y] * ax[idx_xv + dim_xv.y] - xv[idx_xv] * ax[idx_xv]) / dx[j]
                + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) / dy[i]);
        }
}
#endif