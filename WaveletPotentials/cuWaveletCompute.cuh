#ifndef __WAVELETCOMPUTE_CUH__
#define __WAVELETCOMPUTE_CUH__

#include "cudaGlobal.cuh"

__constant__ DTYPE RSW_T1_fp[1] = { -1.0 };
__constant__ DTYPE RSW_T1_fu[1] = { 0.5 };
__constant__ DTYPE RSW_T1_scl[2] = { 1.414213562373095048801, 1. / 1.41421356237309504 };
__constant__ DTYPE RSW_T7_fu[7] = { -0.002441406250000,   0.021484375000000, -0.098144531250000,
    0.500000000000000,   0.098144531250000, -0.021484375000000,   0.002441406250000 };
__constant__ DTYPE RSW_T7_scl[2] = { 1.414213562373095048801, -1. / 1.41421356237309504 };

__constant__ DTYPE RSW_Q5_fu_1[1] = { -1. / 3 };
__constant__ DTYPE RSW_Q5_fp[2] = { -3. / 8, -9. / 8 };
__constant__ DTYPE RSW_Q5_fu_2[5] = { 5. / 288, -34. / 288, 128. / 288, 34. / 288, -5. / 288 };
__constant__ DTYPE RSW_Q7_fu_2[7] = { -0.003797743055556, 0.032552083333333, -0.137044270833333,  
    0.444444444444444, 0.137044270833333, -0.032552083333333, 0.003797743055556 };
__constant__ DTYPE RSW_Q9_fu_2[9] = { 0.000854492187500, -0.008924696180556, 0.044514973958333, -0.149007161458334,
    0.444444444444445, 0.149007161458334, -0.044514973958333, 0.008924696180556, -0.000854492187500 };
__constant__ DTYPE RSW_Q11_fu_2[11] = { -0.000195821126299, 0.002421061197873, -0.014211866590456, 0.053914388019860,
    - 0.157231648760173, 0.444444444436392, 0.157231648759860, -0.053914388019746, 0.014211866590415,
    - 0.002421061197862, 0.000195821126297 };
__constant__ DTYPE RSW_Q5_scl[2] = { 3. / 1.414213562373095048801, 1.414213562373095048801 / 3. };

__constant__ DTYPE RSW_QN5_L1[8] = {
   0.632812500000000,   0.484375000000000,   0.015625000000000, -0.132812500000000,
  -0.132812500000000,   0.015625000000000,   0.484375000000000,   0.632812500000000
};
__constant__ DTYPE RSW_QD5_L1[8] = {
   0.734375000000000,   0.882812500000000, -0.117187500000000, -0.265625000000000,
  -0.265625000000000, -0.117187500000000,   0.882812500000000,   0.734375000000000
};

__constant__ DTYPE RSW_QN5_L10[16] = {
    0.500000000000000,   0.500000000000000,   0.500000000000000,   0.500000000000000,
  -0.382812500000000, -0.234375000000000,   0.234375000000000,   0.382812500000000,
  -0.353553390593274,   0.530330085889911, -0.176776695296637,                   0,
                    0,   0.176776695296637, -0.530330085889911,   0.353553390593274
};
__constant__ DTYPE RSW_iQN5_L10[16] = {
   0.500000000000000, -1.000000000000000, -1.038563084867742,   0.375650477505353,
   0.500000000000000, -0.500000000000000,   1.248485410532498, -0.165728151840597,
   0.500000000000000,   0.500000000000000,   0.165728151840597, -1.248485410532498,
   0.500000000000000,   1.000000000000000, -0.375650477505353,   1.038563084867742
};

__constant__ DTYPE PGW_iQN5_L1[8] = {
    0.500000000000000,                   0,
    0.375000000000000,   0.125000000000000,
    0.125000000000000,   0.375000000000000,
                    0,   0.500000000000000
};
__constant__ DTYPE PGW_iQD5_L1[8] = {
   0.250000000000000,                   0,
   0.375000000000000,   0.125000000000000,
   0.125000000000000,   0.375000000000000,
                   0,   0.250000000000000
};

__constant__ DTYPE RSW_QD5_L10[16] = {
    0.937499999999999,   1.531250000000000,   1.531250000000000,   0.937500000000000,
  -1.000000000000000, -1.000000000000000,   1.000000000000000,   1.000000000000000,
  -0.707106781186548,   0.530330085889911, -0.176776695296637,                   0,
                    0,   0.176776695296637, -0.530330085889911,   0.707106781186548
};
__constant__ DTYPE RSW_iQD5_L10[16] = {
   0.125000000000000, -0.250000000000000, -0.894932019939224,   0.187825238752677,
   0.250000000000000, -0.250000000000000,   0.685009694274468,   0.022097086912080,
   0.250000000000000,   0.250000000000000, -0.022097086912080, -0.685009694274468,
   0.125000000000000,   0.250000000000000, -0.187825238752677,   0.894932019939224
};
__constant__ DTYPE RSW_L6_fp[2] = { -0.5, -0.5 };
__constant__ DTYPE RSW_L6_fu[6] = { 0.009765625, -0.076171875, 0.31640625, 0.31640625, -0.076171875, 0.009765625 };
__constant__ DTYPE RSW_L8_fu[8] = { -0.002136230468750, 0.020446777343750, -0.095397949218750, 0.327087402343750,
    0.327087402343750, -0.095397949218750, 0.020446777343750, -0.002136230468750 };
__constant__ DTYPE RSW_L10_fu[10] = { 0.000480651855469, -0.005500793457033, 0.030059814453132, -0.108856201171902,
    0.333816528320394, 0.333816528320394, -0.108856201171902, 0.030059814453132, -0.005500793457033 };
__constant__ DTYPE RSW_L12_fu[12] = { -0.000110149383545, 0.001471996307373, -0.009356021881103, 0.038321018218994,
    -0.118769645690918, 0.338442802429199, 0.338442802429199, -0.118769645690918, 0.038321018218994,
    -0.009356021881103, 0.001471996307373, -0.000110149383545 };
__constant__ DTYPE RSW_L6_scl[2] = { 1.414213562373095048801, 1. / 1.414213562373095048801 };

__constant__ DTYPE RSW_LN6_L1[15] = {
   0.683593750000000,   0.632812500000000, -0.250000000000000, -0.132812500000000,   0.066406250000000,
  -0.125000000000000,   0.250000000000000,   0.750000000000000,   0.250000000000000, -0.125000000000000,
   0.066406250000000, -0.132812500000000, -0.250000000000000,   0.632812500000000,   0.683593750000000
};
__constant__ DTYPE RSW_LD6_L1[15] = {
   1.000000000000000,                   0,                   0,                   0,                   0,
  -0.191406250000000,   0.382812500000000,   0.617187500000000,   0.382812500000000, -0.191406250000000,
                   0,                   0,                   0,                   0,   1.000000000000000
};

__constant__ DTYPE RSW_LN6_L10[25] = {
   0.867187500000000,   1.265625000000000 ,  0.500000000000000, -0.265625000000000 ,-0.367187500000000,
  -0.367187500000000, -0.265625000000000,   0.500000000000000,   1.265625000000000,   0.867187500000000,
  -0.500000000000000,   0.000000000000000,   1.000000000000000,   0.000000000000000, -0.500000000000000,
  -0.353553390593274,   0.707106781186547, -0.353553390593274,                   0,                   0,
                   0 ,                  0, -0.353553390593274 ,  0.707106781186547, -0.353553390593274
};
__constant__ DTYPE RSW_iLN6_L10[25] = {
   0.500000000000000,                   0, - 0.500000000000000, - 0.894932019939224,   0.187825238752677,
   0.375000000000000,   0.125000000000000,                   0,   0.789970857106846, - 0.082864075920299,
   0.250000000000000,   0.250000000000000,   0.500000000000000, - 0.353553390593274, - 0.353553390593274,
   0.125000000000000,   0.375000000000000,                   0, - 0.082864075920299,   0.789970857106846,
                   0,   0.500000000000000, - 0.500000000000000,   0.187825238752677, - 0.894932019939224
};

__constant__ DTYPE RSW_LZ6_L10[25] = {
   0.867187500000000,   0.632812500000000,   0.250000000000000, - 0.132812500000000, - 0.367187500000000,
  - 0.367187500000000, - 0.132812500000000,   0.250000000000000,   0.632812500000000,   0.867187500000000,
  - 1.000000000000000,   0.000000000000000,   1.000000000000000,   0.000000000000000, - 1.000000000000000,
  - 0.707106781186547,   0.707106781186547, - 0.353553390593274,                   0,                   0,
                   0 ,                  0, - 0.353553390593274,   0.707106781186547, - 0.707106781186547
};
__constant__ DTYPE RSW_iLZ6_L10[25] = {
   0.500000000000000,                   0, - 0.250000000000000, - 0.447466009969612,   0.093912619376338,
   0.750000000000000,   0.250000000000000,                   0,   0.789970857106846, - 0.082864075920299,
   0.500000000000000,   0.500000000000000,   0.500000000000000, - 0.353553390593274, - 0.353553390593274,
   0.250000000000000,   0.750000000000000,                   0, - 0.082864075920299,   0.789970857106846,
                   0,   0.500000000000000, - 0.250000000000000,   0.093912619376338, - 0.447466009969612
};

__constant__ DTYPE RSW_LD6_L10[25] = {
   2.000000000000000,   0.000000000000000, -0.000000000000000,   0.000000000000000 ,  0.000000000000000,
  -0.000000000000000, -0.000000000000000,   0.000000000000000, -0.000000000000000,   2.000000000000000,
  -0.691406250000000,   0.382812500000000,   0.617187500000000,   0.382812500000000, -0.691406250000000,
  -0.353553390593274,   0.707106781186547, -0.353553390593274,                   0,                   0,
                   0 ,                  0, -0.353553390593274 ,  0.707106781186547, -0.353553390593274
};
__constant__ DTYPE RSW_iLD6_L10[25] = {
   0.500000000000000,                   0,   0.000000000000000,                   0,                   0,
   0.375000000000000,   0.125000000000000,   0.500000000000000,   1.143524247700120, - 0.270689314672975,
   0.250000000000000,   0.250000000000000,   1.000000000000000, - 0.541378629345950, - 0.541378629345950,
   0.125000000000000,   0.375000000000000,   0.500000000000000, - 0.270689314672975,   1.143524247700120,
                   0,   0.500000000000000, - 0.000000000000000,                   0,                   0
};

__constant__ DTYPE RSW_C4_fu_1[2] = { -1. / 4, -1. / 4 };
__constant__ DTYPE RSW_C4_fp[2] = { -1.0, -1.0 };
__constant__ DTYPE RSW_C4_fu_2[4] = { -0.03906250, 0.22656250, 0.22656250, -0.03906250 };
__constant__ DTYPE RSW_C6_fu_2[6] = { 0.008544921875000, -0.064697265625000, 0.243652343750000, 0.243652343750000, -0.064697265625000, 0.008544921875000 };
__constant__ DTYPE RSW_C8_fu_2[8] = { -0.001922607421884, 0.018157958984455, -0.082000732422244, 0.253265380860537,
    0.253265380860537, -0.082000732422244, 0.018157958984455, -0.001922607421884 };
__constant__ DTYPE RSW_C10_fu_2[10] = { 0.000440597534168, -0.005006790160998, 0.026969909667233, -0.094337463376291,
    0.259433746330612, 0.259433746330612, -0.094337463376291, 0.026969909667233, -0.005006790160998, 0.000440597534168 };
__constant__ DTYPE RSW_C4_scl[2] = { 2. * 1.41421356237309504880168872420969807856967187537694, -1 / (2. * 1.41421356237309504880168872420969807856967187537694) };
__constant__ DTYPE RSW_C4_scl_inv[2] = { 1. / (2. * 1.41421356237309504880168872420969807856967187537694), -(2. * 1.41421356237309504880168872420969807856967187537694) };

__constant__ DTYPE RSW_CN4_L10[25] = {
   2.125000000000000,   1.687500000000000,   0.500000000000000, - 0.687500000000000, - 1.625000000000000,
  - 1.625000000000000, - 0.687500000000000,   0.500000000000000,   1.687500000000000,   2.125000000000000,
   1.000000000000000,                   0, - 2.000000000000000,                   0,   1.000000000000000,
   0.353553390593274, - 0.618718433538229,   0.353553390593274, - 0.088388347648318,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.618718433538229,   0.353553390593274
};
__constant__ DTYPE RSW_iCN4_L10[25] = {
   0.343750000000000,   0.156250000000000,   0.250000000000000,   0.773398041922786, - 0.066291260736239,
   0.312500000000000,   0.187500000000000,   0.000000000000000, - 1.016465997955662,   0.309359216769115,
   0.250000000000000,   0.250000000000000, - 0.250000000000000,   0.353553390593274,   0.353553390593274,
   0.187500000000000,   0.312500000000000,   0.000000000000000,   0.309359216769115, - 1.016465997955662,
   0.156250000000000,   0.343750000000000,   0.250000000000000, - 0.066291260736239,   0.773398041922786
};

__constant__ DTYPE RSW_CD4_L10[25] = {
   2.050781250000000, - 0.000000000000000,   0.000000000000000, - 0.000000000000000,   0.000000000000000,
  - 0.000000000000000,   0.000000000000000, - 0.000000000000000,   0.000000000000000,   2.050781250000000,
   0.734375000000000, - 0.148437500000000, - 0.468750000000000, - 0.148437500000000,   0.734375000000000,
   0.353553390593274, - 0.441941738241592,   0.353553390593274, - 0.088388347648318,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.441941738241592,   0.353553390593274
};
__constant__ DTYPE RSW_iCD4_L10[25] = {
   0.125000000000000,                   0,                   0,                   0,                   0,
   0.250000000000000,   0.125000000000000, - 1.000000000000000, - 2.077126169735483,   0.751300955010707,
   0.218750000000000,   0.218750000000000, - 1.500000000000000,   0.419844651329513,   0.419844651329513,
   0.125000000000000,   0.250000000000000, - 1.000000000000000,   0.751300955010707, - 2.077126169735483,
                   0,   0.125000000000000,                   0,                   0,                   0
};

__device__ __forceinline__ void cuWT_CDF3_5_n(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
    }

    if (d_idx_y < 0)
    {
        d_idx_y = -1 - d_idx_y;
    }
    else if (d_idx_y >= level_dim)
    {
        d_idx_y = 2 * level_dim - 1 - d_idx_y;
    }
    sign_a = 1.f;
    sign_d = 1.f;
}

__device__ __forceinline__ void cuWT_CDF3_5_d(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (d_idx_y < 0)
    {
        d_idx_y = -1 - d_idx_y;
        sign_d = -1.;
    }
    else if (d_idx_y >= level_dim)
    {
        d_idx_y = 2 * level_dim - 1 - d_idx_y;
        //d_val = -src[data_idx_xz + d_idx_y];
        sign_d = -1.f;
    }
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
        sign_a = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
        sign_a = -1.f;
    }
}

__device__ __forceinline__ void cuUpdateCdf35_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val *= sign_d;
    a_val = sign_a * a_val + RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val += RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1) + RSW_Q5_fp[1] * a_val;

    /*printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, d_val);*/

    a_val += RSW_Q5_fu_2[4] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_Q5_fu_2[3] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_Q5_fu_2[2] * d_val;
    a_val += RSW_Q5_fu_2[1] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_Q5_fu_2[0] * __shfl_down_sync(-1, d_val, 2);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);

    a_val *= RSW_Q5_scl[0];
    d_val *= RSW_Q5_scl[1];

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
}

__device__ __forceinline__ void cuWT_CDF2_6_n(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y;
    }
    sign_a = 1.;
    sign_d = 1.;
}

__device__ __forceinline__ void cuWT_CDF2_6_d(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
    }
    else if (a_idx_y > level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
    }

    sign_a = 1.;
    sign_d = 1.;
    if (d_idx_y < 0)
    {
        d_idx_y = -d_idx_y;
        sign_d = -1.;
    }
    else if (d_idx_y >= level_dim)
    {
        d_idx_y = 2 * level_dim - d_idx_y;
        //d_val = -src[data_idx_xz + d_idx_y];
        sign_d = -1.;
    }
}

__device__ __forceinline__ void cuWT_CDF2_6_z(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y;
    }
    sign_a = (a_idx_y == 0 || a_idx_y == level_dim) ? 2.f : 1.f;
    sign_d = 1.f;
}

__device__ __forceinline__ void cuUpdateCdf26_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    a_val *= sign_a;
    d_val += RSW_L6_fp[0] * a_val + RSW_L6_fp[1] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val *= sign_d;

    //printf("a_idx_y: %d\t d_idx_y: %d\t a_val: %d, d_val: %f\n", a_idx_y, d_idx_y, a_val, d_val);
    a_val += RSW_L6_fu[5] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_L6_fu[4] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_L6_fu[3] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_L6_fu[2] * d_val;
    a_val += RSW_L6_fu[1] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_L6_fu[0] * __shfl_down_sync(-1, d_val, 2);

    a_val *= RSW_L6_scl[0];
    d_val *= RSW_L6_scl[1];

    a_val /= sign_a;
}

__device__ __forceinline__ void cuWT_CDF4_4_n(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y;
    }
    sign_a = 1.;
    sign_d = 1.;
}

__device__ __forceinline__ void cuWT_CDF4_4_d(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y;
        sign_d = -1.;
        sign_a = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y;
        sign_d = -1.;
        sign_a = -1.;
    }
    if (a_idx_y == level_dim)
    {
        sign_a = 1.;
    }
}

__device__ __forceinline__ void cuUpdateCdf44_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val *= sign_a;
    d_val *= sign_d;

    a_val += RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
    d_val += RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);

    a_val += RSW_C4_fu_2[3] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_C4_fu_2[2] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_C4_fu_2[1] * d_val;
    a_val += RSW_C4_fu_2[0] * __shfl_down_sync(-1, d_val, 1);

    a_val *= RSW_C4_scl[0];
    d_val *= RSW_C4_scl[1];
}

__device__ __forceinline__ void my_QN35_ad_idx(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
    }

    if (d_idx_y < 0)
    {
        d_idx_y = -1 - d_idx_y;
    }
    else if (d_idx_y >= level_dim)
    {
        d_idx_y = 2 * level_dim - 1 - d_idx_y;
    }
    sign_a = 1.;
    sign_d = 1.;
}

__device__ __forceinline__ void my_QS35_ad_val(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val *= sign_d;
    a_val = sign_a * a_val + RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val += RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1) + RSW_Q5_fp[1] * a_val;

    /*printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, d_val);*/

    a_val += RSW_Q5_fu_2[4] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_Q5_fu_2[3] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_Q5_fu_2[2] * d_val;
    a_val += RSW_Q5_fu_2[1] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_Q5_fu_2[0] * __shfl_down_sync(-1, d_val, 2);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);

    a_val *= RSW_Q5_scl[0];
    d_val *= RSW_Q5_scl[1];

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
}

__device__ __forceinline__ DTYPE cuUpdateL1(DTYPE* src, DTYPE* coef, int start_idx, int idx_off, int coef_idx, int cols)
{
    DTYPE res = 0;
    coef_idx *= cols;
    for (int i = 0; i < cols; i++)
    {
        res += src[start_idx + i * idx_off] * coef[coef_idx + i];
    }
    return res;
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_l1_n(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    return cuUpdateL1(src, RSW_QN5_L1, start_idx, idx_off, coef_idx, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_l1_d(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    return cuUpdateL1(src, RSW_QD5_L1, start_idx, idx_off, coef_idx, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_l1_n(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    return cuUpdateL1(src, RSW_LN6_L1, start_idx, idx_off, coef_idx, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_l1_d(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    return cuUpdateL1(src, RSW_LD6_L1, start_idx, idx_off, coef_idx, 5);
}

__device__ __forceinline__ DTYPE cuUpdateL10(DTYPE a_val, int idx_coef, DTYPE* coefs, int cols)
{
    idx_coef *= cols;
    DTYPE res = 0.f;
#pragma unroll
    for (int i = 0; i < cols; i++)
    {
        res += __shfl_sync(-1, a_val, i) * coefs[idx_coef + i];
    }
    return res;
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QN5_L10, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QD5_L10, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LN6_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf26_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LD6_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf26_l10_z(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LZ6_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf44_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CN4_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf44_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CD4_L10, 5);
}

__device__ __forceinline__ void cuWT_iCDF3_5_n(int& a_idx_y, DTYPE& sign_a, int level_dim)
{
    sign_a = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
    }
}

__device__ __forceinline__ void cuWT_iCDF3_5_d(int& a_idx_y, DTYPE& sign_a, int level_dim)
{
    sign_a = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
        sign_a = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
        sign_a = -1.;
    }
}

__device__ __forceinline__ void cuUpdateiCdf35_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a)
{
    a_val = sign_a * a_val * RSW_Q5_scl[1];

    d_val -= RSW_Q5_fp[1] * a_val + RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1);
    a_val -= RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);
}

__device__ __forceinline__ DTYPE cuiUpdateCdf35_l1_n(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    coef_idx *= 2;
    return src[start_idx] * PGW_iQN5_L1[coef_idx] + src[start_idx + idx_off] * PGW_iQN5_L1[coef_idx + 1];
}

__device__ __forceinline__ DTYPE cuiUpdateCdf35_l1_d(DTYPE* src, int start_idx, int idx_off, int coef_idx)
{
    coef_idx *= 2;
    return src[start_idx] * PGW_iQD5_L1[coef_idx] + src[start_idx + idx_off] * PGW_iQD5_L1[coef_idx + 1];
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQN5_L10, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf35_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQD5_L10, 4);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLN6_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_il10_z(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLZ6_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf26_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLD6_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf44_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCN4_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf44_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCD4_L10, 5);
}

__device__ __forceinline__ void cuWT_CDF3_5_n_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
        sign_d = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
        sign_d = -1.;
    }
    d_idx_y = a_idx_y;
}
__device__ __forceinline__ void cuWT_CDF3_5_d_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -1 - a_idx_y;
        sign_a = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - 1 - a_idx_y;
        sign_a = -1.;
    }
    d_idx_y = a_idx_y;
}
__device__ __forceinline__ void cuUpdateCdf35_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val = sign_a * a_val * RSW_Q5_scl[1];
    d_val = sign_d * d_val * RSW_Q5_scl[0];

    a_val -= RSW_Q5_fu_2[4] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_Q5_fu_2[3] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_Q5_fu_2[2] * d_val;
    a_val -= RSW_Q5_fu_2[1] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_Q5_fu_2[0] * __shfl_down_sync(-1, d_val, 2);

    //printf("ad_idx_y: %d\t a_val: %f, d_val: %f\n", ad_idx_y, a_val, d_val);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, temp);
    d_val -= RSW_Q5_fp[1] * a_val + RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val, d_val: %f, %f\n", a_val, d_val);
    a_val -= RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);
}

__device__ __forceinline__ void cuWT_CDF2_6_n_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y - 1;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y - 1;
    }
}
__device__ __forceinline__ void cuWT_CDF2_6_d_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y - 1;
        sign_d = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y - 1;
        sign_d = -1.;
    }
}
__device__ __forceinline__ void cuWT_CDF2_6_z_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = a_idx_y == 0 || a_idx_y == level_dim ? 2.f : 1.f;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y - 1;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y - 1;
    }
}
__device__ __forceinline__ void cuUpdateCdf26_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_L6_scl[0];
    a_val = sign_a * a_val * RSW_L6_scl[1];

    a_val -= RSW_L6_fu[5] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_L6_fu[4] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_L6_fu[3] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_L6_fu[2] * d_val;
    a_val -= RSW_L6_fu[1] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_L6_fu[0] * __shfl_down_sync(-1, d_val, 2);

    d_val -= RSW_L6_fp[1] * a_val + RSW_L6_fp[0] * __shfl_down_sync(-1, a_val, 1);
    a_val /= sign_a;
}

__device__ __forceinline__ void cuWT_CDF4_4_n_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y - 1;
    }
    else if (a_idx_y >= level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        d_idx_y = 2 * level_dim - d_idx_y - 1;
    }
}
__device__ __forceinline__ void cuWT_CDF4_4_d_i(int& a_idx_y, int& d_idx_y, DTYPE& sign_a, DTYPE& sign_d, int level_dim)
{
    sign_a = 1.;
    sign_d = 1.;
    if (a_idx_y < 0)
    {
        a_idx_y = -a_idx_y;
        d_idx_y = -d_idx_y - 1;
        sign_d = -1.;
        sign_a = -1.;
    }
    else if (a_idx_y >= level_dim)
    {
        d_idx_y = 2 * level_dim - d_idx_y - 1;
        sign_d = -1.;
    }
    if (a_idx_y > level_dim)
    {
        a_idx_y = 2 * level_dim - a_idx_y;
        sign_a = -1.;
    }
}
__device__ __forceinline__ void cuUpdateCdf44_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_C4_scl_inv[1];
    a_val = sign_a * a_val * RSW_C4_scl_inv[0];

    a_val -= RSW_C4_fu_2[3] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_C4_fu_2[2] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_C4_fu_2[1] * d_val;
    a_val -= RSW_C4_fu_2[0] * __shfl_down_sync(-1, d_val, 1);

    a_val *= sign_a;
    d_val *= sign_d;

    d_val -= RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);
    d_val *= sign_d;

    a_val -= RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
}

// projection with longer coefficient
__constant__ DTYPE RSW_QN7_L10[16] = {
   0.500000000000000,   0.500000000000000,   0.500000000000000,   0.500000000000000,
  - 0.399902343750000, - 0.200195312500000,   0.200195312500000,   0.399902343750000,
  - 0.353553390593274,   0.530330085889911, - 0.176776695296637,                   0,
                   0,   0.176776695296637, - 0.530330085889911,   0.353553390593274
};
__constant__ DTYPE RSW_iQN7_L10[16] = {
   0.500000000000000, - 1.000000000000000, - 0.990225707247567,   0.423987855125528,
   0.500000000000000, - 0.500000000000000,   1.272654099342585, - 0.141559463030510,
   0.500000000000000,   0.500000000000000,   0.141559463030510, - 1.272654099342585,
   0.500000000000000,   1.000000000000000, - 0.423987855125528,   0.990225707247567
};

__constant__ DTYPE RSW_QD7_L10[16] = {
   0.800781250000000,   1.599609375000000,   1.599609375000000,   0.800781250000000,
  - 1.000000000000000, - 1.000000000000000,   1.000000000000000,   1.000000000000000,
  - 0.707106781186548,   0.530330085889911, - 0.176776695296637,                   0,
                   0,   0.176776695296637, - 0.530330085889911,   0.707106781186548
};
__constant__ DTYPE RSW_iQD7_L10[16] = {
   0.125000000000000, - 0.250000000000000, - 0.919100708749311,   0.211993927562764,
   0.250000000000000, - 0.250000000000000,   0.636672316654294,   0.070434464532254,
   0.250000000000000,   0.250000000000000, - 0.070434464532254, - 0.636672316654294,
   0.125000000000000,   0.250000000000000, - 0.211993927562764,   0.919100708749311
};

__constant__ DTYPE RSW_LN8_L10[25] = {
   0.850097656250000,   1.299804687500000,   0.500000000000000, -0.299804687500000, -0.350097656250000,
  -0.350097656250000, -0.299804687500000,   0.500000000000000,   1.299804687500000,   0.850097656250000,
  -0.500000000000000,                   0,   1.000000000000000,                   0, -0.500000000000000,
  -0.353553390593274,   0.707106781186548, -0.353553390593274,                   0,                   0,
                   0,                   0, -0.353553390593274,   0.707106781186548, -0.353553390593274
};
__constant__ DTYPE RSW_iLN8_L10[25] = {
   0.500000000000000,                   0, -0.500000000000000, -0.919100708749311,   0.211993927562764,
   0.375000000000000,   0.125000000000000,                   0,   0.777886512701802, -0.070779731515255,
   0.250000000000000,   0.250000000000000,   0.500000000000000, -0.353553390593274, -0.353553390593274,
   0.125000000000000,   0.375000000000000,                   0, -0.070779731515255,   0.777886512701803,
                   0,   0.500000000000000, -0.500000000000000,   0.211993927562764, -0.919100708749311
};

__constant__ DTYPE RSW_LD8_L10[25] = {
   0.905622728168965,   0.000000000000000,   0.000000000000000,   0.000000000000000, -0.000000000000000,
   0.000000000000000, -0.000000000000000, -0.000000000000000, -0.000000000000000,   0.905622728168965,
  -0.443145751953125,   0.399902343750000,   0.600097656250000,   0.399902343750000, -0.443145751953125,
  -0.353553390593274,   0.707106781186548, -0.353553390593274,                   0,                   0,
                   0,                   0, -0.353553390593274,   0.707106781186548, -0.353553390593274

};
__constant__ DTYPE RSW_iLD8_L10[25] = {
   0.500000000000000,                   0,   0.000000000000000, -0.000000000000000,                   0,
   0.375000000000000,   0.125000000000000,   0.500000000000000,   1.131439903295076, -0.282773659078019,
   0.250000000000000,   0.250000000000000,   1.000000000000000, -0.565547318156037, -0.565547318156038,
   0.125000000000000,   0.375000000000000,   0.500000000000000, -0.282773659078019,   1.131439903295076,
                   0,   0.500000000000000, -0.000000000000000,                   0,   0.000000000000000
};

__constant__ DTYPE RSW_LZ8_L10[25] = {
   0.850097656250000,   0.649902343750000,   0.250000000000000, - 0.149902343750000, - 0.350097656250000,
  - 0.350097656250000, - 0.149902343750000,   0.250000000000000,   0.649902343750000,   0.850097656250000,
  - 1.000000000000000,                   0,   1.000000000000000,                   0, - 1.000000000000000,
  - 0.707106781186548,   0.707106781186548, - 0.353553390593274,                   0,                   0,
                   0,                   0, - 0.353553390593274,   0.707106781186548, - 0.707106781186548

};
__constant__ DTYPE RSW_iLZ8_L10[25] = {
   0.500000000000000,                   0, - 0.250000000000000, - 0.459550354374656,   0.105996963781382,
   0.750000000000000,   0.250000000000000,                   0,   0.777886512701802, - 0.070779731515255,
   0.500000000000000,   0.500000000000000,   0.500000000000000, - 0.353553390593274, - 0.353553390593274,
   0.250000000000000,   0.750000000000000,                   0, - 0.070779731515255,   0.777886512701803,
                   0,   0.500000000000000, - 0.250000000000000,   0.105996963781382, - 0.459550354374656
};

__constant__ DTYPE RSW_CN6_L10[25] = {
   1.851562500000001,   2.097656250000000,   0.500000000000000, - 1.097656250000000, - 1.351562500000000,
  - 1.351562500000000, - 1.097656250000000,   0.500000000000000,   2.097656250000000,   1.851562500000001,
   1.000000000000000,   0.000000000000000, - 2.000000000000000,   0.000000000000000,   1.000000000000000,
   0.353553390593274, - 0.618718433538229,   0.353553390593274, - 0.088388347648318,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.618718433538229,   0.353553390593274
};
__constant__ DTYPE RSW_iCN6_L10[25] = {
   0.343750000000000,   0.156250000000000,   0.250000000000000,   0.918410174783309, -0.211303393596761,
   0.312500000000000,   0.187500000000000,                   0, -0.919791242715314,   0.212684461528766,
   0.250000000000000,   0.250000000000000, -0.250000000000000,   0.353553390593274,   0.353553390593274,
   0.187500000000000,   0.312500000000000,                   0,   0.212684461528766, -0.919791242715314,
   0.156250000000000,   0.343750000000000,   0.250000000000000, -0.211303393596761,   0.918410174783309
};

__constant__ DTYPE RSW_CD6_L10[25] = {
   2.102851867675782, - 0.000000000000000, - 0.000000000000000, - 0.000000000000000,   0.000000000000000,
  - 0.000000000000000,   0.000000000000000,   0.000000000000000,   0.000000000000000,   2.102851867675782,
   0.683105468750000, - 0.199707031250000, - 0.400390625000000, - 0.199707031250000,   0.683105468750000,
   0.353553390593274, - 0.441941738241592,   0.353553390593274, - 0.088388347648318,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.441941738241592,   0.353553390593274
};
__constant__ DTYPE RSW_iCD6_L10[25] = {
   0.281250000000000,                   0,   0.000000000000000,   0.000000000000000, -0.000000000000000,
   0.312500000000000,   0.125000000000000, -1.000000000000000, -1.980451414495135,   0.847975710251055,
   0.234375000000000,   0.234375000000000, -1.500000000000000,   0.564856784190035,   0.564856784190035,
   0.125000000000000,   0.312500000000000, -1.000000000000000,   0.847975710251055, -1.980451414495135,
                   0,   0.281250000000000, -0.000000000000000,   0.000000000000000, -0.000000000000000
};

__device__ __forceinline__ void cuUpdateCdf37_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val *= sign_d;
    a_val = sign_a * a_val + RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val += RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1) + RSW_Q5_fp[1] * a_val;

    /*printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, d_val);*/

    a_val += RSW_Q7_fu_2[6] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_Q7_fu_2[5] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_Q7_fu_2[4] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_Q7_fu_2[3] * d_val;
    a_val += RSW_Q7_fu_2[2] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_Q7_fu_2[1] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_Q7_fu_2[0] * __shfl_down_sync(-1, d_val, 3);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);

    a_val *= RSW_Q5_scl[0];
    d_val *= RSW_Q5_scl[1];

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
}
__device__ __forceinline__ void cuUpdateCdf28_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    a_val *= sign_a;
    d_val += RSW_L6_fp[0] * a_val + RSW_L6_fp[1] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val *= sign_d;

    //printf("a_idx_y: %d\t d_idx_y: %d\t a_val: %d, d_val: %f\n", a_idx_y, d_idx_y, a_val, d_val);
    a_val += RSW_L8_fu[7] * __shfl_up_sync(-1, d_val, 4);
    a_val += RSW_L8_fu[6] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_L8_fu[5] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_L8_fu[4] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_L8_fu[3] * d_val;
    a_val += RSW_L8_fu[2] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_L8_fu[1] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_L8_fu[0] * __shfl_down_sync(-1, d_val, 3);

    a_val *= RSW_L6_scl[0];
    d_val *= RSW_L6_scl[1];
    a_val /= sign_a;
}
__device__ __forceinline__ void cuUpdateCdf46_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val *= sign_a;
    d_val *= sign_d;

    a_val += RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
    d_val += RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);

    a_val += RSW_C6_fu_2[5] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_C6_fu_2[4] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_C6_fu_2[3] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_C6_fu_2[2] * d_val;
    a_val += RSW_C6_fu_2[1] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_C6_fu_2[0] * __shfl_down_sync(-1, d_val, 2);

    a_val *= RSW_C4_scl[0];
    d_val *= RSW_C4_scl[1];
}

__device__ __forceinline__ void cuUpdateCdf37_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val = sign_a * a_val * RSW_Q5_scl[1];
    d_val = sign_d * d_val * RSW_Q5_scl[0];

    a_val -= RSW_Q7_fu_2[6] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_Q7_fu_2[5] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_Q7_fu_2[4] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_Q7_fu_2[3] * d_val;
    a_val -= RSW_Q7_fu_2[2] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_Q7_fu_2[1] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_Q7_fu_2[0] * __shfl_down_sync(-1, d_val, 3);

    //printf("ad_idx_y: %d\t a_val: %f, d_val: %f\n", ad_idx_y, a_val, d_val);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, temp);
    d_val -= RSW_Q5_fp[1] * a_val + RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val, d_val: %f, %f\n", a_val, d_val);
    a_val -= RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);
}
__device__ __forceinline__ void cuUpdateCdf28_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_L6_scl[0];
    a_val = sign_a * a_val * RSW_L6_scl[1];

    a_val -= RSW_L8_fu[7] * __shfl_up_sync(-1, d_val, 4);
    a_val -= RSW_L8_fu[6] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_L8_fu[5] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_L8_fu[4] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_L8_fu[3] * d_val;
    a_val -= RSW_L8_fu[2] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_L8_fu[1] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_L8_fu[0] * __shfl_down_sync(-1, d_val, 3);

    d_val -= RSW_L6_fp[1] * a_val + RSW_L6_fp[0] * __shfl_down_sync(-1, a_val, 1);
    a_val /= sign_a;
}
__device__ __forceinline__ void cuUpdateCdf46_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_C4_scl_inv[1];
    a_val = sign_a * a_val * RSW_C4_scl_inv[0];

    a_val -= RSW_C6_fu_2[5] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_C6_fu_2[4] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_C6_fu_2[3] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_C6_fu_2[2] * d_val;
    a_val -= RSW_C6_fu_2[1] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_C6_fu_2[0] * __shfl_down_sync(-1, d_val, 2);

    a_val *= sign_a;
    d_val *= sign_d;

    d_val -= RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);
    d_val *= sign_d;

    a_val -= RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
}

__device__ __forceinline__ DTYPE cuUpdateCdf37_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QN7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf37_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QD7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LN8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LD8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_l10_z(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LZ8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf46_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CN6_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf46_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CD6_L10, 5);
}

__device__ __forceinline__ DTYPE cuUpdateCdf37_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQN7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf37_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQD7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLN8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLD8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf28_il10_z(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLZ8_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf46_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCN6_L10, 5);
}
__device__ __forceinline__ DTYPE cuUpdateCdf46_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCD6_L10, 5);
}

// ----------------------------------------------------------cdf 311--------------------------------------------
__constant__ DTYPE RSW_QN11_L10[64] = {
   0.353553390627855,   0.353553390592116,   0.353553390589562,   0.353553390564651,   0.353553390621897,   0.353553390596986,   0.353553390594432,   0.353553390558693,
  - 0.323785366110183, - 0.278627624193305, - 0.194607743736687, - 0.071569616306587,   0.071569616321321,   0.194607743728499,   0.278627624207115,   0.323785366089827,
  - 0.535702705382080, - 0.132911682134061,   0.541545867930499,   0.548578262321690, - 0.137460708622509, - 0.363780975335252, - 0.044853210448411,   0.124585151670121,
  - 0.124585151669832,   0.044853210445904,   0.363780975341616,   0.137460708622898, - 0.548578262339444, - 0.541545867914912,   0.132911682140169,   0.535702705373601,
  - 0.353553390593274,   0.530330085889911, - 0.176776695296637,                   0,                   0,                   0,                   0,                   0,
                   0,   0.176776695296637, - 0.530330085889911,   0.530330085889911, - 0.176776695296637,                   0,                   0,                   0,
                   0,                   0,                   0,   0.176776695296637, - 0.530330085889911,   0.530330085889911, - 0.176776695296637,                   0,
                   0,                   0,                   0,                   0,                   0,   0.176776695296637, - 0.530330085889911,   0.353553390593274
};
__constant__ DTYPE RSW_iQN11_L10[64] = {
   0.353553390593274, - 0.707106781180524, - 0.677764892575426,   0.322235107416056, - 0.913905519611462,   0.328753510573946, - 0.126955748606592,   0.044598783569049,
   0.353553390593274, - 0.618718433532206, - 0.280544281001245,   0.219455718990237,   1.267925560129504, - 0.095838561763922,   0.029860199232363, - 0.020589241259352,
   0.353553390593274, - 0.441941738235569,   0.513896942147116,   0.013896942138599, - 0.025266529906070, - 0.945022706440336,   0.343492094910524, - 0.150965290916259,
   0.353553390593274, - 0.176776695290614,   0.705558776865236, - 0.294441223139023, - 0.449858602256839,   0.969032248788601,   0.010527945583068,   0.005850656922770,
   0.353553390593274,   0.176776695302660,   0.294441223153113, - 0.705558776842628, - 0.005850656922805, - 0.010527945569492, - 0.969032248750003,   0.449858602257733,
   0.353553390593274,   0.441941738247615, - 0.013896942131061, - 0.513896942122543,   0.150965290915936, - 0.343492094909849,   0.945022706479007,   0.025266529919651,
   0.353553390593274,   0.618718433544252, - 0.219455718987286,   0.280544281021231,   0.020589241259382, - 0.029860199232472,   0.095838561777718, - 1.267925560091477,
   0.353553390593274,   0.707106781192570, - 0.322235107415399,   0.677764892593118, - 0.044598783568946,   0.126955748606342, - 0.328753510573267,   0.913905519636587
};

__constant__ DTYPE RSW_QD11_L10[64] = {
   0.286278465346205,   0.778430975001444,   1.114510496828568,   1.295141464322892,   1.295141464477147,   1.114510496773112,   0.778430974859299,   0.286278465165427,
  - 0.502795044057268, - 1.162816040363813, - 1.162816040390209, - 0.502795043982985,   0.502795044042912,   1.162816040343034,   1.162816040369429,   0.502795043968629,
  - 0.843019485494917, - 0.952663421620188,   0.544029235850398,   0.830143928514973, - 0.169856071475924, - 0.455970764151824,   0.047336578368272,   0.156980514523505,
  - 0.156980514523226, - 0.047336578370589,   0.455970764158218,   0.169856071476259, - 0.830143928532844, - 0.544029235839560,   0.952663421640952,   0.843019485458353,
  - 0.707106781186548,   0.530330085889911, - 0.176776695296637,                   0,                   0,                   0,                   0,                   0,
                   0,   0.176776695296637, - 0.530330085889911,   0.530330085889911, - 0.176776695296637,                   0,                   0,                   0,
                   0,                   0,                   0,   0.176776695296637, - 0.530330085889911,   0.530330085889911, - 0.176776695296637,                   0,
                   0,                   0,                   0,                   0,                   0,   0.176776695296637, - 0.530330085889911,   0.707106781186548
};
__constant__ DTYPE RSW_iQD11_L10[64] = {
   0.044194173824159, - 0.088388347645024, - 0.330558776844685,   0.080558776854004, - 0.957260802534962,   0.335931287462584, - 0.108076657872344,   0.022299391784532,
   0.110485434560398, - 0.176776695288400, - 0.388896942117483,   0.138896942136121,   0.603240762094286,   0.139193844714539, - 0.037037976121048,   0.001710150525128,
   0.154679608384557, - 0.176776695285105,   0.155544281026291,   0.094455718992346, - 0.018088753017431, - 0.926143615706088,   0.321192703126007, - 0.084067115562677,
   0.176776695296637, - 0.088388347635140,   0.302764892595885, - 0.052764892577247, - 0.214826195778379,   0.902134073435189,   0.032827337367549, - 0.013028433811454,
   0.176776695296637,   0.088388347661497,   0.052764892591297, - 0.302764892572660,   0.013028433811443, - 0.032827337354009, - 0.902134073396421,   0.214826195778797,
   0.154679608384557,   0.176776695308168, - 0.094455718985177, - 0.155544280996185,   0.084067115562524, - 0.321192703125369,   0.926143615744783,   0.018088753030966,
   0.110485434560398,   0.176776695304874, - 0.138896942133539,   0.388896942152177, - 0.001710150525135,   0.037037976121110, - 0.139193844701218, - 0.603240762054947,
   0.044194173824159,   0.088388347651613, - 0.080558776853860,   0.330558776863179, - 0.022299391784465,   0.108076657872118, - 0.335931287461951,   0.957260802587203
};

__constant__ DTYPE RSW_LN12_L10[81] = {
   0.611071870465493,   1.150574124626437,   0.955966380898819,   0.677338756699653,   0.353553390593274,   0.029768024486895, - 0.248859599712271, - 0.443467343439889, - 0.257518479872219,
  - 0.257518479872219, - 0.443467343439889, - 0.248859599712271,   0.029768024486895,   0.353553390593274,   0.677338756699653,   0.955966380898819,   1.150574124626437,   0.611071870465493,
  - 0.416402771092719, - 0.581408020187658, - 0.000000000000000,   0.581408020187658,   0.832805542185437,   0.581408020187658,   0.000000000000000, - 0.581408020187658, - 0.416402771092719,
  - 0.460754871368408, - 0.078490257263183,   0.874173164367676,   0.330143928527832, - 0.500000000000000, - 0.330143928527832,   0.125826835632324,   0.078490257263183, - 0.039245128631592,
  - 0.039245128631592,   0.078490257263184,   0.125826835632324, - 0.330143928527832, - 0.500000000000000,   0.330143928527832,   0.874173164367676, - 0.078490257263183, - 0.460754871368408,
  - 0.353553390593274,   0.707106781186548, - 0.353553390593274,                   0,                   0,                   0,                   0,                   0,                   0,
                   0,                   0, - 0.353553390593274,   0.707106781186548, - 0.353553390593274,                   0,                   0,                   0,                   0,
                   0,                   0,                   0,                   0, - 0.353553390593274,   0.707106781186548, - 0.353553390593274,                   0,                   0,
                   0,                   0,                   0,                   0,                   0,                   0, - 0.353553390593274,   0.707106781186548, - 0.353553390593274
};
__constant__ DTYPE RSW_iLN12_L10[81] = {
   0.353553390593274,                   0, - 0.353553390593274, - 0.661117553710938,   0.161117553710938, - 0.957260802565862,   0.335931287468687, - 0.108076657874303,   0.022299391784931,
   0.309359216769114,   0.044194173824159, - 0.176776695296637, - 0.058338165283203,   0.058338165283204,   0.780250782315870, - 0.098368721375698,   0.035519340876253, - 0.010294620629878,
   0.265165042944955,   0.088388347648318,                   0,   0.544441223144531, - 0.044441223144531, - 0.310664757548588, - 0.532668730220082,   0.179115339626809, - 0.042888633044686,
   0.220970869120796,   0.132582521472478,   0.176776695296637,   0.147220611572265, - 0.147220611572265, - 0.098368721375698,   0.914138844567821, - 0.144182682881829,   0.035519340876253,
   0.176776695296637,   0.176776695296637,   0.353553390593274, - 0.250000000000000, - 0.250000000000000,   0.113927314797192, - 0.467480705390466, - 0.467480705390466,   0.113927314797192,
   0.132582521472478,   0.220970869120796,   0.176776695296637, - 0.147220611572266,   0.147220611572265,   0.035519340876253, - 0.144182682881829,   0.914138844567821, - 0.098368721375698,
   0.088388347648318,   0.265165042944955,                   0, - 0.044441223144531,   0.544441223144531, - 0.042888633044686,   0.179115339626809, - 0.532668730220082, - 0.310664757548588,
   0.044194173824159,   0.309359216769114, - 0.176776695296637,   0.058338165283204, - 0.058338165283203, - 0.010294620629878,   0.035519340876253, - 0.098368721375698,   0.780250782315870,
                   0,   0.353553390593274, - 0.353553390593274,   0.161117553710938, - 0.661117553710938,   0.022299391784931, - 0.108076657874303,   0.335931287468687, - 0.957260802565862
};

__constant__ DTYPE RSW_LD12_L10[81] = {
   0.818930771126050,  -0.000000000000000,   0.000000000000000,  -0.000000000000000,  -0.000000000000000,   0.000000000000000,  -0.000000000000000,   0.000000000000000,  -0.000000000000000,
   0.000000000000000,  -0.000000000000000,  -0.000000000000000,  -0.000000000000000,   0.000000000000000,  -0.000000000000000,  -0.000000000000000,  -0.000000000000000,   0.818930771126050,
  -0.437750668272425,   0.161892683053189,   0.301206495152773,   0.398510367016582,   0.434295175168856,   0.398510367016582,   0.301206495152772,   0.161892683053189,  -0.437750668272425,
  -0.460699796676636,   0.535702705383300,   0.668614387512207,   0.127068519592286,  -0.421509742736817,  -0.284049034118653,   0.079731941223145,   0.124585151672363,  -0.033886194229126,
  -0.033886194229126,   0.124585151672363,   0.079731941223145,  -0.284049034118653,  -0.421509742736817,   0.127068519592285,   0.668614387512207,   0.535702705383300,  -0.460699796676636,
  -0.353553390593274,   0.707106781186548,  -0.353553390593274,                   0,                   0,                   0,                   0,                   0,                   0,
                   0,                   0,  -0.353553390593274,   0.707106781186548,  -0.353553390593274,                   0,                   0,                   0,                   0,
                   0,                   0,                   0,                   0,  -0.353553390593274,   0.707106781186548,  -0.353553390593274,                   0,                   0,
                   0,                   0,                   0,                   0,                   0,                   0,  -0.353553390593274,   0.707106781186548,  -0.353553390593274

};
__constant__ DTYPE RSW_iLD12_L10[81] = {
   0.353553390593274,                   0, - 0.000000000000000,   0.000000000000000, - 0.000000000000000,                   0,                   0, - 0.000000000000000,                   0,
   0.309359216769114,   0.044194173824159,   0.176776695296637,   0.397220611572265, - 0.102779388427734,   1.090915539864458, - 0.212296036172890,   0.078407973920939, - 0.032594012414808,
   0.265165042944955,   0.088388347648318,   0.353553390593274,   0.794441223144531, - 0.205558776855469, - 0.646596045017275, - 0.424592072345780,   0.156815947841878, - 0.065188024829617,
   0.220970869120796,   0.132582521472478,   0.530330085889910,   0.191661834716796, - 0.308338165283203, - 0.212296036172890,   0.957027477612507, - 0.166482074666759,   0.078407973920939,
   0.176776695296637,   0.176776695296637,   0.707106781186547, - 0.411117553710938, - 0.411117553710938,   0.222003972671495, - 0.489780097175396, - 0.489780097175396,   0.222003972671495,
   0.132582521472478,   0.220970869120796,   0.530330085889910, - 0.308338165283203,   0.191661834716797,   0.078407973920939, - 0.166482074666759,   0.957027477612507, - 0.212296036172890,
   0.088388347648318,   0.265165042944955,   0.353553390593274, - 0.205558776855469,   0.794441223144531, - 0.065188024829617,   0.156815947841878, - 0.424592072345780, - 0.646596045017275,
   0.044194173824159,   0.309359216769114,   0.176776695296637, - 0.102779388427734,   0.397220611572266, - 0.032594012414808,   0.078407973920939, - 0.212296036172890,   1.090915539864458,
                   0,   0.353553390593274,   0.000000000000000,   0.000000000000000, - 0.000000000000000,                   0,   0.000000000000000,                   0,                   0
};

__constant__ DTYPE RSW_CN10_L10[81] = {
   1.321890556550059,   2.322163429339827,   1.697871477956067,   1.076077260792937,   0.353553390572726, - 0.368970479888945, - 0.990764696948595, - 1.615056647990104, - 0.968337165637781,
  - 0.968337165637783, - 1.615056647990104, - 0.990764696948594, - 0.368970479888944,   0.353553390572726,   1.076077260792935,   1.697871477956069,   2.322163429339827,   1.321890556550061,
   1.005590088042233,   1.320041992682841,   0.000000000000000, - 1.320041992682841, - 2.011180176084466, - 1.320041992682841,   0.000000000000000,   1.320041992682841,   1.005590088042232,
   0.843019485477997,   0.109643936152444, - 1.496692657477755, - 0.286114692674810,   1.000000000000000,   0.286114692674810, - 0.503307342522245, - 0.109643936152444,   0.156980514522004,
   0.156980514522004, - 0.109643936152444, - 0.503307342522245,   0.286114692674810,   1.000000000000000, - 0.286114692674810, - 1.496692657477755,   0.109643936152444,   0.843019485477997,
   0.353553390593274, - 0.618718433538229,   0.353553390593274, - 0.088388347648318,                   0,                   0,                   0,                   0,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.530330085889911,   0.353553390593274, - 0.088388347648318,                   0,                   0,                   0,
                   0,                   0,                   0, - 0.088388347648318,   0.353553390593274, - 0.530330085889911,   0.353553390593274, - 0.088388347648318,                   0,
                   0,                   0,                   0,                   0,                   0, - 0.088388347648318,   0.353553390593274, - 0.618718433538229,   0.353553390593274
};
__constant__ DTYPE RSW_iCN10_L10[81] = {
   0.237543684304856,   0.116009706288418,   0.132582521442632,   0.386146545364206, - 0.136146545406414,   1.020636175666368, - 0.395443986254364,   0.088976380476372, - 0.007061788761519,
   0.232019412576836,   0.121533978016438,   0.088388347618473,   0.055587768510996, - 0.055587768553204, - 0.893885429451366,   0.276418588664278, - 0.127176935266281,   0.037536994807130,
   0.218208733256786,   0.135344657336488, - 0.000000000029845, - 0.333309173622029,   0.083309173579821,   0.312596094706002,   0.554806278071370, - 0.201252887507942,   0.040957295857427,
   0.198873782208716,   0.154679608384557, - 0.088388347678164, - 0.177764892608264,   0.177764892566056,   0.276418588664278, - 1.297480953381925,   0.441132518737689, - 0.127176935266281,
   0.176776695296637,   0.176776695296637, - 0.132582521502323,   0.124999999978896,   0.124999999978896, - 0.153233802888996,   0.506787193452424,   0.506787193452424, - 0.153233802888996,
   0.154679608384557,   0.198873782208716, - 0.088388347678164,   0.177764892566056, - 0.177764892608264, - 0.127176935266281,   0.441132518737689, - 1.297480953381925,   0.276418588664278,
   0.135344657336488,   0.218208733256786, - 0.000000000029845,   0.083309173579821, - 0.333309173622029,   0.040957295857427, - 0.201252887507942,   0.554806278071370,   0.312596094706002,
   0.121533978016438,   0.232019412576836,   0.088388347618473, - 0.055587768553204,   0.055587768510996,   0.037536994807130, - 0.127176935266281,   0.276418588664278, - 0.893885429451366,
   0.116009706288418,   0.237543684304856,   0.132582521442632, - 0.136146545406414,   0.386146545364206, - 0.007061788761519,   0.088976380476372, - 0.395443986254364,   1.020636175666368
};

__constant__ DTYPE RSW_CD10_L10[81] = {
   2.520165618960755, - 0.000000000000000,   0.000000000000000,   0.000000000000000, - 0.000000000000000,   0.000000000000000, - 0.000000000000000, - 0.000000000000000,   0.000000000000000,
   0.000000000000000,   0.000000000000000,   0.000000000000000, - 0.000000000000000,   0.000000000000000, - 0.000000000000000, - 0.000000000000000,   0.000000000000000,   2.520165618960755,
   0.905558607532715, - 0.090315483792618, - 0.168039760931542, - 0.246076254833121, - 0.286278465273481, - 0.246076254833121, - 0.168039760931542, - 0.090315483792618,   0.905558607532716,
   0.760251045233752, - 0.402791023234440, - 0.674457550064915, - 0.007032394407186,   0.686038970955993,   0.226320266712074, - 0.318927764890595, - 0.169438362115180,   0.135544776912722,
   0.135544776912722, - 0.169438362115180, - 0.318927764890595,   0.226320266712074,   0.686038970955993, - 0.007032394407186, - 0.674457550064915, - 0.402791023234440,   0.760251045233752,
   0.353553390593274, - 0.441941738241592,   0.353553390593274, - 0.088388347648318,                   0,                   0,                   0,                   0,                   0,
                   0, - 0.088388347648318,   0.353553390593274, - 0.530330085889911,   0.353553390593274, - 0.088388347648318,                   0,                   0,                   0,
                   0,                   0,                   0, - 0.088388347648318,   0.353553390593274, - 0.530330085889911,   0.353553390593274, - 0.088388347648318,                   0,
                   0,                   0,                   0,                   0,                   0, - 0.088388347648318,   0.353553390593274, - 0.441941738241592,   0.353553390593274
};
__constant__ DTYPE RSW_iCD10_L10[81] = {
   0.149155336656537,                   0,   0.000000000000000,   0.000000000000000, - 0.000000000000000,                   0,                   0,                   0,   0.000000000000000,
   0.209922325664756,   0.044194173824159, - 0.353553390593274, - 0.677764892587160,   0.322235107412840, - 1.827811039256913,   0.657507021141269, - 0.253911497210710,   0.089197567137298,
   0.211303393596761,   0.087007279716313, - 0.662912607362388, - 0.958309173600925,   0.541690826399075,   0.708040080960366,   0.465829897594998, - 0.194191098746423,   0.048019084618946,
   0.193349510480697,   0.127058249744458, - 0.883883476483184, - 0.444412231467900,   0.555587768532100,   0.657507021141269, - 1.424215515326353,   0.492793091067858, - 0.253911497210710,
   0.162966015976587,   0.162966015976587, - 0.972271824131503,   0.261146545385310,   0.261146545385310, - 0.242210183365368,   0.513848982213943,   0.513848982213943, - 0.242210183365368,
   0.127058249744458,   0.193349510480697, - 0.883883476483184,   0.555587768532100, - 0.444412231467900, - 0.253911497210710,   0.492793091067858, - 1.424215515326353,   0.657507021141269,
   0.087007279716313,   0.211303393596761, - 0.662912607362388,   0.541690826399075, - 0.958309173600925,   0.048019084618946, - 0.194191098746423,   0.465829897594998,   0.708040080960366,
   0.044194173824159,   0.209922325664756, - 0.353553390593274,   0.322235107412840, - 0.677764892587160,   0.089197567137298, - 0.253911497210710,   0.657507021141269, - 1.827811039256913,
                   0,   0.149155336656537, - 0.000000000000000,   0.000000000000000, - 0.000000000000000, - 0.000000000000000,                   0,                   0,                   0
};

__device__ __forceinline__ void cuUpdateCdf311_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val *= sign_d;
    a_val = sign_a * a_val + RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val += RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1) + RSW_Q5_fp[1] * a_val;

    /*printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, d_val);*/

    a_val += RSW_Q11_fu_2[10] * __shfl_up_sync(-1, d_val, 5);
    a_val += RSW_Q11_fu_2[9] * __shfl_up_sync(-1, d_val, 4);
    a_val += RSW_Q11_fu_2[8] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_Q11_fu_2[7] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_Q11_fu_2[6] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_Q11_fu_2[5] * d_val;
    a_val += RSW_Q11_fu_2[4] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_Q11_fu_2[3] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_Q11_fu_2[2] * __shfl_down_sync(-1, d_val, 3);
    a_val += RSW_Q11_fu_2[1] * __shfl_down_sync(-1, d_val, 4);
    a_val += RSW_Q11_fu_2[0] * __shfl_down_sync(-1, d_val, 5);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);

    a_val *= RSW_Q5_scl[0];
    d_val *= RSW_Q5_scl[1];

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
}
__device__ __forceinline__ void cuUpdateCdf212_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val += RSW_L6_fp[0] * a_val + RSW_L6_fp[1] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val: %f, d_val: %f\n", a_val, d_val);
    d_val *= sign_d;

    //printf("a_idx_y: %d\t d_idx_y: %d\t a_val: %d, d_val: %f\n", a_idx_y, d_idx_y, a_val, d_val);
    a_val += RSW_L12_fu[11] * __shfl_up_sync(-1, d_val, 6);
    a_val += RSW_L12_fu[10] * __shfl_up_sync(-1, d_val, 5);
    a_val += RSW_L12_fu[9] * __shfl_up_sync(-1, d_val, 4);
    a_val += RSW_L12_fu[8] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_L12_fu[7] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_L12_fu[6] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_L12_fu[5] * d_val;
    a_val += RSW_L12_fu[4] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_L12_fu[3] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_L12_fu[2] * __shfl_down_sync(-1, d_val, 3);
    a_val += RSW_L12_fu[1] * __shfl_down_sync(-1, d_val, 4);
    a_val += RSW_L12_fu[0] * __shfl_down_sync(-1, d_val, 5);

    a_val *= RSW_L6_scl[0];
    d_val *= RSW_L6_scl[1];
}
__device__ __forceinline__ void cuUpdateCdf410_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val *= sign_a;
    d_val *= sign_d;

    a_val += RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
    d_val += RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);

    a_val += RSW_C10_fu_2[9] * __shfl_up_sync(-1, d_val, 5);
    a_val += RSW_C10_fu_2[8] * __shfl_up_sync(-1, d_val, 4);
    a_val += RSW_C10_fu_2[7] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_C10_fu_2[6] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_C10_fu_2[5] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_C10_fu_2[4] * d_val;
    a_val += RSW_C10_fu_2[3] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_C10_fu_2[2] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_C10_fu_2[1] * __shfl_down_sync(-1, d_val, 3);
    a_val += RSW_C10_fu_2[0] * __shfl_down_sync(-1, d_val, 4);

    a_val *= RSW_C4_scl[0];
    d_val *= RSW_C4_scl[1];
}

__device__ __forceinline__ void cuUpdateCdf311_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val = sign_a * a_val * RSW_Q5_scl[1];
    d_val = sign_d * d_val * RSW_Q5_scl[0];

    a_val -= RSW_Q11_fu_2[10] * __shfl_up_sync(-1, d_val, 5);
    a_val -= RSW_Q11_fu_2[9] * __shfl_up_sync(-1, d_val, 4);
    a_val -= RSW_Q11_fu_2[8] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_Q11_fu_2[7] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_Q11_fu_2[6] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_Q11_fu_2[5] * d_val;
    a_val -= RSW_Q11_fu_2[4] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_Q11_fu_2[3] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_Q11_fu_2[2] * __shfl_down_sync(-1, d_val, 3);
    a_val -= RSW_Q11_fu_2[1] * __shfl_down_sync(-1, d_val, 4);
    a_val -= RSW_Q11_fu_2[0] * __shfl_down_sync(-1, d_val, 5);

    //printf("ad_idx_y: %d\t a_val: %f, d_val: %f\n", ad_idx_y, a_val, d_val);

    //DTYPE temp = __shfl_down_sync(-1, d_val, 1);
    //printf("d_idx_y: %d\t data_idx_xz: %d, val: %f\n", d_idx_y, data_idx_xz, temp);
    d_val -= RSW_Q5_fp[1] * a_val + RSW_Q5_fp[0] * __shfl_down_sync(-1, a_val, 1);

    //printf("a_val, d_val: %f, %f\n", a_val, d_val);
    a_val -= RSW_Q5_fu_1[0] * __shfl_up_sync(-1, d_val, 1);
}
__device__ __forceinline__ void cuUpdateCdf212_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_L6_scl[0];
    a_val = sign_a * a_val * RSW_L6_scl[1];

    a_val -= RSW_L12_fu[11] * __shfl_up_sync(-1, d_val, 6);
    a_val -= RSW_L12_fu[10] * __shfl_up_sync(-1, d_val, 5);
    a_val -= RSW_L12_fu[9] * __shfl_up_sync(-1, d_val, 4);
    a_val -= RSW_L12_fu[8] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_L12_fu[7] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_L12_fu[6] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_L12_fu[5] * d_val;
    a_val -= RSW_L12_fu[4] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_L12_fu[3] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_L12_fu[2] * __shfl_down_sync(-1, d_val, 3);
    a_val -= RSW_L12_fu[1] * __shfl_down_sync(-1, d_val, 4);
    a_val -= RSW_L12_fu[0] * __shfl_down_sync(-1, d_val, 5);

    d_val -= RSW_L6_fp[1] * a_val + RSW_L6_fp[0] * __shfl_down_sync(-1, a_val, 1);
}
__device__ __forceinline__ void cuUpdateCdf410_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val * RSW_C4_scl_inv[1];
    a_val = sign_a * a_val * RSW_C4_scl_inv[0];

    a_val -= RSW_C10_fu_2[9] * __shfl_up_sync(-1, d_val, 5);
    a_val -= RSW_C10_fu_2[8] * __shfl_up_sync(-1, d_val, 4);
    a_val -= RSW_C10_fu_2[7] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_C10_fu_2[6] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_C10_fu_2[5] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_C10_fu_2[4] * d_val;
    a_val -= RSW_C10_fu_2[3] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_C10_fu_2[2] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_C10_fu_2[1] * __shfl_down_sync(-1, d_val, 3);
    a_val -= RSW_C10_fu_2[0] * __shfl_down_sync(-1, d_val, 4);

    a_val *= sign_a;
    d_val *= sign_d;

    d_val -= RSW_C4_fp[1] * a_val + RSW_C4_fp[0] * __shfl_down_sync(-1, a_val, 1);
    d_val *= sign_d;

    a_val -= RSW_C4_fu_1[1] * __shfl_up_sync(-1, d_val, 1) + RSW_C4_fu_1[0] * d_val;
}

__device__ __forceinline__ DTYPE cuUpdateCdf311_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QN11_L10, 8);
}
__device__ __forceinline__ DTYPE cuUpdateCdf311_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_QD11_L10, 8);
}
__device__ __forceinline__ DTYPE cuUpdateCdf212_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LN12_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf212_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_LD12_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf410_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CN10_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf410_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_CD10_L10, 9);
}

__device__ __forceinline__ DTYPE cuUpdateCdf311_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQN11_L10, 8);
}
__device__ __forceinline__ DTYPE cuUpdateCdf311_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iQD11_L10, 8);
}
__device__ __forceinline__ DTYPE cuUpdateCdf212_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLN12_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf212_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iLD12_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf410_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCN10_L10, 9);
}
__device__ __forceinline__ DTYPE cuUpdateCdf410_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iCD10_L10, 9);
}

// ----------------------------------------------------------cdf 20--------------------------------------------
__constant__ DTYPE RSW_20_L10[25] = {
   2.000000000000000,                   0,                   0,                   0,                   0,
                   0,                   0,                   0,                   0,   2.000000000000000,
   0.500000000000000,                   0, - 1.000000000000000,                   0,   0.500000000000000,
   0.353553390593274, - 0.707106781186547,   0.353553390593274,                   0,                   0,
                   0,                   0,   0.353553390593274, - 0.707106781186547,   0.353553390593274
};
__constant__ DTYPE RSW_i20_L10[25] = {
   0.500000000000000,                   0,                   0,                   0,                   0,
   0.375000000000000,   0.125000000000000, - 0.500000000000000, - 1.414213562373095,                   0,
   0.250000000000000,   0.250000000000000, - 1.000000000000000,                   0,                   0,
   0.125000000000000,   0.375000000000000, - 0.500000000000000,                   0, - 1.414213562373095,
                   0,   0.500000000000000,                   0,                   0,                   0
};

__device__ __forceinline__ void cuUpdateCdf20_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val += RSW_L6_fp[0] * a_val + RSW_L6_fp[1] * __shfl_down_sync(-1, a_val, 1);

    a_val *= RSW_L6_scl[0];
    d_val *= -RSW_L6_scl[1];
}
__device__ __forceinline__ DTYPE cuUpdateCdf20_l10(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_20_L10, 5);
}

__device__ __forceinline__ void cuUpdateCdf20_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = -sign_d * d_val * RSW_L6_scl[0];
    a_val = sign_a * a_val * RSW_L6_scl[1];

    d_val -= RSW_L6_fp[1] * a_val + RSW_L6_fp[0] * __shfl_down_sync(-1, a_val, 1);
}
__device__ __forceinline__ DTYPE cuUpdateCdf20_il10(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_i20_L10, 5);
}

// ----------------------------------------------------------cdf 11--------------------------------------------
__constant__ DTYPE RSW_11_L10[16] = {
   0.500000000000000,   0.500000000000000,   0.500000000000000,   0.500000000000000,
  - 0.500000000000000, - 0.500000000000000,   0.500000000000000,   0.500000000000000,
  - 0.707106781186548,   0.707106781186548,                   0,                   0,
                   0,                   0, - 0.707106781186548,   0.707106781186548
};
__constant__ DTYPE RSW_i11_L10[16] = {
   0.500000000000000, - 0.500000000000000, - 0.707106781186547,                   0,
   0.500000000000000, - 0.500000000000000,   0.707106781186547,                   0,
   0.500000000000000,   0.500000000000000,                   0, - 0.707106781186547,
   0.500000000000000,   0.500000000000000,                   0,   0.707106781186547
};

__device__ __forceinline__ void cuUpdateCdf11_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val *= sign_a;
    d_val = sign_d * d_val + RSW_T1_fp[0] * a_val;
    a_val += RSW_T1_fu[0] * d_val;

    a_val *= RSW_T1_scl[0];
    d_val *= RSW_T1_scl[1];
}
__device__ __forceinline__ DTYPE cuUpdateCdf11_l10(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_11_L10, 4);
}

__device__ __forceinline__ void cuUpdateCdf11_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = d_val * RSW_T1_scl[0];
    a_val = a_val * RSW_T1_scl[1];

    a_val -= RSW_T1_fu[0] * d_val;
    d_val -= RSW_T1_fp[0] * a_val;
}
__device__ __forceinline__ DTYPE cuUpdateCdf11_il10(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_i11_L10, 4);
}

// ----------------------------------------------------------cdf 17--------------------------------------------
__constant__ DTYPE RSW_TD7_L10[16] = {
   0.308593750000000,   0.691406250000000,   0.691406250000000,   0.308593750000000,
   0.500000000000000,   0.500000000000000, - 0.500000000000000, - 0.500000000000000,
   0.707106781186548, - 0.707106781186548,                   0,                   0,
                   0,                   0,   0.707106781186548, - 0.707106781186548
};
__constant__ DTYPE RSW_iTD7_L10[16] = {
   0.500000000000000,   0.500000000000000,   0.842451438523035, -0.135344657336488,
   0.500000000000000,   0.500000000000000, -0.571762123850060, -0.135344657336488,
   0.500000000000000, -0.500000000000000,   0.135344657336488,   0.571762123850060,
   0.500000000000000, -0.500000000000000,   0.135344657336488, -0.842451438523035
};
__constant__ DTYPE RSW_TN7_L10[16] = {
   0.500000000000000,   0.500000000000000,   0.500000000000000,   0.500000000000000,
   0.691406250000000,   0.308593750000000, - 0.308593750000000, - 0.691406250000000,
   0.707106781186548, - 0.707106781186548,                   0,                   0,
                   0,                   0,   0.707106781186548, - 0.707106781186548
};
__constant__ DTYPE RSW_iTN7_L10[16] = {
   0.500000000000000,   0.500000000000000,   0.571762123850060, - 0.135344657336488,
   0.500000000000000,   0.500000000000000, - 0.842451438523035, - 0.135344657336488,
   0.500000000000000, - 0.500000000000000,   0.135344657336488,   0.842451438523035,
   0.500000000000000, - 0.500000000000000,   0.135344657336488, - 0.571762123850060
};

__device__ __forceinline__ void cuUpdateCdf17_ad(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    a_val *= sign_a;
    d_val = sign_d * d_val + RSW_T1_fp[0] * a_val;

    a_val += RSW_T7_fu[6] * __shfl_up_sync(-1, d_val, 3);
    a_val += RSW_T7_fu[5] * __shfl_up_sync(-1, d_val, 2);
    a_val += RSW_T7_fu[4] * __shfl_up_sync(-1, d_val, 1);
    a_val += RSW_T7_fu[3] * d_val;
    a_val += RSW_T7_fu[2] * __shfl_down_sync(-1, d_val, 1);
    a_val += RSW_T7_fu[1] * __shfl_down_sync(-1, d_val, 2);
    a_val += RSW_T7_fu[0] * __shfl_down_sync(-1, d_val, 3);

    a_val *= RSW_T7_scl[0];
    d_val *= RSW_T7_scl[1];
}
__device__ __forceinline__ DTYPE cuUpdateCdf17_l10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_TD7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf17_l10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_TN7_L10, 4);
}

__device__ __forceinline__ void cuUpdateCdf17_ad_i(DTYPE& a_val, DTYPE& d_val, DTYPE sign_a, DTYPE sign_d)
{
    d_val = sign_d * d_val / RSW_T7_scl[1];
    a_val = sign_a * a_val / RSW_T7_scl[0];

    a_val -= RSW_T7_fu[6] * __shfl_up_sync(-1, d_val, 3);
    a_val -= RSW_T7_fu[5] * __shfl_up_sync(-1, d_val, 2);
    a_val -= RSW_T7_fu[4] * __shfl_up_sync(-1, d_val, 1);
    a_val -= RSW_T7_fu[3] * d_val;
    a_val -= RSW_T7_fu[2] * __shfl_down_sync(-1, d_val, 1);
    a_val -= RSW_T7_fu[1] * __shfl_down_sync(-1, d_val, 2);
    a_val -= RSW_T7_fu[0] * __shfl_down_sync(-1, d_val, 3);

    d_val -= RSW_T1_fp[0] * a_val;
}
__device__ __forceinline__ DTYPE cuUpdateCdf17_il10_d(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iTD7_L10, 4);
}
__device__ __forceinline__ DTYPE cuUpdateCdf17_il10_n(DTYPE a_val, int idx_coef)
{
    return cuUpdateL10(a_val, idx_coef, RSW_iTN7_L10, 4);
}
#endif