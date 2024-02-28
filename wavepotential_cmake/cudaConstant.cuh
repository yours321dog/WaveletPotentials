//#ifndef _CUDACONSTANT_H_
//#define _CUDACONSTANT_H_
//
//#include "cudaGlobal.cuh"
//#include "Utils.h"
//
//__constant__ extern DTYPE Q1_fu_1[1];
//__constant__ extern DTYPE Q1_fp[2];
//__constant__ extern DTYPE Q1_fu_2[2];
////__constant__ DTYPE Q1_scl[2] = { 2.121320343559643, 0.471404520791032};//{3./sqrt(2.),sqrt(2.)/3.}
//__constant__ extern DTYPE Q1_scl[2];//{3./sqrt(2.),sqrt(2.)/3.}
//
//__constant__ extern DTYPE Q3_fu_1[1];// = {-1./3};
//__constant__ extern DTYPE Q3_fp[2];// = {-3./8,-9./8};
//__constant__ extern DTYPE Q3_fu_2[3];// = {-1./12,4./9,1./12};
//__constant__ extern DTYPE Q3_scl[2];// = {3./sqrt(2.),sqrt(2.)/3};
//
//__constant__ extern DTYPE Q5_fu_1[1];
//__constant__ extern DTYPE Q5_fp[2];
//__constant__ extern DTYPE Q5_fu_2[5];
//__constant__ extern DTYPE Q5_scl[2];
//
//__constant__ extern DTYPE L4_fp[2];
//__constant__ extern DTYPE L4_fu[4];
//__constant__ extern DTYPE L4_scl[2];
//
//__constant__ extern DTYPE L6_fp[2];
//__constant__ extern DTYPE L6_fu[6];
//__constant__ extern DTYPE L6_scl[2];
//
//__constant__ extern DTYPE L6_fu_D_LEVEL1[6];
//__constant__ extern DTYPE L6_fu_D_LEVEL0[2];
//
//__constant__ extern DTYPE C4_fu_1[2]/* = { -1. / 4, -1. / 4 }*/;                                    //[0] update filter
//__constant__ extern DTYPE C4_fp[2]/* = { -1.0, -1.0 }*/;                                            //[1] predict filter 
//__constant__ extern DTYPE C4_fu_2[4]/* = { -0.03906250, 0.22656250, 0.22656250, -0.03906250 }*/;    //[1] update  filter
//__constant__ extern DTYPE C4_scl[2] /*= { 2.*sqrt(2.), 1 / (2.*sqrt(2.)) }*/;                       //scale   filter
//
//__constant__ extern DTYPE Q5_CONV_A[12];
//__constant__ extern DTYPE Q5_CONV_D[4];
//
//__constant__ extern DTYPE QN5_CONV_LEVEL1[16];
//
//__constant__ extern DTYPE QN5_CONV_A_LEVEL0[2];
//__constant__ extern DTYPE QN5_CONV_D_LEVEL0[2];
//
//__constant__ extern DTYPE QD5_CONV_LEVEL1[16];
//__constant__ extern DTYPE QD5_CONV_LEVEL0[4];
//
//__constant__ extern DTYPE Q5_iCONV_A[11];
//__constant__ extern DTYPE Q5_iCONV_D[11];
//
//__constant__ extern DTYPE QN5_iCONV_LEVEL1[16];
//
//__constant__ extern DTYPE QN5_iCONV_A_LEVEL0[2];
//__constant__ extern DTYPE QN5_iCONV_D_LEVEL0[2];
//
//__constant__ extern DTYPE QD5_iCONV_LEVEL1[16];
//__constant__ extern DTYPE QD5_iCONV_LEVEL0[4];
//
//__constant__ extern DTYPE L6_CONV_A[13];
//__constant__ extern DTYPE L6_CONV_D[3];
//
//__constant__ extern DTYPE LD6_CONV_A1_LEVEL0[3];
//__constant__ extern DTYPE LD6_CONV_D_LEVEL0[3];
//__constant__ extern DTYPE LD6_CONV_A2_LEVEL0[3];
//
//__constant__ extern DTYPE LD6_CONV_LEVEL1[25];
//__constant__ extern DTYPE LD6_CONV_LEVEL0[9];
//
//__constant__ extern DTYPE LN6_CONV_LEVEL1[25];
//__constant__ extern DTYPE LN6_CONV_LEVEL0[9];
//
//__constant__ extern DTYPE L6_iCONV_A[11];
//__constant__ extern DTYPE L6_iCONV_D[13];
//
//__constant__ extern DTYPE LD6_iCONV_LEVEL0[9];
//
//__constant__ extern DTYPE LN6_iCONV_LEVEL1[25];
//__constant__ extern DTYPE LN6_iCONV_LEVEL0[9];
//
//__constant__ extern DTYPE C4_CONV_A[11];
//__constant__ extern DTYPE C4_CONV_D[5];
//
//__constant__ extern DTYPE CD4_CONV_LEVEL1[25];
//__constant__ extern DTYPE CD4_CONV_LEVEL0[9];
//
//__constant__ extern DTYPE CN4_CONV_LEVEL1[25];
//__constant__ extern DTYPE CN4_CONV_LEVEL0[9];
//
//__constant__ extern DTYPE C4_iCONV_A[11];
//__constant__ extern DTYPE C4_iCONV_D[9];
//
//__constant__ extern DTYPE CD4_iCONV_LEVEL1[25];
//__constant__ extern DTYPE CD4_iCONV_LEVEL0[9];
//
//__constant__ extern DTYPE CN4_iCONV_LEVEL1[25];
//__constant__ extern DTYPE CN4_iCONV_LEVEL0[9];
//
//__constant__ extern DTYPE CUDA_SQRT_2;
//
////__constant__ extern unsigned int devDims[3];
////__constant__ extern unsigned int devLogicDims[3];
////__constant__ extern unsigned int devOffsets[3];
//
//__constant__ extern unsigned int g_pPow2[31];
//
//#endif