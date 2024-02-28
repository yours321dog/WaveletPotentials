#ifndef __CUWAVELETUTIL_H__
#define __CUWAVELETUTIL_H__

int myLog2(double number);

int CalculateNP2TotalLen(int majorLen, int extent = 0, int extentA = 4, int extentD = 2);

int CalculateNP2LevelLen(int majorLen, int level, int extent = 0, int extentA = 4, int extentD = 2);

int CalculateNP2LevelLen_new(int majorLen, int level, int extent = 0, int extentA = 4, int extentD = 2);

int CalculateNP2LevelDN(int majorLen, int level, int extent = 0, int extentA = 4, int extentD = 2);

bool GetIsP2(int majorLen, int level, int extent = 0, int extentA = 4, int extentD = 2);

#endif
