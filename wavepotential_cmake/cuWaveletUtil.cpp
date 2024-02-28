#include "cuWaveletUtil.h"
#include <cmath>
#include <cstdio>

int myLog2(double number)
{
    return (int)(log(number) / log(2.));
}

int CalculateNP2TotalLen(int majorLen, int extent, int extentA, int extentD)
{
    if (majorLen == 1)
    {
        return 1;
    }

    int majorDimLevel = int(myLog2((double)majorLen - extent - 0.1)) + 1;

    int currentAN = majorLen;
    int totalMajorN = 0;
    int nextAN = currentAN / 2 + extentA;
    int nextDN = nextAN - extentA + extentD;

    for (int i = 0; i < majorDimLevel; i++)
    {
        int idxD1Level = majorDimLevel - i;

        if (currentAN + 12 <= (int)pow(2., double(idxD1Level)) + extent)
        {
            totalMajorN += nextDN;
            currentAN = nextAN;
            nextAN = currentAN / 2 + extentA;
            nextDN = nextAN - extentA + extentD;
        }
        else
        {
            totalMajorN += (int)pow(2., double(idxD1Level)) + extent;
            break;
        }
    }

    //printf("totalMajorN: %d\n", totalMajorN);
    return totalMajorN;
}

int CalculateNP2LevelLen(int majorLen, int level, int extent, int extentA, int extentD)
{
    int majorDimLevel = int(myLog2((double)majorLen - extent - 0.1)) + 1;

    int currentAN = majorLen;
    int totalMajorN = currentAN;
    int nextAN = currentAN / 2 + extentA;
    int nextDN = nextAN - extentA + extentD;

    for (int i = 0; i < level; i++)
    {
        int idxD1Level = majorDimLevel - i;

        if (currentAN + 12 <= (int)pow(2., double(idxD1Level)) + extent)
        {
            totalMajorN = nextAN;
            currentAN = nextAN;
            nextAN = currentAN / 2 + extentA;
            nextDN = nextAN - extentA + extentD;
        }
        else
        {
            totalMajorN = (int)pow(2., double(idxD1Level)) + extent;
        }
    }

    //printf("totalMajorN: %d\n", totalMajorN);
    return totalMajorN;
}

int CalculateNP2LevelLen_new(int majorLen, int level, int extent, int extentA, int extentD)
{
    int majorDimLevel = int(myLog2((double)majorLen - extent - 0.1)) + 1;

    int currentAN = majorLen;
    int totalMajorN = currentAN;
    int nextAN = currentAN / 2 + extentA;
    int nextDN = nextAN - extentA + extentD;

    for (int i = 0; i <= level; i++)
    {
        int idxD1Level = majorDimLevel - i/* - 1*/;

        if (currentAN + 12 <= (int)pow(2., double(idxD1Level)) + extent)
        {
            totalMajorN = currentAN;
            currentAN = nextAN;
            nextAN = currentAN / 2 + extentA;
            nextDN = nextAN - extentA + extentD;
        }
        else
        {
            totalMajorN = (int)pow(2., double(idxD1Level)) + extent;
        }
    }

    //printf("totalMajorN: %d\n", totalMajorN);
    return totalMajorN;
}

int CalculateNP2LevelDN(int majorLen, int level, int extent, int extentA, int extentD)
{
    int majorDimLevel = int(myLog2((double)majorLen - extent - 0.1)) + 1;

    int currentAN = majorLen;
    int totalMajorN = currentAN;
    int nextAN = currentAN / 2 + extentA;
    int nextDN = nextAN - extentA + extentD;

    for (int i = 0; i <= level; i++)
    {
        int idxD1Level = majorDimLevel - i;

        if (currentAN + 12 <= (int)pow(2., double(idxD1Level)) + extent)
        {
            totalMajorN = nextDN;
            currentAN = nextAN;
            nextAN = currentAN / 2 + extentA;
            nextDN = nextAN - extentA + extentD;
        }
        else
        {
            totalMajorN = (int)pow(2., double(idxD1Level)) / 2;
        }
    }

    //printf("totalMajorN: %d\n", totalMajorN);
    return totalMajorN;
}

bool GetIsP2(int majorLen, int level, int extent, int extentA, int extentD)
{
    int majorDimLevel = int(myLog2((double)majorLen - extent - 0.1)) + 1;

    int currentAN = majorLen;
    //int totalMajorN = currentAN;
    int nextAN = currentAN / 2 + extentA;
    bool flag = false;
    for (int i = 0; i <= level; i++)
    {
        int idxD1Level = majorDimLevel - i;

        //printf("currentAN : %d, pow2 : %d\n", currentAN, (int)pow(2., double(idxD1Level)));
        if (currentAN + 12 <= (int)pow(2., double(idxD1Level)) + extent)
        {
            currentAN = nextAN;
            nextAN = currentAN / 2 + extentA;
        }
        else
        {
            return true;
        }
    }

    //printf("totalMajorN: %d\n", totalMajorN);
    return flag;
}