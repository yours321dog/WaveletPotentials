#include "Utils.h"
#include "cuWaveletUtil.h"
#include "math.h"
#include <sys/stat.h>
#include <fstream>
#include <random>
#include <iomanip>
void PrintMat(C_DTYPE* mat, int3 dim)
{
    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                printf("%f ", mat[i + j * dim.y + k * dim.x * dim.y]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}


void PrintMat(C_DTYPE* mat, int2 dim)
{
    for (int j = 0; j < dim.x; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            printf("%f ", mat[i + j * dim.y]);
        }
        printf("\n");
    }
    printf("\n");
}

void PrintMat(C_DTYPE* mat, int3 dim_dst, int3 dim)
{
    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                printf("%f ", mat[i + j * dim_dst.y + k * dim_dst.x * dim_dst.y]);
            }
            printf("\n");
        }
    }
    printf("\n");
}

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int size)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int i = 0; i < size; i++)
    {
        err += abs(value1[i] - value2[i]);
        //if (abs((value1[i] - value2[i]) / (abs(value1[i]) + 1e-12) ) > 0.001)
        //{
        //    printf("i : %d, values1 : %f, values2 : %f\n", i, value1[i], value2[i]);
        //    //system("pause");
        //}
        valueTotle += abs(value1[i]);
    }

    return err / valueTotle;
}

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err += abs(value1[idx] - value2[idx]);
                //if (abs((value1[idx] - value2[idx]) / (abs(value1[idx]) + 1e-12)) > 0.001)
                //{
                //    printf("x, y, z : %d, %d, %d, values1 : %f, values2 : %f\n", j, i, k, value1[idx], value2[idx]);
                //    //system("pause");
                //}
                valueTotle += abs(value1[idx]);
            }
        }

    }

    return err / valueTotle;
}

std::pair<DTYPE, DTYPE> CalculateErrorPair(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err += abs(value1[idx] - value2[idx]);
                //if (abs((value1[idx] - value2[idx]) / (abs(value1[idx]) + 1e-12)) > 0.001)
                //{
                //    printf("x, y, z : %d, %d, %d, values1 : %f, values2 : %f\n", j, i, k, value1[idx], value2[idx]);
                //    //system("pause");
                //}
                valueTotle += abs(value1[idx]);
            }
        }

    }

    return std::make_pair(err, valueTotle);
}

DTYPE CalculateError(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3)
{
    std::pair<DTYPE, DTYPE> err_1 = CalculateErrorPair(base1, check1, size1);
    std::pair<DTYPE, DTYPE> err_2 = CalculateErrorPair(base2, check2, size1);
    std::pair<DTYPE, DTYPE> err_3 = CalculateErrorPair(base3, check3, size1);
    return (err_1.first + err_2.first + err_3.first) / (err_1.second + err_2.second + err_3.second + eps<DTYPE>);
}


DTYPE CalculateErrorL2(C_DTYPE* value1, C_DTYPE* value2, int2 size2)
{
    return CalculateErrorL2(value1, value2, make_int3(size2.x, size2.y, 1));
}

std::pair<DTYPE, DTYPE> CalculateErrorL2Pair(C_DTYPE* value1, C_DTYPE* value2, int2 size2)
{
    return CalculateErrorL2Pair(value1, value2, make_int3(size2.x, size2.y, 1));
}

DTYPE CalculateErrorMax(C_DTYPE* value1, C_DTYPE* value2, int2 size2)
{
    return CalculateErrorMax(value1, value2, make_int3(size2.x, size2.y, 1));
}

DTYPE CalculateErrorRelativeMax(C_DTYPE* value1, C_DTYPE* value2, int2 size2)
{
    return CalculateErrorRelativeMax(value1, value2, make_int3(size2.x, size2.y, 1));
}

DTYPE CalculateErrorL2(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2)
{
    std::pair<DTYPE, DTYPE> err_1 = CalculateErrorL2Pair(base1, check1, size1);
    std::pair<DTYPE, DTYPE> err_2 = CalculateErrorL2Pair(base2, check2, size1);
    return std::sqrt((err_1.first + err_2.first) / (err_1.second + err_2.second + eps<DTYPE>));
}

DTYPE CalculateErrorMax(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2)
{
    DTYPE err_1 = CalculateErrorMax(base1, check1, size1);
    DTYPE err_2 = CalculateErrorMax(base2, check2, size1);
    return fmaxf(err_1, err_2);
}

DTYPE CalculateErrorRelativeMax(C_DTYPE* base1, C_DTYPE* check1, int2 size1,
    C_DTYPE* base2, C_DTYPE* check2, int2 size2)
{
    DTYPE err_1 = CalculateErrorRelativeMax(base1, check1, size1);
    DTYPE err_2 = CalculateErrorRelativeMax(base2, check2, size1);
    return fmaxf(err_1, err_2);
}


DTYPE CalculateErrorL2(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err += (value1[idx] - value2[idx]) * (value1[idx] - value2[idx]);
                //if (abs((value1[idx] - value2[idx]) / (abs(value1[idx]) + 1e-12)) > 0.001)
                //{
                //    printf("x, y, z : %d, %d, %d, values1 : %f, values2 : %f\n", j, i, k, value1[idx], value2[idx]);
                //    //system("pause");
                //}
                valueTotle += (value1[idx]) * (value1[idx]);
            }
        }

    }

    return std::sqrt(err / valueTotle);
}

DTYPE CalculateErrorMax(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err = fmaxf(abs(value1[idx] - value2[idx]), err);

            }
        }
    }
    return err;
}

DTYPE CalculateErrorMaxL2(DTYPE2* value1, DTYPE2* value2, int2 size3)
{
    DTYPE err = 0.;

    for (int j = 0; j < size3.x; j++)
    {
        for (int i = 0; i < size3.y; i++)
        {
            int idx = i + j * size3.y;
            err = fmaxf(length(value1[idx] - value2[idx]), err);
        }
    }

    return err;
}

DTYPE CalculateErrorMaxL2(DTYPE3* value1, DTYPE3* value2, int3 size3)
{
    DTYPE err = 0.;
    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = INDEX3D(i, j, k, size3);
                err = fmaxf(length(value1[idx] - value2[idx]), err);
            }
        }
    }

    return err;
}

DTYPE CalculateErrorRelativeMax(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;
    DTYPE err_max = 0.f;
    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err_max = fmaxf((abs(value1[idx]) + eps<DTYPE>), err_max) ;

            }
        }
    }

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err = fmaxf(abs(value1[idx] - value2[idx]) / err_max, err);

            }
        }
    }
    return err;
}

std::pair<DTYPE, DTYPE> CalculateErrorL2Pair(C_DTYPE* value1, C_DTYPE* value2, int3 size3)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int k = 0; k < size3.z; k++)
    {
        for (int j = 0; j < size3.x; j++)
        {
            for (int i = 0; i < size3.y; i++)
            {
                int idx = i + j * size3.y + k * size3.x * size3.y;
                err += (value1[idx] - value2[idx]) * (value1[idx] - value2[idx]);
                //if (abs((value1[idx] - value2[idx]) / (abs(value1[idx]) + 1e-12)) > 0.001)
                //{
                //    printf("x, y, z : %d, %d, %d, values1 : %f, values2 : %f\n", j, i, k, value1[idx], value2[idx]);
                //    //system("pause");
                //}
                valueTotle += (value1[idx]) * (value1[idx]);
            }
        }

    }

    return std::make_pair(err, valueTotle);
}

DTYPE CalculateErrorL2(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3)
{
    std::pair<DTYPE, DTYPE> err_1 = CalculateErrorL2Pair(base1, check1, size1);
    std::pair<DTYPE, DTYPE> err_2 = CalculateErrorL2Pair(base2, check2, size2);
    std::pair<DTYPE, DTYPE> err_3 = CalculateErrorL2Pair(base3, check3, size3);
    return std::sqrt((err_1.first + err_2.first + err_3.first) / (err_1.second + err_2.second + err_3.second + eps<DTYPE>));
}

DTYPE CalculateErrorMax(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3)
{
    DTYPE err_1 = CalculateErrorMax(base1, check1, size1);
    DTYPE err_2 = CalculateErrorMax(base2, check2, size1);
    DTYPE err_3 = CalculateErrorMax(base3, check3, size1);
    return fmaxf(err_1, fmaxf(err_2, err_3));
}

DTYPE CalculateErrorRelativeMax(C_DTYPE* base1, C_DTYPE* check1, int3 size1,
    C_DTYPE* base2, C_DTYPE* check2, int3 size2,
    C_DTYPE* base3, C_DTYPE* check3, int3 size3)
{
    DTYPE err_1 = CalculateErrorRelativeMax(base1, check1, size1);
    DTYPE err_2 = CalculateErrorRelativeMax(base2, check2, size1);
    DTYPE err_3 = CalculateErrorRelativeMax(base3, check3, size1);
    return fmaxf(err_1, fmaxf(err_2, err_3));
}

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, int3 dim, int3 dim_dst)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                int idx = i + j * dim_dst.y + k * dim_dst.x * dim_dst.y;
                err += abs(value1[idx] - value2[idx]);
                //if (abs((value1[idx] - value2[idx])/* / value1[i]*/) > 0.001)
                //{
                //    printf("x, y, z : %d, %d, %d, values1 : %f, values2 : %f\n", j, i, k, value1[idx], value2[idx]);
                //    //system("pause");
                //}
                valueTotle += abs(value1[idx]);
            }
        }

    }

    return err / valueTotle;
}

DTYPE CalculateErrorWithPrint(C_DTYPE* value1, C_DTYPE* value2, int size)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    for (int i = 0; i < size; i++)
    {
        err += abs(value1[i] - value2[i]);
        //if (abs(value1[i] - value2[i]) > 0.00001)
        //if (i < 10)
        {
            printf("i : %d, values1 : %f, values2 : %f\n", i, value1[i], value2[i]);
        }
        valueTotle += abs(value1[i]);
    }

    return err / valueTotle;
}

DTYPE CalculateError(C_DTYPE* value1, C_DTYPE* value2, C_INT* size1, C_INT* size2, int extent)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    int mt1X = CalculateNP2TotalLen(size1[1], extent);
    int mt1Y = CalculateNP2TotalLen(size1[0], extent);

    int mt2X = CalculateNP2TotalLen(size2[1], extent);
    int mt2Y = CalculateNP2TotalLen(size2[0], extent);

    for (int k = 0; k < size2[2]; k++)
    {
        for (int j = 0; j < size2[1]; j++)
        {
            for (int i = 0; i < size2[0]; i++)
            {
                int idx1 = i + j * mt1Y + k * mt1Y * mt1X;
                int idx2 = i + j * mt2Y + k * mt2Y * mt2X;

                err += abs(value1[idx1] - value2[idx2]);
                if (abs(value1[idx1] - value2[idx2]) > 0.000002)
                {
                    printf("i : %d, j : %d, idx1 : %d, values1 : %f, idx2 : %d, values2 : %f\n", i, j, idx1, value1[idx1], idx2, value2[idx2]);
                }
                valueTotle += abs(value1[idx1]);
            }
        }
    }

    return err / valueTotle;
}

DTYPE CalculateErrorWithPrint(C_DTYPE* value1, C_DTYPE* value2, C_INT* size1, C_INT* size2, int extent)
{
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;

    int mt1X = CalculateNP2TotalLen(size1[1], extent);
    int mt1Y = CalculateNP2TotalLen(size1[0], extent);

    int mt2X = CalculateNP2TotalLen(size2[1], extent);
    int mt2Y = CalculateNP2TotalLen(size2[0], extent);

    for (int k = 0; k < size2[2]; k++)
    {
        for (int j = 0; j < size2[1]; j++)
        {
            for (int i = 0; i < size2[0]; i++)
            {
                int idx1 = i + j * mt1Y + k * mt1Y * mt1X;
                int idx2 = i + j * mt2Y + k * mt2Y * mt2X;

                err += abs(value1[idx1] - value2[idx2]);
                /*if (abs(value1[idx1] - value2[idx2]) > 0.00001) */printf("idx1 : %d, values1 : %f, idx2 : %d, values2 : %f\n", idx1, value1[idx1], idx2, value2[idx2]);
                valueTotle += abs(value1[idx1]);
            }
        }
    }

    return err / valueTotle;
}

DTYPE CalculateErrorWithoutMean(C_DTYPE* value1, C_DTYPE* value2, int size)
{
    DTYPE aver1 = 0.;
    DTYPE aver2 = 0.;
    DTYPE* tmp1 = new DTYPE[size];
    DTYPE* tmp2 = new DTYPE[size];

    for (int i = 0; i < size; i++)
    {
        aver1 += value1[i];
        aver2 += value2[i];

        tmp1[i] = value1[i];
        tmp2[i] = value2[i];
    }
    aver1 /= size;
    aver2 /= size;
    for (int i = 0; i < size; i++)
    {
        tmp1[i] -= aver1;
        tmp2[i] -= aver2;
    }

    DTYPE err = CalculateError(tmp1, tmp2, size);
    delete[] tmp1;
    delete[] tmp2;

    return err;
}

DTYPE CalculateErrorNP2(C_DTYPE* value, C_DTYPE* np2Value, C_INT* pDims, C_INT* np2Dims, char direction, int extent)
{
    int totalMajorN;
    int valueIdx;
    int np2ValueIdx;
    int currentAN;
    int nextAN;
    int nextDN;
    int valueLevelPow2;
    int valueStride;
    DTYPE err = 0.;
    DTYPE valueTotle = 0.;
    switch (direction)
    {
    case 'X':
    case 'x':
        totalMajorN = CalculateNP2TotalLen(np2Dims[1], extent);
        //printf("totalMajorN : %d\n", totalMajorN);
        for (int k = 0; k < np2Dims[2]; k++)
        {
            currentAN = np2Dims[1];
            nextAN = currentAN / 2 + 4;
            nextDN = currentAN / 2 + 2;
            valueLevelPow2 = (int)pow(2., myLog2(np2Dims[1] - extent - 0.1));
            valueStride = 0;
            for (int j = 0; j < totalMajorN; j++)
            {
                for (int i = 0; i < np2Dims[0]; i++)
                {
                    np2ValueIdx = i + j * np2Dims[0] + k * np2Dims[0] * totalMajorN;
                    valueIdx = i + (j + valueStride) * pDims[0] + k * pDims[0] * pDims[1];
                    //printf("valueIdx: %d, np2ValueIdx: %d, value: %f, np2Value: %f\n", valueIdx, np2ValueIdx, value[valueIdx], np2Value[np2ValueIdx]);
                    if (abs(value[valueIdx] - np2Value[np2ValueIdx]) > 0.0001)
                    {
                        printf("valueIdx: %d, np2ValueIdx: %d, value: %f, np2Value: %f\n", valueIdx, np2ValueIdx, value[valueIdx], np2Value[np2ValueIdx]);
                    }
                    err += abs(value[valueIdx] - np2Value[np2ValueIdx]);
                    valueTotle += abs(value[valueIdx]);
                }
                if (currentAN + 12 <= valueLevelPow2 * 2 + extent)
                {
                    nextDN--;
                    if (nextDN == 0)
                    {
                        //valueStartIdx += valueLevelPow2;
                        valueStride += valueLevelPow2 - nextAN + 2;
                        valueLevelPow2 /= 2;
                        currentAN = nextAN;
                        nextAN = currentAN / 2 + 4;
                        nextDN = nextAN - 2;
                    }
                }
            }
        }
        break;
    default:
        totalMajorN = CalculateNP2TotalLen(np2Dims[0], extent);
        for (int k = 0; k < np2Dims[2]; k++)
        {
            for (int j = 0; j < np2Dims[1]; j++)
            {
                currentAN = np2Dims[0];
                nextAN = currentAN / 2 + 4;
                nextDN = currentAN / 2 + 2;
                valueLevelPow2 = (int)pow(2., myLog2(np2Dims[0] - extent - 0.1));
                valueStride = 0;
                for (int i = 0; i < totalMajorN; i++)
                {
                    np2ValueIdx = i + j * totalMajorN + k * np2Dims[1] * totalMajorN;
                    valueIdx = i + valueStride + j * pDims[0] + k * pDims[0] * pDims[1];
                    //printf("valueIdx: %d, np2ValueIdx: %d, value: %f, np2Value: %f\n", valueIdx, np2ValueIdx, value[valueIdx], np2Value[np2ValueIdx]);
                    //if (abs(value[valueIdx] - np2Value[np2ValueIdx]) > 0.0001)
                    //{
                    //    printf("valueIdx: %d, np2ValueIdx: %d, value: %f, np2Value: %f\n", valueIdx, np2ValueIdx, value[valueIdx], np2Value[np2ValueIdx]);
                    //}
                    err += abs(value[valueIdx] - np2Value[np2ValueIdx]);
                    valueTotle += abs(value[valueIdx]);

                    if (currentAN + 12 <= valueLevelPow2 * 2 + extent)
                    {
                        nextDN--;
                        if (nextDN == 0)
                        {
                            //valueStartIdx += valueLevelPow2;
                            valueStride += valueLevelPow2 - nextAN + 2;
                            valueLevelPow2 /= 2;
                            currentAN = nextAN;
                            nextAN = currentAN / 2 + 4;
                            nextDN = nextAN - 2;
                        }
                    }
                }
            }
        }
    }

    return err / valueTotle;
}

void SwapPointer(void** ptr1, void** ptr2)
{
    void* tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

void Array2Ddda(DTYPE* vel_ddda, DTYPE* vel, int levels, int3 dim, char direction, int is_ext)
{
    switch (direction)
    {
    case 'x':
    case 'X':
        for (int l = 0; l < levels; l++)
        {
            int length = 1 << l;
            int stride = 1 << (levels - l);
            int start = (stride >> 1);

            for (int k = 0; k < dim.z; k++)
            {
                for (int j = 0; j < length; j++)
                {
                    for (int i = 0; i < dim.y; i++)
                    {
                        int j_ddda = j + length + is_ext;
                        int idx_ddda = i + j_ddda * dim.y + k * dim.x * dim.y;
                        int j_src = start + j * stride;
                        int idx = i + j_src * dim.y + k * dim.x * dim.y;
                        vel_ddda[idx_ddda] = vel[idx];
                    }
                }
            }
        }
        for (int k = 0; k < dim.z; k++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                int idx_ddda = i + k * dim.x * dim.y;
                int idx = i + k * dim.x * dim.y;
                vel_ddda[idx_ddda] = vel[idx];
                if (is_ext > 0)
                {
                    idx_ddda += dim.y;
                    idx += (dim.x - 1) * dim.y;
                    vel_ddda[idx_ddda] = vel[idx];
                }
            }
        }
        break;
    case 'z':
    case 'Z':
        for (int l = 0; l < levels; l++)
        {
            int length = 1 << l;
            int stride = 1 << (levels - l);
            int start = (stride >> 1);

            for (int k = 0; k < length; k++)
            {
                for (int j = 0; j < dim.x; j++)
                {
                    for (int i = 0; i < dim.y; i++)
                    {
                        int k_ddda = k + length + is_ext;
                        int idx_ddda = i + j * dim.y + k_ddda * dim.x * dim.y;
                        int k_src = start + k * stride;
                        int idx = i + j * dim.y + k_src * dim.x * dim.y;
                        vel_ddda[idx_ddda] = vel[idx];
                    }
                }
            }
        }
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                int idx_ddda = i + j * dim.y;
                int idx = i + j * dim.y;
                vel_ddda[idx_ddda] = vel[idx];
                if (is_ext > 0)
                {
                    idx_ddda += dim.y * dim.x;
                    idx += (dim.z - 1) * dim.y * dim.x;
                    vel_ddda[idx_ddda] = vel[idx];
                }
            }
        }
        break;
    default:
        for (int l = 0; l < levels; l++)
        {
            int length = 1 << l;
            int stride = 1 << (levels - l);
            int start = (stride >> 1);

            for (int k = 0; k < dim.z; k++)
            {
                for (int j = 0; j < dim.x; j++)
                {
                    for (int i = 0; i < length; i++)
                    {
                        int i_ddda = i + length + is_ext;
                        int idx_ddda = i_ddda + j * dim.y + k * dim.x * dim.y;
                        int i_src = start + i * stride;
                        int idx = i_src + j * dim.y + k * dim.x * dim.y;
                        vel_ddda[idx_ddda] = vel[idx];
                    }
                }
            }
        }
        for (int k = 0; k < dim.z; k++)
        {
            for (int j = 0; j < dim.x; j++)
            {
                int idx_ddda = j * dim.y + k * dim.x * dim.y;
                int idx = j * dim.y + k * dim.x * dim.y;
                vel_ddda[idx_ddda] = vel[idx];
                if (is_ext > 0)
                {
                    idx_ddda += 1;
                    idx += dim.y - 1;
                    vel_ddda[idx_ddda] = vel[idx];
                }
            }
        }
        break;
    }
}

void Array2Ddda3D(DTYPE* vel_ddda, DTYPE* vel, int3 levels, int3 dim)
{
    int size = dim.x * dim.y * dim.z;
    DTYPE* tmp = new DTYPE[size];
    Array2Ddda(vel_ddda, vel, levels.x, dim, 'x', dim.x & 1);
    Array2Ddda(tmp, vel_ddda, levels.y, dim, 'y', dim.y & 1);
    Array2Ddda(vel_ddda, tmp, levels.z, dim, 'z', dim.z & 1);
    delete[] tmp;
}

void ReadCsv(std::vector<DTYPE>& res, std::ifstream& fp, int* res_dim)
{
    if (res_dim != nullptr)
    {
        res_dim[0] = 0;
        res_dim[1] = 0;
    }
    std::string line;
    bool is_write_dim_1 = false;
    while (getline(fp, line))
    {
        std::string number;
        std::istringstream readstr(line); //string数据流化
        //将一行数据按'，'分割

        while (getline(readstr, number, ','))
        {
            res.push_back(atof(number.c_str())); //字符串传int
            if (res_dim != nullptr && !is_write_dim_1) res_dim[1]++;
        }
        if (res_dim != nullptr)
        {
            res_dim[0]++;
            is_write_dim_1 = true;
        }
    }
}

void ReadCsv3D(std::vector<DTYPE>& res, std::ifstream& fp, int* res_dim)
{
    std::string line;
    bool is_write_dim_1 = false;
    while (getline(fp, line))
    {
        std::string number;
        std::istringstream readstr(line); //string数据流化
        //将一行数据按'，'分割

        while (getline(readstr, number, ','))
        {
            res.push_back(atof(number.c_str())); //字符串传int
            if (res_dim != nullptr && !is_write_dim_1) res_dim[1]++;
        }
        if (res_dim != nullptr)
        {
            res_dim[0]++;
            is_write_dim_1 = true;
        }
    }
}

void WriteCsvYLine(std::ofstream& fp, int* data, int3 dim)
{
    int slice_xz = dim.x * dim.z;
    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp << data[j * dim.y + i];
            if (i < dim.y - 1)
            {
                fp << ", ";
            }
            else
            {
                fp << std::endl;
            }
        }
    }
}

void WriteDataTxt(std::ofstream& fp, DTYPE* data, int3 dim)
{
    int slice_xz = dim.x * dim.z;
    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp << data[j * dim.y + i];
            if (i < dim.y - 1)
            {
                fp << " ";
            }
            else
            {
                fp << std::endl;
            }
        }
    }
}

void WriteDataTxt(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ofstream fp(fp_name);
    fp.setf(std::ios::fixed);
    fp.precision(12);
    int slice_xz = dim.x * dim.z;

    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp << data[j * dim.y + i];
            if (i < dim.y - 1)
            {
                fp << " ";
            }
            else
            {
                fp << '\n';
            }
        }
    }
    fp.precision(6);
}

void WriteDataTxt_sparse(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ofstream fp(fp_name);
    int expect_precision = 6;
    //fp.setf(std::ios::fixed);
    //fp.precision(expect_precision);
    fp << std::scientific << std::setprecision(5);
    float out_thredshold = std::pow(10, -expect_precision);
    int slice_xz = dim.x * dim.z;

    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                DTYPE res = data[INDEX3D(i, j, k, dim)];
                if (abs(res) >= out_thredshold)
                    fp << j << ',' << i << ',' << k << ',' << res << '\n';
            }
        }
    }

    fp.close();
}

void WriteVectorTxt_sparse(const char* fp_name, DTYPE3* data, int3 dim)
{
    std::ofstream fp(fp_name);
    int expect_precision = 6;
    //fp.setf(std::ios::fixed);
    //fp.precision(expect_precision);
    fp << std::scientific << std::setprecision(5);
    float out_thredshold = std::pow(10, -expect_precision);
    int slice_xz = dim.x * dim.z;

    for (int k = 0; k < dim.z; k++)
    {
        for (int j = 0; j < dim.x; j++)
        {
            for (int i = 0; i < dim.y; i++)
            {
                DTYPE3 res = data[INDEX3D(i, j, k, dim)];
                if (length(res) >= out_thredshold)
                    fp << j << ',' << i << ',' << k << ',' << res.x << ',' << res.y << ',' << res.z << '\n';
            }
        }
    }

    fp.close();
}

void WriteDataBinary(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ofstream fp(fp_name);
    int size = getsize(dim);
    fp.write((char*)data, size * sizeof(DTYPE));
}

void WriteDataTxt_nonewline(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ofstream fp(fp_name);
    fp.setf(std::ios::fixed);
    int slice_xz = dim.x * dim.z;
    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp << data[j * dim.y + i];
            if (i < dim.y - 1)
            {
                fp << " ";
            }
            else
            {
                fp << std::endl;
            }
        }
    }
}

void WriteDataETxt(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ofstream fp(fp_name);
    fp << std::scientific << std::setprecision(14);
    int slice_xz = dim.x * dim.z;
    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp << data[j * dim.y + i];
            if (i < dim.y - 1)
            {
                fp << " ";
            }
            else
            {
                fp << std::endl;
            }
        }
    }
}

void ReadsDataTxt(const char* fp_name, DTYPE* data, int3 dim)
{
    std::ifstream fp(fp_name);
    int slice_xz = dim.x * dim.z;
    for (int j = 0; j < slice_xz; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            fp >> data[j * dim.y + i];
        }
    }
}

void WriteDataTxt_sparse_2d(const char* fp_name, DTYPE* data, int2 dim)
{
    std::ofstream fp(fp_name);
    int expect_precision = 6;
    fp << std::scientific << std::setprecision(5);
    float out_thredshold = std::pow(10, -expect_precision);

    for (int j = 0; j < dim.x; j++)
    {
        for (int i = 0; i < dim.y; i++)
        {
            DTYPE res = data[INDEX2D(i, j, dim)];
            if (abs(res) >= out_thredshold)
                fp << j << ',' << i << ',' << res << '\n';
        }
    }

    fp.close();
}

// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
    return x && !(x & (x - 1));
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

DTYPE GenerateRandomDtype(DTYPE min, DTYPE max) 
{
    //static std::random_device rd;
    //static std::mt19937 gen(rd());
    //std::uniform_real_distribution<> dis(min, max);
    //return dis(gen);


    static std::default_random_engine dre;
    static std::uniform_real_distribution<DTYPE> u(min, max);
    static bool first_time = true;
    if (first_time)
    {
        dre.seed(31);
        first_time = false;
    }
    return u(dre);
}