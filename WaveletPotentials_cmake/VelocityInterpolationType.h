#ifndef __VELOCITYINTERPOLATIONTYPE_H__
#define __VELOCITYINTERPOLATIONTYPE_H__

enum class VelInterplateType :unsigned int
{
    VIT_DIRECT,
    VIT_SWEEP,
    VIT_SWEEPDST,
    VIT_WAVELET,
    VIT_DEFINED_SIN_3D,
    VIT_DEFINED_SIN_2D,
    VIT_WAVELET_LINEAR,
    VIT_MONOTONIC_CUBIC
};

inline char MapVIT2Char(VelInterplateType vi_type)
{
    switch (vi_type)
    {
    case VelInterplateType::VIT_DIRECT:
        return 'd';
    case VelInterplateType::VIT_SWEEP:
        return 's';
    case VelInterplateType::VIT_SWEEPDST:
        return 'q';
    case VelInterplateType::VIT_WAVELET:
        return 'w';
    case VelInterplateType::VIT_DEFINED_SIN_3D:
        return '3';
    case VelInterplateType::VIT_DEFINED_SIN_2D:
        return '2';
    case VelInterplateType::VIT_WAVELET_LINEAR:
        return 'l';
    case VelInterplateType::VIT_MONOTONIC_CUBIC:
        return 'm';
    default:
        return '0';
    }
}

#endif
