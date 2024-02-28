#ifndef __WAVELETTYPES_H__
#define __WAVELETTYPES_H__

#include <string>
#include "cudaGlobal.cuh"

enum class WaveletType :unsigned int
{
	WT_CDF3_9,
	WT_CDF3_11,
	WT_CDF3_5,
	WT_CDF2_10,
	WT_CDF2_12,
	WT_CDF2_6,
	WT_CDF4_4,
	WT_CDF4_6,
	WT_CDF4_8,
	WT_CDF4_10,
	WT_CDF2_8,
	WT_CDF3_7,
	WT_CDF26_44,
	WT_CDF28_46,
	WT_CDF44_26,
	WT_CDF46_28,
	WT_CDF2_0,
	WT_CDF1_1,
	WT_CDF1_7
};

struct WaveletType2
{
	WaveletType x, y;
};

static __host__ __device__ WaveletType2 make_WaveletType2(WaveletType x, WaveletType y)
{
	WaveletType2 t; t.x = x; t.y = y; return t;
}

constexpr unsigned int inline switch_pair(WaveletType a, char f) {
	return (static_cast<unsigned int>(a) << 16) + f;
}

constexpr unsigned int inline switch_pair(WaveletType a, int f) {
	return (static_cast<unsigned int>(a) << 16) + f;
}

constexpr unsigned int inline switch_pair(WaveletType a, char f, char side) {
	return (static_cast<unsigned int>(a) << 16) + (static_cast<unsigned int>(f) << 8) + side;
}

constexpr unsigned int inline switch_pair(WaveletType a, char f, char side, bool forward) {
	return (static_cast<unsigned int>(a) << 16) + (static_cast<unsigned int>(f) << 8) + (static_cast<unsigned int>(forward) << 24) + side;
}

constexpr unsigned int inline switch_pair(WaveletType a, WaveletType b, char f, char side) {
	return (static_cast<unsigned int>(a) << 16) + (static_cast<unsigned int>(f) << 8) + (static_cast<unsigned int>(b) << 24) + side;
}

const WaveletType type_from_string(std::string str);

const inline unsigned int uitype_from_string(std::string str)
{
	return switch_pair(type_from_string(str), str[0]);
}

const inline std::string string_from_type(WaveletType a, char bc)
{
	switch (a)
	{
	case WaveletType::WT_CDF3_5:
		return std::string(1, bc) + std::string("_cdf3.5");
	case WaveletType::WT_CDF3_7:
		return std::string(1, bc) + std::string("_cdf3.7");
	case WaveletType::WT_CDF3_9:
		return std::string(1, bc) + std::string("_cdf3.9");
	case WaveletType::WT_CDF3_11:
		return std::string(1, bc) + std::string("_cdf3.11");
	case WaveletType::WT_CDF2_6:
		return std::string(1, bc) + std::string("_cdf2.6");
	case WaveletType::WT_CDF2_8:
		return std::string(1, bc) + std::string("_cdf2.8");
	case WaveletType::WT_CDF2_10:
		return std::string(1, bc) + std::string("_cdf2.10");
	case WaveletType::WT_CDF2_12:
		return std::string(1, bc) + std::string("_cdf2.12");
	case WaveletType::WT_CDF4_4:
		return std::string(1, bc) + std::string("_cdf4.4");
	case WaveletType::WT_CDF4_6:
		return std::string(1, bc) + std::string("_cdf4.6");
	case WaveletType::WT_CDF4_8:
		return std::string(1, bc) + std::string("_cdf4.8");
	case WaveletType::WT_CDF4_10:
		return std::string(1, bc) + std::string("_cdf4.10");
	case WaveletType::WT_CDF2_0:
		return std::string(1, bc) + std::string("_cdf2.0");
	case WaveletType::WT_CDF1_1:
		return std::string(1, bc) + std::string("_cdf1.1");
	}
}

constexpr char inline proj_bc(char bc)
{
	//if (bc == 'p')
	//{
	//	return 'p';
	//}
	//return bc == 'n' ? 'd' : 'n';
	if (bc == 'n')
	{
		return 'd';
	}
	else if(bc == 'd')
	{
		return 'n';
	}
	return bc;
}

constexpr WaveletType inline proj_WaveletType(WaveletType type)
{
	switch (type)
	{
	case WaveletType::WT_CDF3_9:
		return WaveletType::WT_CDF2_10;
	case WaveletType::WT_CDF3_11:
		return WaveletType::WT_CDF2_12;
	case WaveletType::WT_CDF3_5:
		return WaveletType::WT_CDF2_6;
	case WaveletType::WT_CDF4_4:
		return WaveletType::WT_CDF3_5;
	case WaveletType::WT_CDF4_6:
		return WaveletType::WT_CDF3_7;
	case WaveletType::WT_CDF3_7:
		return WaveletType::WT_CDF2_8;
	case WaveletType::WT_CDF4_8:
		return WaveletType::WT_CDF3_9;
	case WaveletType::WT_CDF4_10:
		return WaveletType::WT_CDF3_11;
	case WaveletType::WT_CDF2_0:
		return WaveletType::WT_CDF1_1;
	}
}

typedef void (*pidx_ad_func)(int&, int&, DTYPE&, DTYPE&, int);
typedef void (*pupdate_ad_func)(DTYPE&, DTYPE&, DTYPE, DTYPE);

typedef DTYPE(*pupdate_l1_func)(DTYPE*, int, int, int);
typedef DTYPE(*pupdate_l10_func)(DTYPE, int);

#endif