#include "WaveletTypes.h"
#include <iostream>

const WaveletType type_from_string(std::string str)
{
	if (str.substr(2).compare("cdf3.5") == 0)
	{
		return WaveletType::WT_CDF3_5;
	}
	else if (str.substr(2).compare("cdf2.6") == 0)
	{
		return WaveletType::WT_CDF2_6;
	}
	else if (str.substr(2).compare("cdf4.4") == 0)
	{
		return WaveletType::WT_CDF4_4;
	}
	else if (str.substr(2).compare("cdf4.6") == 0)
	{
		return WaveletType::WT_CDF4_6;
	}
	else if (str.substr(2).compare("cdf3.7") == 0)
	{
		return WaveletType::WT_CDF3_7;
	}
	else if (str.substr(2).compare("cdf2.8") == 0)
	{
		return WaveletType::WT_CDF2_8;
	}
	else if (str.substr(2).compare("cdf3.9") == 0)
	{
		return WaveletType::WT_CDF3_9;
	}
	else if (str.substr(2).compare("cdf3.11") == 0)
	{
		return WaveletType::WT_CDF3_11;
	}
	else if (str.substr(2).compare("cdf2.10") == 0)
	{
		return WaveletType::WT_CDF2_10;
	}
	else if (str.substr(2).compare("cdf2.12") == 0)
	{
		return WaveletType::WT_CDF2_12;
	}
	else if (str.substr(2).compare("cdf4.8") == 0)
	{
		return WaveletType::WT_CDF4_8;
	}
	else if (str.substr(2).compare("cdf4.10") == 0)
	{
		return WaveletType::WT_CDF4_10;
	}
	else if (str.substr(2).compare("cdf2.0") == 0)
	{
		return WaveletType::WT_CDF2_0;
	}
	else if (str.substr(2).compare("cdf1.1") == 0)
	{
		return WaveletType::WT_CDF1_1;
	}
	else if (str.substr(2).compare("cdf1.7") == 0)
	{
		return WaveletType::WT_CDF1_7;
	}
	return WaveletType::WT_CDF3_5;
}