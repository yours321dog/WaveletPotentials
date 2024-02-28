#include "DivL0Matrix.h"

void GetDivL0Matrix(DTYPE* dst, int2 levels)
{
    int min_level = std::min(levels.x, levels.y);
    int2 min_levels = { levels.x - min_level, levels.y - min_level };

	DTYPE4 res;
	switch (switch_pair(std::max(min_levels.x, min_levels.y), 0))
	{
	case switch_pair(1, 0):
		res = { -0.600000000000000, -0.400000000000000, -0.450000000000000, -0.050000000000000 };
		break;
	case switch_pair(2, 0):
		res = { -1.058823529411764, -0.941176470588235, -0.485294117647059, -0.014705882352941 };
		break;
	case switch_pair(3, 0):
		res = { -2.030769230769231, -1.969230769230771, -0.496153846153847, -0.003846153846153 };
		break;
	case switch_pair(4, 0):
		res = { -4.015564202334636, -3.984435797665360, -0.499027237354084, -0.000972762645915 };
		break;
	case switch_pair(5, 0):
		res = { -8.007804878048813 - 7.992195121951190 - 0.499756097560973 - 0.000243902439025 };
		break;
	case switch_pair(6, 0):
		res = { -16.003905296558496, -15.996094703441514, -0.499938979741274, -0.000061020258726 };
		break;
	case switch_pair(7, 0):
		res = { -32.001953005797915, -31.998046994202070, -0.499984742142203, -0.000015257857796 };
		break;
	case switch_pair(8, 0):
		res = { -64.000976547601056, -63.999023452398980, -0.499996185360941, -0.000003814639067 };
		break;
	default:
		res = { -0.374999999999990, -0.124999999999991, -0.375000000000009, -0.125000000000009 };
		break;
	}

	if (min_levels.x < min_levels.y)
	{
		std::swap(res.x, res.z);
		std::swap(res.y, res.w);
	}

	dst[0] = res.x; dst[1] = res.y;	dst[2] = res.z;	dst[3] = res.w;
	dst[4] = -res.x; dst[5] = -res.y; dst[6] = res.w; dst[7] = res.z;
	dst[8] = res.y; dst[9] = res.x; dst[10] = -res.z; dst[11] = -res.w;
	dst[12] = -res.y; dst[13] = -res.x; dst[14] = -res.w; dst[15] = -res.z;
}
