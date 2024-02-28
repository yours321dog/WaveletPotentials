#include "cuRestriction2D.cuh"
#include "cuMemoryManager.cuh"
#include "cudaMath.cuh"
CuRestriction2D::CuRestriction2D() = default;
CuRestriction2D::~CuRestriction2D() = default;

std::auto_ptr<CuRestriction2D> CuRestriction2D::instance_;

__global__ void cuRestriction2D_2nm1(DTYPE * dst, DTYPE * src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = (idx_x + 1) * 2 - 1;
		int idx_y_src = (idx_y + 1) * 2 - 1;

		dst[idx] = 0.0625 * src[INDEX2D(idx_y_src - 1, idx_x_src - 1, dim_src)] +
					0.125 * src[INDEX2D(idx_y_src, idx_x_src - 1, dim_src)] +
				   0.0625 * src[INDEX2D(idx_y_src + 1, idx_x_src - 1, dim_src)] +
					0.125 * src[INDEX2D(idx_y_src - 1, idx_x_src, dim_src)] +
					 0.25 * src[INDEX2D(idx_y_src, idx_x_src, dim_src)] +
					0.125 * src[INDEX2D(idx_y_src + 1, idx_x_src, dim_src)] +
				   0.0625 * src[INDEX2D(idx_y_src - 1, idx_x_src + 1, dim_src)] +
					0.125 * src[INDEX2D(idx_y_src, idx_x_src + 1, dim_src)] +
				   0.0625 * src[INDEX2D(idx_y_src + 1, idx_x_src + 1, dim_src)];
	}
}

__global__ void cuRestriction2D_p2(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = (idx_x + 1) * 2 - 1;
		int idx_y_src = (idx_y + 1) * 2 - 1;

		DTYPE res = 0.0625f * src[INDEX2D(idx_y_src - 1, idx_x_src - 1, dim_src)] +
			0.125f * src[INDEX2D(idx_y_src, idx_x_src - 1, dim_src)] +
			0.125f * src[INDEX2D(idx_y_src - 1, idx_x_src, dim_src)] +
			0.25f * src[INDEX2D(idx_y_src, idx_x_src, dim_src)];

		if (idx_y < dim.y - 1)
		{
			res += 0.0625f * src[INDEX2D(idx_y_src + 1, idx_x_src - 1, dim_src)] + 0.125f * src[INDEX2D(idx_y_src + 1, idx_x_src, dim_src)];
		}
		if (idx_x < dim.x - 1)
		{
			res += 0.0625f * src[INDEX2D(idx_y_src - 1, idx_x_src + 1, dim_src)] + 0.125f * src[INDEX2D(idx_y_src, idx_x_src + 1, dim_src)];
		}
		if (idx_y < dim.y - 1 && idx_x < dim.x - 1)
		{
			res += 0.0625f * src[INDEX2D(idx_y_src + 1, idx_x_src + 1, dim_src)];
		}
		dst[idx] = res;
	}
}

__global__ void cuRestriction2D_p2_neumann(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = (idx_x + 1) * 2 - 1;
		int idx_y_src = (idx_y + 1) * 2 - 1;

		DTYPE lb_coef = 0.0625f;
		DTYPE lm_coef = 0.125f;
		DTYPE lt_coef = 0.0625f;
		DTYPE mb_coef = 0.125f;
		DTYPE rb_coef = 0.0625f;

		if (idx_x == 0)
		{
			lb_coef *= 2.f;
			lm_coef *= 2.f;
			lt_coef *= 2.f;
		}
		if (idx_y == 0)
		{
			lb_coef *= 2.f;
			mb_coef *= 2.f;
			rb_coef *= 2.f;
		}

		DTYPE res = lb_coef * src[INDEX2D(idx_y_src - 1, idx_x_src - 1, dim_src)] +
			lm_coef * src[INDEX2D(idx_y_src, idx_x_src - 1, dim_src)] +
			mb_coef * src[INDEX2D(idx_y_src - 1, idx_x_src, dim_src)] +
			0.25f * src[INDEX2D(idx_y_src, idx_x_src, dim_src)];

		if (idx_y < dim.y - 1)
		{
			res += lt_coef * src[INDEX2D(idx_y_src + 1, idx_x_src - 1, dim_src)] + 0.125f * src[INDEX2D(idx_y_src + 1, idx_x_src, dim_src)];
		}
		if (idx_x < dim.x - 1)
		{
			res += rb_coef * src[INDEX2D(idx_y_src - 1, idx_x_src + 1, dim_src)] + 0.125f * src[INDEX2D(idx_y_src, idx_x_src + 1, dim_src)];
		}
		if (idx_y < dim.y - 1 && idx_x < dim.x - 1)
		{
			res += 0.0625f * src[INDEX2D(idx_y_src + 1, idx_x_src + 1, dim_src)];
		}
		dst[idx] = res;
	}
}

__global__ void cuRestriction2D_p2_221(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y * 2;

		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		dst[idx] = 0.25f * (src[idx_src] + src[idx_src + 1] + src[idx_src + dim_src.y] + src[idx_src + dim_src.y + 1]);
	}
}

__global__ void cuRestrict2D_p2_c4_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x_src, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);
		DTYPE res = 0.375f * (src[idx_src - dim_src.y] + src[idx_src]);
		if (idx_x > 0)
		{
			res += 0.125f * src[idx_src - 2 * dim_src.y];
		}
		if (idx_x < dim.x - 1)
		{
			res += 0.125f * src[idx_src + dim_src.y];
		}
		dst[idx_dst] = res;
	}
}

__global__ void cuRestrict2D_p2_c4_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		//int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);
		DTYPE res = 0.375f * (src[idx_src - 1] + src[idx_src]);
		if (idx_y > 0)
		{
			res += 0.125f * src[idx_src - 2];
		}
		if (idx_y < dim.y - 1)
		{
			res += 0.125f * src[idx_src + 1];
		}
		dst[idx_dst] = res;
	}
}

__global__ void cuRestrict2D_m1_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y;

		int idx_dst = INDEX2D(idx_y, idx_x_src, dim_src);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		dst[idx_dst] = 0.25f * src[idx_src - dim_src.y] + 0.5f * src[idx_src] + 0.25f * src[idx_src + dim_src.y];
	}
}

__global__ void cuRestrict2D_m1_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);
		dst[idx_dst] = 0.25f * src[idx_src - 1] + 0.5f * src[idx_src] + 0.25f * src[idx_src + 1];
	}
}

__global__ void cuRestrict2D_p2_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x_src - 1, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src - dim_src.y] + src[idx_src]);
	}
}

__global__ void cuRestrict2D_p2_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src - 1] + src[idx_src]);
	}
}

__global__ void cuRestrict2D_m1_n_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x_src, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);

		DTYPE l_coef = idx_x == 0 ? 0.5f : 0.25f;
		DTYPE r_coef = idx_x == dim.x - 1 ? 0.5f : 0.25f;

		dst[idx_dst] = l_coef * src[idx_src - dim_src.y] + r_coef * src[idx_src + dim_src.y] + 0.5f * src[idx_src];
	}
}

__global__ void cuRestrict2D_m1_n_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);


		DTYPE l_coef = idx_y == 0 ? 0.5f : 0.25f;
		DTYPE r_coef = idx_y == dim.y - 1 ? 0.5f : 0.25f;

		dst[idx_dst] = l_coef * src[idx_src - 1] + r_coef * src[idx_src + 1] + 0.5f * src[idx_src];
	}
}

__global__ void cuRestrict2D_m1_main_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y;

		int idx_dst = INDEX2D(idx_y, idx_x_src, dim_src);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		dst[idx_dst] = src[idx_src - dim_src.y];
	}
}

__global__ void cuRestrict2D_m1_main_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);
		dst[idx_dst] = src[idx_src - 1];
	}
}

__global__ void cuRestrict2D_p1_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);

		DTYPE val = 0.5f * src[idx_src];
		if (idx_x < dim.x - 1)
		{
			val += 0.5f * src[idx_src + dim_src.y];
		}
		else
		{
			val *= 2.f;
		}

		dst[idx_dst] = val;
	}
}

__global__ void cuRestrict2D_p1_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		DTYPE val = 0.5f * src[idx_src];
		if (idx_y < dim.y - 1)
		{
			val += 0.5f * src[idx_src + 1];
		}
		else
		{
			val *= 2.f;
		}

		dst[idx_dst] = val;
	}
}

__global__ void cuRestrict2D_p1_main_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		//int off_idx = idx_x < dim.x - 1 ? dim_src.y : 2 * dim_src.y;
		//if (idx_x == dim.x - 1)
		//{
		//	idx_src -= dim_src.y;
		//	//idx_dst -= dim_src.y;
		//}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict2D_p1_main_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		//if (idx_y == dim.y - 1)
		//{
		//	idx_src -= 1;
		//}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict2D_p2_lesser_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src] + src[idx_src + dim_src.y]);
	}
}

__global__ void cuRestrict2D_p2_lesser_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src + 1] + src[idx_src]);
	}
}

__global__ void cuRestrict2D_p2_main_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		if (idx_y == dim.y - 1)
		{
			idx_src -= 1;
		}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict2D_p2_main_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		int off_idx = idx_x < dim.x - 1 ? dim_src.y : 2 * dim_src.y;
		if (idx_x == dim.x - 1)
		{
			idx_src -= dim_src.y;
			//idx_dst -= dim_src.y;
		}
		dst[idx_dst] = src[idx_src];
	}
}


__global__ void cuRestriction2D_2nm1_rn(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_c = idx_x * 2;
		int idx_y_c = idx_y * 2;

		int idx_x_l = idx_x_c - 1;
		int idx_x_r = idx_x_c + 1;
		int idx_y_b = idx_y_c - 1;
		int idx_y_t = idx_y_c + 1;

		DTYPE val_rb = cuInRange(idx_x_r, 0, dim_src.x - 1) && cuInRange(idx_y_b, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_b, idx_x_r, dim_src)] : 0.f;
		DTYPE val_rc = cuInRange(idx_x_r, 0, dim_src.x - 1) && cuInRange(idx_y_c, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_c, idx_x_r, dim_src)] : 0.f;
		DTYPE val_rt = cuInRange(idx_x_r, 0, dim_src.x - 1) && cuInRange(idx_y_t, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_t, idx_x_r, dim_src)] : 0.f;
		DTYPE val_cb = cuInRange(idx_x_c, 0, dim_src.x - 1) && cuInRange(idx_y_b, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_b, idx_x_c, dim_src)] : 0.f;
		DTYPE val_cc = cuInRange(idx_x_c, 0, dim_src.x - 1) && cuInRange(idx_y_c, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_c, idx_x_c, dim_src)] : 0.f;
		DTYPE val_ct = cuInRange(idx_x_c, 0, dim_src.x - 1) && cuInRange(idx_y_t, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_t, idx_x_c, dim_src)] : 0.f;
		DTYPE val_lb = cuInRange(idx_x_l, 0, dim_src.x - 1) && cuInRange(idx_y_b, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_b, idx_x_l, dim_src)] : 0.f;
		DTYPE val_lc = cuInRange(idx_x_l, 0, dim_src.x - 1) && cuInRange(idx_y_c, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_c, idx_x_l, dim_src)] : 0.f;
		DTYPE val_lt = cuInRange(idx_x_l, 0, dim_src.x - 1) && cuInRange(idx_y_t, 0, dim_src.y - 1) ? src[INDEX2D(idx_y_t, idx_x_l, dim_src)] : 0.f;

		dst[idx] = 0.0625f * val_lb + 0.125f * val_lc +
			0.0625f * val_lt + 0.125f * val_cb +
			0.25f * val_cc + 0.125f * val_ct +
			0.0625f * val_rb + 0.125f * val_rc +
			0.0625f * val_rt;
	}
}


__global__ void cuRestrict2D_p1_dx_x(DTYPE* dst, DTYPE* src, DTYPE* dx, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);

		DTYPE val = dx[idx_x_src] * src[idx_src];
		DTYPE w = dx[idx_x_src];
		if (idx_x_src < dim_src.x - 1)
		{
			val += dx[idx_x_src + 1] * src[idx_src + dim_src.y];
			w += dx[idx_x_src + 1];
		}

		dst[idx_dst] = val / w;
	}
}

__global__ void cuRestrict2D_p1_dy_y(DTYPE* dst, DTYPE* src, DTYPE* dy, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		DTYPE val = dy[idx_y_src] * src[idx_src];
		DTYPE w = dy[idx_y_src];
		if (idx_y_src < dim_src.y - 1)
		{
			val += dy[idx_y_src + 1] * src[idx_src + 1];
			w += dy[idx_y_src + 1];
		}

		dst[idx_dst] = val / w;
	}
}

__global__ void cuRestrict2D_p1_nodx_x(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim_src);
		int idx_src = INDEX2D(idx_y, idx_x_src, dim_src);

		DTYPE val = 0.5f * src[idx_src];
		if (idx_x < dim.x - 1)
		{
			val += 0.5f * src[idx_src + dim_src.y];
		}

		dst[idx_dst] = val;
	}
}

__global__ void cuRestrict2D_p1_nody_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX2D(idx_y, idx_x, dim);
		int idx_src = INDEX2D(idx_y_src, idx_x_src, dim_src);

		DTYPE val = 0.5f * src[idx_src];
		if (idx_y < dim.y - 1)
		{
			val += 0.5f * src[idx_src + 1];
		}

		dst[idx_dst] = val;
	}
}

CuRestriction2D* CuRestriction2D::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuRestriction2D>(new CuRestriction2D); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void CuRestriction2D::Solve(DTYPE* dst, DTYPE* src, int2 dim, RestrictionType rs_type, char bc)
{
	int max_size;
	int2 dim_dst;
	DTYPE* tmp;
	int2 dim_thread = { dim.x, dim.y };
	int thread_size;

	switch (rs_type)
	{
	case RestrictionType::RT_2N:
		dim_dst.x = dim.x / 2;
		dim_dst.y = dim.y / 2;
		max_size = dim_dst.x * dim_dst.y;
		if (bc == 'n')
		{
			cuRestriction2D_p2_neumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
		}
		else
		{
			cuRestriction2D_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
		}
		//cuRestriction2D_p2_221 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);

		//tmp = CuMemoryManager::GetInstance()->GetData("tmp", dim.x * dim.y);
		//dim_thread.x = dim_dst.x;
		//thread_size = dim_thread.x * dim_thread.y;
		//cuRestrict2D_p2_c4_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
		//thread_size = dim_dst.x * dim_dst.y;
		//cuRestrict2D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
		break;
	case RestrictionType::RT_2N_C4:
		dim_dst.x = dim.x / 2;
		dim_dst.y = dim.y / 2;
		max_size = dim_dst.x * dim_dst.y;
		if (bc == 'n')
		{
			//cuRestriction2D_p2_neumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
			tmp = CuMemoryManager::GetInstance()->GetData("tmp", dim.x * dim.y);
			dim_thread.x = dim_dst.x;
			thread_size = dim_thread.x * dim_thread.y;
			cuRestrict2D_p2_c4_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
			thread_size = dim_dst.x * dim_dst.y;
			cuRestrict2D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
		}
		else
		{
			cuRestriction2D_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
		}
		break;
	default:
		dim_dst.x = (dim.x + 1) / 2 - 1;
		dim_dst.y = (dim.y + 1) / 2 - 1;
		max_size = dim_dst.x * dim_dst.y;
		cuRestriction2D_2nm1 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
	}
}

void CuRestriction2D::SolveRN(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return ((dim + 1) >> 1) - 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		if (bc == 'n')
		{
			cuRestrict2D_m1_n_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
		}
		else
		{
			cuRestrict2D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
		}
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
	else
	{
		if (bc == 'n')
		{
			cuRestrict2D_m1_n_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
		}
		else
		{
			cuRestrict2D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
		}
	}
}

void CuRestriction2D::Solve2NM1RN(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	//int max_size = dim.x * dim.y;
	//DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int
	{
			return ((dim) >> 1) + 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim(dim.y) };
	int max_size = dim_dst.x * dim_dst.y;
	cuRestriction2D_2nm1 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuRestriction2D::SolveP1RN(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict2D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
	else
	{
		cuRestrict2D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
}

void CuRestriction2D::SolveP1RN(DTYPE* dst, DTYPE* src, DTYPE* dx, DTYPE* dy, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	//if ((dim.x & 1) == 0)
	//{
	//	cuRestrict2D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	//}
	//else
	//{
	//	cuRestrict2D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	//}
	cuRestrict2D_p1_dx_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dx, dim_thread, dim, thread_size);

	thread_size = dim_dst.x * dim_dst.y;
	//if ((dim.y & 1) == 0)
	//{
	//	cuRestrict2D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	//}
	//else
	//{
	//	cuRestrict2D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	//}
	cuRestrict2D_p1_dy_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dy, dim_dst, dim, thread_size);
}


void CuRestriction2D::SolveAx(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return ((dim + 1) >> 1) - 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim_main(dim.x), rs_dim(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict2D_m1_main_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
	else
	{
		cuRestrict2D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
}

void CuRestriction2D::SolveAy(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return ((dim + 1) >> 1) - 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim_main(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict2D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
	else
	{
		cuRestrict2D_m1_main_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
}

void CuRestriction2D::SolveP1Ax(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		//if ((dim & 1) == 0)
		//	return dim >> 1;
		//else
		//	return (dim >> 1) + 1;
		return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim_main(dim.x), rs_dim(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;
	//CudaPrintfMat(src, dim);
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_main_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict2D_p1_main_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		cuRestrict2D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
}

void CuRestriction2D::SolveP1Ay(DTYPE* dst, DTYPE* src, int2 dim, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		//if ((dim & 1) == 0)
		//	return dim >> 1;
		//else
		//	return (dim >> 1) + 1;
		return (dim >> 1) + 1;
	};

	int2 dim_dst = { rs_dim(dim.x), rs_dim_main(dim.y) };
	int2 dim_thread = { dim_dst.x, dim.y };
	int thread_size = dim_thread.x * dim_thread.y;

	//CudaPrintfMat(src, dim);
	if ((dim.x & 1) == 0)
	{
		cuRestrict2D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict2D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y;
	if ((dim.y & 1) == 0)
	{
		cuRestrict2D_p2_main_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		cuRestrict2D_p1_main_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
}