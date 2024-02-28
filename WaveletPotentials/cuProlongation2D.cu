#include "cuProlongation2D.cuh"
#include "cuMemoryManager.cuh"

CuProlongation2D::CuProlongation2D() = default;
CuProlongation2D::~CuProlongation2D() = default;

std::auto_ptr<CuProlongation2D> CuProlongation2D::instance_;

__global__ void cuProlongation2D_2nm1(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y + 1);
		int idx_y = idx - idx_x * (dim_src.y + 1);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE res_00 = 0.f;
		DTYPE res_10 = 0.f;
		DTYPE res_01 = 0.f;
		DTYPE res_11 = 0.f;

		if (idx_x > 0)
		{
			if (idx_y > 0)
			{
				res_00 = 0.25 * src[INDEX2D(idx_y - 1, idx_x - 1, dim_src)];
			}
			if (idx_y < dim_src.y)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y, idx_x - 1, dim_src)];
				res_10 += 0.5 * src[INDEX2D(idx_y, idx_x - 1, dim_src)];

				//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d, res_00: %f, res_10: %f\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst, res_00, res_10);
			}
		}
		if (idx_x < dim_src.x)
		{
			if (idx_y > 0)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
				res_01 += 0.5 * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
			}
			if (idx_y < dim_src.y)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y, idx_x, dim_src)];
				res_10 += 0.5 * src[INDEX2D(idx_y, idx_x, dim_src)];
				res_01 += 0.5 * src[INDEX2D(idx_y, idx_x, dim_src)];
				dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] = src[INDEX2D(idx_y, idx_x, dim_src)];
			}
		}
		if (idx_y < dim_src.y)
		{
			dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] = res_10;
		}
		if (idx_x < dim_src.x)
		{
			dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] = res_01;
		}
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] = res_00;

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

__global__ void cuProlongation2D_2nm1_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y + 1);
		int idx_y = idx - idx_x * (dim_src.y + 1);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE res_00 = 0.f;
		DTYPE res_10 = 0.f;
		DTYPE res_01 = 0.f;
		DTYPE res_11 = 0.f;

		if (idx_x > 0)
		{
			if (idx_y > 0)
			{
				res_00 = 0.25 * src[INDEX2D(idx_y - 1, idx_x - 1, dim_src)];
			}
			if (idx_y < dim_src.y)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y, idx_x - 1, dim_src)];
				res_10 += 0.5 * src[INDEX2D(idx_y, idx_x - 1, dim_src)];
			}
		}
		if (idx_x < dim_src.x)
		{
			if (idx_y > 0)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
				res_01 += 0.5 * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
			}
			if (idx_y < dim_src.y)
			{
				res_00 += 0.25 * src[INDEX2D(idx_y, idx_x, dim_src)];
				res_10 += 0.5 * src[INDEX2D(idx_y, idx_x, dim_src)];
				res_01 += 0.5 * src[INDEX2D(idx_y, idx_x, dim_src)];
				dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] += src[INDEX2D(idx_y, idx_x, dim_src)];
			}
		}
		if (idx_y < dim_src.y)
		{
			dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] += res_10;
		}
		if (idx_x < dim_src.x)
		{
			dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] += res_01;
		}
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] += res_00;

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

__global__ void cuProlongation2D_p2_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE res_00 = 0.f;
		DTYPE res_10 = 0.f;
		DTYPE res_01 = 0.f;
		DTYPE res_11 = 0.f;

		if (idx_x > 0)
		{
			if (idx_y > 0)
			{
				res_00 = 0.25f * src[INDEX2D(idx_y - 1, idx_x - 1, dim_src)];
			}

			res_00 += 0.25f * src[INDEX2D(idx_y, idx_x - 1, dim_src)];
			res_10 += 0.5f * src[INDEX2D(idx_y, idx_x - 1, dim_src)];
		}

		if (idx_y > 0)
		{
			res_00 += 0.25f * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
			res_01 += 0.5f * src[INDEX2D(idx_y - 1, idx_x, dim_src)];
		}

		res_00 += 0.25f * src[INDEX2D(idx_y, idx_x, dim_src)];
		res_10 += 0.5f * src[INDEX2D(idx_y, idx_x, dim_src)];
		res_01 += 0.5f * src[INDEX2D(idx_y, idx_x, dim_src)];

		dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] += src[INDEX2D(idx_y, idx_x, dim_src)];
		dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] += res_10;
		dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] += res_01;
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] += res_00;

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

__global__ void cuProlongation2D_p2_neumann_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE res_00 = 0.f;
		DTYPE res_10 = 0.f;
		DTYPE res_01 = 0.f;
		DTYPE res_11 = 0.f;

		res_00 = 0.25f * src[INDEX2D(max(idx_y - 1, 0), max(idx_x - 1, 0), dim_src)];

		res_00 += 0.25f * src[INDEX2D(idx_y, max(idx_x - 1, 0), dim_src)];
		res_10 += 0.5f * src[INDEX2D(idx_y, max(idx_x - 1, 0), dim_src)];


		res_00 += 0.25f * src[INDEX2D(max(idx_y - 1, 0), idx_x, dim_src)];
		res_01 += 0.5f * src[INDEX2D(max(idx_y - 1, 0), idx_x, dim_src)];

		res_00 += 0.25f * src[INDEX2D(idx_y, idx_x, dim_src)];
		res_10 += 0.5f * src[INDEX2D(idx_y, idx_x, dim_src)];
		res_01 += 0.5f * src[INDEX2D(idx_y, idx_x, dim_src)];

		dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] += src[INDEX2D(idx_y, idx_x, dim_src)];
		dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] += res_10;
		dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] += res_01;
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] += res_00;

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

__global__ void cuProlongation2D_p2_221_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE val = src[INDEX2D(idx_y, idx_x, dim_src)];
		dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] += val;
		dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] += val;
		dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] += val;
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] += val;

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

__global__ void cuProlongate2D_p2_c4_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = INDEX2D(idx_y, idx_x, dim_src);
		DTYPE val_0 = 0.75f * src[idx_src];
		DTYPE val_m1 = val_0;
		DTYPE val_p1 = val_0;

		if (idx_y > 0)
		{
			val_m1 += 0.25f * src[idx_src - 1];
		}
		if (idx_y < dim_src.y - 1)
		{
			val_p1 += 0.25f * src[idx_src + 1];
		}
		dst[idx_dst + 1] = val_p1;
		dst[idx_dst] = val_m1;
	}
}

__global__ void cuProlongate2D_p2_c4_x_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = idx_dst + dim.y;
		//DTYPE val_0 = src[idx_src];
		DTYPE val_m1 = 0.75f * src[idx_src];
		DTYPE val_p1 = val_m1;

		if (idx_x > 0)
		{
			val_m1 += 0.25f * src[idx_src - 2 * dim.y];
		}
		if (idx_x < dim_src.x - 1)
		{
			val_p1 += 0.25f * src[idx_src + 2 * dim.y];
		}
		dst[idx_dst + dim.y] += val_p1;
		dst[idx_dst] += val_m1;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate2D_m1_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y + 1);
		int idx_y = idx - idx_x * (dim_src.y + 1);

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = INDEX2D(idx_y, idx_x, dim_src);
		DTYPE val_0 = idx_y > 0 ? 0.5 * src[idx_src - 1] : 0.f;
		//DTYPE val_1 = 0.f;
		if (idx_y < dim_src.y)
		{
			DTYPE val_1 = src[idx_src];
			val_0 += 0.5 * val_1;
			dst[idx_dst + 1] = val_1;
		}
		dst[idx_dst] = val_0;
		//printf("idx: %d, %d, idx_src: %d, idx_dst: %d\n", idx_x, idx_y, idx_src, idx_dst);
	}
}

__global__ void cuProlongate2D_m1_x_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		//int idx_src = INDEX3D(idx_y_dst, idx_x_dst + 1, idx_z_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = idx_x > 0 ? 0.5 * src[idx_src - 2 * dim.y] : 0.f;
		DTYPE val_1 = 0.f;
		if (idx_x < dim_src.x)
		{
			val_1 = src[idx_src];
			val_0 += 0.5 * val_1;
			dst[idx_dst + dim.y] += val_1;
		}
		dst[idx_dst] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate2D_m1_n_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y + 1);
		int idx_y = idx - idx_x * (dim_src.y + 1);

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = INDEX2D(idx_y, idx_x, dim_src);
		DTYPE val_0 = 0.5f * (idx_y > 0 ? src[idx_src - 1] : src[idx_src]);
		//DTYPE val_1 = 0.f;
		if (idx_y < dim_src.y)
		{
			DTYPE val_1 = src[idx_src];
			val_0 += 0.5f * val_1;
			dst[idx_dst + 1] = val_1;
		}
		else
		{
			val_0 *= 2.f;
		}
		dst[idx_dst] = val_0;
	}
}

__global__ void cuProlongate2D_m1_n_x_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		//int idx_src = INDEX3D(idx_y_dst, idx_x_dst + 1, idx_z_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = 0.5f * (idx_x > 0 ? src[idx_src - 2 * dim.y] : src[idx_src]);
		//DTYPE val_1 = 0.f;
		if (idx_x < dim_src.x)
		{
			DTYPE val_1 = src[idx_src];
			val_0 += 0.5f * val_1;
			dst[idx_dst + dim.y] += val_1;
		}
		else
		{
			val_0 *= 2.f;
		}
		dst[idx_dst] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate2D_p2_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = INDEX2D(idx_y, idx_x, dim_src);
		DTYPE val_0 = src[idx_src];

		dst[idx_dst + 1] = val_0;
		dst[idx_dst] = val_0;
	}
}

__global__ void cuProlongate2D_p2_x_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = src[idx_src];
		dst[idx_dst + dim.y] += val_0;
		dst[idx_dst] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate2D_p1_y(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = INDEX2D(idx_y, idx_x, dim_src);
		DTYPE val_0 = src[idx_src];

		dst[idx_dst] = val_0;

		if (idx_y < dim_src.y - 1)
			dst[idx_dst + 1] = val_0;
	}
}

__global__ void cuProlongate2D_p1_x_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / (dim_src.y);
		int idx_y = idx - idx_x * (dim_src.y);

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;

		int idx_dst = INDEX2D(idx_y_dst, idx_x_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = src[idx_src];
		dst[idx_dst] += val_0;

		if (idx_x < dim_src.x - 1)
			dst[idx_dst + dim.y] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}


__global__ void cuProlongation2D_2nm1_rn_add(DTYPE* dst, DTYPE* src, int2 dim, int2 dim_src, int2 dim_thread, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim_thread.y;
		int idx_y = idx - idx_x * dim_thread.y;

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y * 2;

		DTYPE res_00 = 0.f;
		DTYPE res_10 = 0.f;
		DTYPE res_01 = 0.f;
		DTYPE res_11 = 0.f;

		res_00 = src[INDEX2D(idx_y, idx_x , dim_src)];
		res_11 = 0.25f * res_00;
		res_01 = 0.5f * res_00;
		res_10 = 0.5f * res_00;
		if (idx_x + 1 < dim_src.x)
		{
			res_01 += 0.5f * src[INDEX2D(idx_y, idx_x + 1, dim_src)];
			res_11 += 0.25f * src[INDEX2D(idx_y, idx_x + 1, dim_src)];
			if (idx_y + 1 < dim_src.y)
			{
				//res_10 += 0.5f * src[INDEX2D(idx_y + 1, idx_x + 1, dim_src)];
				//res_01 += 0.5f * src[INDEX2D(idx_y + 1, idx_x + 1, dim_src)];
				res_11 += 0.25f * src[INDEX2D(idx_y + 1, idx_x + 1, dim_src)];
			}
		}
		dst[INDEX2D(idx_y_dst, idx_x_dst, dim)] += res_00;
		if (idx_y + 1 < dim_src.y)
		{
			res_10 += 0.5f * src[INDEX2D(idx_y + 1, idx_x, dim_src)];
			res_11 += 0.25f * src[INDEX2D(idx_y + 1, idx_x, dim_src)];
			dst[INDEX2D(idx_y_dst + 1, idx_x_dst, dim)] += res_10;
		}
		if (idx_x + 1 < dim_src.x)
		{
			dst[INDEX2D(idx_y_dst, idx_x_dst + 1, dim)] += res_01;
			if (idx_y + 1 < dim_src.y)
				dst[INDEX2D(idx_y_dst + 1, idx_x_dst + 1, dim)] += res_11;
		}

		//printf("idx: %d, idx_x, idx_y: %d, %d, idx_x_dst, idx_y_dst: %d, %d\n", idx, idx_x, idx_y, idx_x_dst, idx_y_dst);
	}
}

CuProlongation2D* CuProlongation2D::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuProlongation2D>(new CuProlongation2D); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void CuProlongation2D::Solve(DTYPE* dst, DTYPE* src, int2 dim, ProlongationType pg_type)
{
	int max_size;
	int2 dim_dst;

	switch (pg_type)
	{
	default:
		dim_dst.x = (dim.x + 1) * 2 - 1;
		dim_dst.y = (dim.y + 1) * 2 - 1;
		max_size = (dim.x + 1) * (dim.y + 1);
		cuProlongation2D_2nm1 << <BLOCKS(max_size), NUM_THREADS >> > (dst, src, dim_dst, dim, max_size);
	}
}

void CuProlongation2D::SolveAdd(DTYPE* dst, DTYPE* src, int2 dim, ProlongationType pg_type, char bc)
{
	int max_size;
	int2 dim_dst;

	int2 dim_thread = dim;
	int thread_size;

	DTYPE* tmp;

	switch (pg_type)
	{
	case ProlongationType::PT_2N:
		dim_dst.x = dim.x * 2;
		dim_dst.y = dim.y * 2;
		max_size = dim.x * dim.y;
		if (bc == 'n')
			cuProlongation2D_p2_neumann_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
		else
			cuProlongation2D_p2_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
		//break;
		//cuProlongation2D_p2_221_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);

		//tmp = CuMemoryManager::GetInstance()->GetData("tmp", dim_dst.x * dim_dst.y);
		//thread_size = dim_thread.x * dim_thread.y;
		//cuProlongate2D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		//dim_thread.y = dim_dst.y;
		//thread_size = dim_thread.x * dim_thread.y;
		//cuProlongate2D_p2_c4_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);

		break;
	case ProlongationType::PT_2N_C4:
		dim_dst.x = dim.x * 2;
		dim_dst.y = dim.y * 2;
		max_size = dim.x * dim.y;
		if (bc == 'n')
		{
			tmp = CuMemoryManager::GetInstance()->GetData("tmp", dim_dst.x * dim_dst.y);
			thread_size = dim_thread.x * dim_thread.y;
			cuProlongate2D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
			dim_thread.y = dim_dst.y;
			thread_size = dim_thread.x * dim_thread.y;
			cuProlongate2D_p2_c4_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		}
		else
			cuProlongation2D_p2_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);

		break;
	default:
		dim_dst.x = (dim.x + 1) * 2 - 1;
		dim_dst.y = (dim.y + 1) * 2 - 1;
		max_size = (dim.x + 1) * (dim.y + 1);
		cuProlongation2D_2nm1_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
	}
}

void CuProlongation2D::SolveRNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc)
{
	int max_size = dim_dst.x * dim_dst.y;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	//cudaMemset(tmp, 0, sizeof(DTYPE) * max_size);
	int2 dim_thread = dim;
	int thread_size;
	if ((dim_dst.y & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y;
		cuProlongate2D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * (dim_thread.y + 1);
		if (bc == 'n')
		{
			cuProlongate2D_m1_n_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		}
		else
		{
			cuProlongate2D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		}
	}

	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.y = dim_dst.y;
	if ((dim_dst.x & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y;
		cuProlongate2D_p2_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = (dim_thread.x + 1) * dim_thread.y;	
		if (bc == 'n')
		{
			cuProlongate2D_m1_n_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		}
		else
		{
			cuProlongate2D_m1_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		}
	}
	//CudaPrintfMat(dst, dim_dst);

	//cuProlongation2D_2nm1_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuProlongation2D::Solve2NM1RNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc)
{
	int2 dim_thread = { (dim_dst.x + 1) >> 1, (dim_dst.y + 1) >> 1 };
	int max_size = dim_thread.x * dim_thread.y;
	cuProlongation2D_2nm1_rn_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, dim_thread, max_size);
}

void CuProlongation2D::SolveP1RNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc)
{
	int max_size = dim_dst.x * dim_dst.y;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	//cudaMemset(tmp, 0, sizeof(DTYPE) * max_size);
	int2 dim_thread = dim;
	int thread_size;
	if ((dim_dst.y & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y;
		cuProlongate2D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * (dim_thread.y);
		cuProlongate2D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		//if (bc == 'n')
		//{
		//	cuProlongate2D_m1_n_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		//}
		//else
		//{
		//	cuProlongate2D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
		//}
	}

	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.y = dim_dst.y;
	if ((dim_dst.x & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y;
		cuProlongate2D_p2_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = (dim_thread.x) * dim_thread.y;
		cuProlongate2D_p1_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		//if (bc == 'n')
		//{
		//	cuProlongate2D_m1_n_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		//}
		//else
		//{
		//	cuProlongate2D_m1_x_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
		//}
	}
	//CudaPrintfMat(dst, dim_dst);

	//cuProlongation2D_2nm1_add << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dst, src, dim_dst, dim, max_size);
}