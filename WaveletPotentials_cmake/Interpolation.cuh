#ifndef __INTERPOLATION_CUH__
#define __INTERPOLATION_CUH__

#include "cudaGlobal.cuh"
#include "cudaMath.cuh"
#include <cuda_runtime.h>

#ifdef USE_DOUBLE
#define CLAMP_L 0.0000000001f
#define CLAMP_R 1.0000000001f
#else
#define CLAMP_L 0.00001f
#define CLAMP_R 1.00001f
#endif

inline __host__ __device__ void cuGetIxFx(int& ix, DTYPE& fx, DTYPE x, int dim_1d)
{
	x = max(CLAMP_L, x);
	ix = int(x);
	if (ix >= dim_1d - 1)
	{
		ix = dim_1d - 2;
		fx = 1.f - CLAMP_L;
	}
	else
	{
		fx = x - ix;
	}
}

inline __host__ __device__ void cuGetIx(int& ix, DTYPE x, int dim_1d)
{
	x = max(CLAMP_L, x);
	ix = int(x);
	if (ix >= dim_1d - 1)
	{
		ix = dim_1d - 2;
	}
}

inline __host__ __device__ DTYPE InterpolateLinear3D(int* val, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;

	int ix, iy, iz;
	DTYPE fx, fy, fz;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);
	cuGetIxFx(iz, fz, z, dim.z);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };
	DTYPE wz[2] = { 1 - fz, fz };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE res = val[idx] * wx[0] * wy[0] * wz[0];
	//printf("idx: %d, val: %f\n", idx, val[idx]);
	res += val[idx + 1] * wx[0] * wy[1] * wz[0];
	res += val[idx + dim.y] * wx[1] * wy[0] * wz[0];
	res += val[idx + dim.y + 1] * wx[1] * wy[1] * wz[0];
	idx += slice;
	res += val[idx] * wx[0] * wy[0] * wz[1];
	res += val[idx + 1] * wx[0] * wy[1] * wz[1];
	res += val[idx + dim.y] * wx[1] * wy[0] * wz[1];
	res += val[idx + dim.y + 1] * wx[1] * wy[1] * wz[1];
	return res;
}

inline __host__ __device__ DTYPE InterpolateLinear3D(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);


	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;
	int ix, iy, iz;
	DTYPE fx, fy, fz;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);
	cuGetIxFx(iz, fz, z, dim.z);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };
	DTYPE wz[2] = { 1 - fz, fz };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE res = val[idx] * wx[0] * wy[0] * wz[0];
	//printf("idx: %d, val: %f\n", idx, val[idx]);
	res += val[idx + 1] * wx[0] * wy[1] * wz[0];
	res += val[idx + dim.y] * wx[1] * wy[0] * wz[0];
	res += val[idx + dim.y + 1] * wx[1] * wy[1] * wz[0];
	idx += slice;
	res += val[idx] * wx[0] * wy[0] * wz[1];
	res += val[idx + 1] * wx[0] * wy[1] * wz[1];
	res += val[idx + dim.y] * wx[1] * wy[0] * wz[1];
	res += val[idx + dim.y + 1] * wx[1] * wy[1] * wz[1];
	return res;
}

inline __host__ __device__ DTYPE InterpolateQuadratic3D(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	int ix = floor(x - 0.5f);
	int iy = floor(y - 0.5f);
	int iz = floor(z - 0.5f);

	DTYPE fx = x - ix;
	DTYPE fy = y - iy;
	DTYPE fz = z - iz;

	DTYPE wx[3] = { 0.5f * my_pow2(1.5f - fx), 0.75f - my_pow2(fx - 1.f), 0.5f * my_pow2(fx - 0.5f) };
	DTYPE wy[3] = { 0.5f * my_pow2(1.5f - fy), 0.75f - my_pow2(fy - 1.f), 0.5f * my_pow2(fy - 0.5f) };
	DTYPE wz[3] = { 0.5f * my_pow2(1.5f - fz), 0.75f - my_pow2(fz - 1.f), 0.5f * my_pow2(fz - 0.5f) };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE res = 0.f;
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				int idx_x = clamp(ix + j, 0, dim.x - 1);
				int idx_y = clamp(iy + i, 0, dim.y - 1);
				int idx_z = clamp(iz + k, 0, dim.z - 1);

				int idx = idx_y + idx_x * dim.y + idx_z * slice;
				res += wx[j] * wy[i] * wz[k] * val[idx];
			}
		}
	}
	return res;
}

inline __host__ __device__ DTYPE3 InterpolateQuadratic3D_dxyz(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3 dim, DTYPE inv_dx)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	int ix = floor(x - 0.5f);
	int iy = floor(y - 0.5f);
	int iz = floor(z - 0.5f);

	DTYPE fx = x - ix;
	DTYPE fy = y - iy;
	DTYPE fz = z - iz;

	DTYPE wx[3] = { 0.5f * my_pow2(1.5f - fx), 0.75f - my_pow2(fx - 1.f), 0.5f * my_pow2(fx - 0.5f) };
	DTYPE wy[3] = { 0.5f * my_pow2(1.5f - fy), 0.75f - my_pow2(fy - 1.f), 0.5f * my_pow2(fy - 0.5f) };
	DTYPE wz[3] = { 0.5f * my_pow2(1.5f - fz), 0.75f - my_pow2(fz - 1.f), 0.5f * my_pow2(fz - 0.5f) };

	DTYPE dwx[3] = { (fx - 1.5f) * inv_dx, -2.f * (fx - 1.f) * inv_dx, (fx - 0.5f) * inv_dx };
	DTYPE dwy[3] = { (fy - 1.5f) * inv_dx, -2.f * (fy - 1.f) * inv_dx, (fy - 0.5f) * inv_dx };
	DTYPE dwz[3] = { (fz - 1.5f) * inv_dx, -2.f * (fz - 1.f) * inv_dx, (fz - 0.5f) * inv_dx };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE3 res = { 0.f, 0.f, 0.f };
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				int idx_x = clamp(ix + j, 0, dim.x - 1);
				int idx_y = clamp(iy + i, 0, dim.y - 1);
				int idx_z = clamp(iz + k, 0, dim.z - 1);

				int idx = idx_y + idx_x * dim.y + idx_z * slice;
				res.x += dwx[j] * wy[i] * wz[k] * val[idx];
				res.y += wx[j] * dwy[i] * wz[k] * val[idx];
				res.z += wx[j] * wy[i] * dwz[k] * val[idx];
			}
		}
	}
	return res;
}

inline __host__ __device__ void cuRefIdx(int& idx, DTYPE& ref_sign, int dim)
{
	if (idx < 0)
	{
		idx = -idx;
		ref_sign *= -1.f;
	}
	else if (idx >= dim)
	{
		idx = 2 * (dim - 1) - idx;
		ref_sign *= -1.f;
	}
}

inline __host__ __device__ DTYPE GetExtValue(DTYPE* val, int3 idx, int3& dim, char dire)
{
	DTYPE ref_sign = 1.f;
	if (dire == 'x')
	{
		idx.x = clamp(idx.x, 0, dim.x - 1);
		cuRefIdx(idx.y, ref_sign, dim.y);
		cuRefIdx(idx.z, ref_sign, dim.z);
	}
	else if (dire == 'y')
	{
		cuRefIdx(idx.x, ref_sign, dim.x);
		idx.y = clamp(idx.y, 0, dim.y - 1);
		cuRefIdx(idx.z, ref_sign, dim.z);
	}
	else
	{
		cuRefIdx(idx.x, ref_sign, dim.x);
		cuRefIdx(idx.y, ref_sign, dim.y);
		idx.z = clamp(idx.z, 0, dim.z - 1);
	}
	//cuRefIdx(idx.x, ref_sign, dim.x);
	//cuRefIdx(idx.y, ref_sign, dim.y);
	//cuRefIdx(idx.z, ref_sign, dim.z);
	int idx_val = idx.y + idx.x * dim.y + idx.z * dim.x * dim.y;
	return val[idx_val] * ref_sign;
}

inline __host__ __device__ DTYPE cuRefVal(DTYPE* val, int idx, int idx_bef)
{
	return val[idx] - val[idx_bef];
}

inline __host__ __device__ DTYPE GetExtValue_linear(DTYPE* val, int3 idx, int3& dim, char dire)
{
	DTYPE ref_sign = 1.f;
	int3 clamp_idx = make_int3(clamp(idx.x, 0, dim.x - 1), clamp(idx.y, 0, dim.y - 1),
		clamp(idx.z, 0, dim.z - 1));
	int idx_val = INDEX3D(clamp_idx.y, clamp_idx.x, clamp_idx.z, dim);
	DTYPE add_val1 = 0.f;
	DTYPE add_val2 = 0.f;
	if (dire == 'x')
	{
		//idx.x = clamp(idx.x, 0, dim.x - 1);
		if (idx.y < 0)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val + 1);
		}
		else if (idx.y >= dim.y)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val - 1);
		}
		if (idx.z < 0)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val + dim.x * dim.y);
		}
		else if (idx.z >= dim.z)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val - dim.x * dim.y);
		}
	}
	else if (dire == 'y')
	{
		if (idx.x < 0)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val + dim.y);
		}
		else if (idx.x >= dim.x)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val - dim.y);
		}
		if (idx.z < 0)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val + dim.x * dim.y);
		}
		else if (idx.z >= dim.z)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val - dim.x * dim.y);
		}
	}
	else
	{
		if (idx.x < 0)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val + dim.y);
		}
		else if (idx.x >= dim.x)
		{
			add_val1 = cuRefVal(val, idx_val, idx_val - dim.y);
		}
		if (idx.y < 0)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val + 1);
		}
		else if (idx.y >= dim.y)
		{
			add_val2 = cuRefVal(val, idx_val, idx_val - 1);
		}
		//idx.z = clamp(idx.z, 0, dim.z - 1);
	}
	return val[idx_val] + add_val1 + add_val2;
}

inline __host__ __device__ DTYPE3 InterpolateQuadratic3D_dxyz_ext(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3& dim, DTYPE inv_dx, char dire)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	int ix = floor(x - 0.5f);
	int iy = floor(y - 0.5f);
	int iz = floor(z - 0.5f);

	DTYPE fx = x - ix;
	DTYPE fy = y - iy;
	DTYPE fz = z - iz;

	DTYPE wx[3] = { 0.5f * my_pow2(1.5f - fx), 0.75f - my_pow2(fx - 1.f), 0.5f * my_pow2(fx - 0.5f) };
	DTYPE wy[3] = { 0.5f * my_pow2(1.5f - fy), 0.75f - my_pow2(fy - 1.f), 0.5f * my_pow2(fy - 0.5f) };
	DTYPE wz[3] = { 0.5f * my_pow2(1.5f - fz), 0.75f - my_pow2(fz - 1.f), 0.5f * my_pow2(fz - 0.5f) };

	DTYPE dwx[3] = { (fx - 1.5f) * inv_dx, -2.f * (fx - 1.f) * inv_dx, (fx - 0.5f) * inv_dx };
	DTYPE dwy[3] = { (fy - 1.5f) * inv_dx, -2.f * (fy - 1.f) * inv_dx, (fy - 0.5f) * inv_dx };
	DTYPE dwz[3] = { (fz - 1.5f) * inv_dx, -2.f * (fz - 1.f) * inv_dx, (fz - 0.5f) * inv_dx };

	int slice = dim.x * dim.y;
	DTYPE3 res = { 0.f, 0.f, 0.f };
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				//DTYPE val_c = GetExtValue_linear(val, { ix + j, iy + i, iz + k }, dim, dire);
				DTYPE val_c = GetExtValue(val, { ix + j, iy + i, iz + k }, dim, dire);
				res.x += dwx[j] * wy[i] * wz[k] * val_c;
				res.y += wx[j] * dwy[i] * wz[k] * val_c;
				res.z += wx[j] * wy[i] * dwz[k] * val_c;
			}
		}
	}
	return res;
}

inline __host__ __device__ DTYPE3 InterpolateLinear3D_dxyz(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3 dim, DTYPE inv_dx)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;

	int ix, iy, iz;
	DTYPE fx, fy, fz;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);
	cuGetIxFx(iz, fz, z, dim.z);

	DTYPE wx[2] = { 1.f-fx, fx };
	DTYPE wy[2] = { 1.f-fy, fy };
	DTYPE wz[2] = { 1.f-fz, fz };

	DTYPE dwx[2] = { -inv_dx, inv_dx };
	DTYPE dwy[2] = { -inv_dx, inv_dx };
	DTYPE dwz[2] = { -inv_dx, inv_dx };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE3 res = { 0.f, 0.f, 0.f };
	for (int k = 0; k < 2; k++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int i = 0; i < 2; i++)
			{
				int idx_x = clamp(ix + j, 0, dim.x - 1);
				int idx_y = clamp(iy + i, 0, dim.y - 1);
				int idx_z = clamp(iz + k, 0, dim.z - 1);

				int idx = idx_y + idx_x * dim.y + idx_z * slice;
				res.x += dwx[j] * wy[i] * wz[k] * val[idx];
				res.y += wx[j] * dwy[i] * wz[k] * val[idx];
				res.z += wx[j] * wy[i] * dwz[k] * val[idx];
			}
		}
	}
	return res;
}

inline __device__ DTYPE3 cuFindCloestPoint_OneIter(DTYPE* ls, const DTYPE3& xg, const int3& dim, const DTYPE& inv_dx)
{
	DTYPE3 dls_x = InterpolateQuadratic3D_dxyz(ls, xg.x - 0.5f, xg.y - 0.5f, xg.z - 0.5f, dim, inv_dx);
	DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x - 0.5f, xg.y - 0.5f, xg.z - 0.5f, dim);
	return xg - normalize(dls_x) * ls_p * inv_dx;
}

inline __device__ DTYPE3 cuFindCloestPoint(DTYPE* ls, const DTYPE3& xg, const int3& dim, const DTYPE& inv_dx)
{
	DTYPE3 res = xg;
	for (int i = 0; i < 3; i++)
	{
		res = cuFindCloestPoint_OneIter(ls, res, dim, inv_dx);
	}
	return res;
}


inline __host__ __device__ DTYPE GetExtValue2D(DTYPE* val, int2 idx, int2& dim)
{
	DTYPE ref_sign = 1.f;

	cuRefIdx(idx.x, ref_sign, dim.x);
	cuRefIdx(idx.y, ref_sign, dim.y);

	int idx_val = idx.y + idx.x * dim.y;
	return val[idx_val] * ref_sign;
}

inline __host__ __device__ DTYPE InterpolateLinear2D(int* val, DTYPE x, DTYPE y, int2 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;

	int ix, iy;
	DTYPE fx, fy;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };

	int idx = iy + ix * dim.y;
	DTYPE res = val[idx] * wx[0] * wy[0];
	res += val[idx + 1] * wx[0] * wy[1];
	res += val[idx + dim.y] * wx[1] * wy[0];
	res += val[idx + dim.y + 1] * wx[1] * wy[1];
	return res;
}

inline __host__ __device__ DTYPE InterpolateLinear2D(DTYPE* val, DTYPE x, DTYPE y, int2 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;

	int ix, iy;
	DTYPE fx, fy;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };

	int idx = iy + ix * dim.y;
	DTYPE res = val[idx] * wx[0] * wy[0];
	//printf("idx: %d, val: %f\n", idx, val[idx]);
	res += val[idx + 1] * wx[0] * wy[1];
	res += val[idx + dim.y] * wx[1] * wy[0];
	res += val[idx + dim.y + 1] * wx[1] * wy[1];
	return res;
}

inline __host__ __device__ DTYPE2 InterpolateLinear2D_dxy(DTYPE* val, DTYPE x, DTYPE y, int2 dim, DTYPE inv_dx)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;

	int ix, iy;
	DTYPE fx, fy;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);

	DTYPE wx[2] = { 1.f - fx, fx };
	DTYPE wy[2] = { 1.f - fy, fy };

	DTYPE dwx[2] = { -inv_dx, inv_dx };
	DTYPE dwy[2] = { -inv_dx, inv_dx };

	int idx = iy + ix * dim.y;
	DTYPE2 res = { 0.f, 0.f };
	for (int j = 0; j < 2; j++)
	{
		for (int i = 0; i < 2; i++)
		{
			int idx_x = clamp(ix + j, 0, dim.x - 1);
			int idx_y = clamp(iy + i, 0, dim.y - 1);

			int idx = idx_y + idx_x * dim.y;
			res.x += dwx[j] * wy[i] * val[idx];
			res.y += wx[j] * dwy[i] * val[idx];
		}
	}
	return res;
}

inline __host__ __device__ DTYPE2 InterpolateQuadratic2D_dxyz_ext(DTYPE* val, DTYPE x, DTYPE y, int2& dim, DTYPE inv_dx)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	int ix = floor(x - 0.5f);
	int iy = floor(y - 0.5f);

	DTYPE fx = x - ix;
	DTYPE fy = y - iy;

	DTYPE wx[3] = { 0.5f * my_pow2(1.5f - fx), 0.75f - my_pow2(fx - 1.f), 0.5f * my_pow2(fx - 0.5f) };
	DTYPE wy[3] = { 0.5f * my_pow2(1.5f - fy), 0.75f - my_pow2(fy - 1.f), 0.5f * my_pow2(fy - 0.5f) };

	DTYPE dwx[3] = { (fx - 1.5f) * inv_dx, -2.f * (fx - 1.f) * inv_dx, (fx - 0.5f) * inv_dx };
	DTYPE dwy[3] = { (fy - 1.5f) * inv_dx, -2.f * (fy - 1.f) * inv_dx, (fy - 0.5f) * inv_dx };

	DTYPE2 res = { 0.f, 0.f};
	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++)
		{
			DTYPE val_c = GetExtValue2D(val, { ix + j, iy + i}, dim);
			res.x += dwx[j] * wy[i] * val_c;
			res.y += wx[j] * dwy[i] * val_c;
		}
	}
	return res;
}

inline __device__ void InterpolateVel3D_1c(DTYPE* val, DTYPE vel_p, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;

	int ix, iy, iz;
	DTYPE fx, fy, fz;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);
	cuGetIxFx(iz, fz, z, dim.z);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };
	DTYPE wz[2] = { 1 - fz, fz };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	for (int k = 0; k < 2; k++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int i = 0; i < 2; i++)
			{
				int ix_off = ix + j;
				int iy_off = iy + i;
				int iz_off = iz + k;
				int idx_x = clamp(ix_off, 0, dim.x - 1);
				int idx_y = clamp(iy_off, 0, dim.y - 1);
				int idx_z = clamp(iz_off, 0, dim.z - 1);

				int idx = idx_y + idx_x * dim.y + idx_z * slice;

				DTYPE cf = 0.f;
				if (ix_off >= 0 && ix_off < dim.x && iy_off >= 0 && iy_off < dim.y && iz_off >= 0 && iz_off < dim.z)
				{
					cf = vel_p * wx[j] * wy[i] * wz[k];
				}

				atomicAdd(&val[idx], cf);
			}
		}
	}
}

inline __device__ void InterpolateVel3D_1c(DTYPE* val, DTYPE* val_w, DTYPE vel_p, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	//DTYPE fx = x - ix;
	//DTYPE fy = y - iy;
	//DTYPE fz = z - iz;

	int ix, iy, iz;
	DTYPE fx, fy, fz;
	cuGetIxFx(ix, fx, x, dim.x);
	cuGetIxFx(iy, fy, y, dim.y);
	cuGetIxFx(iz, fz, z, dim.z);

	DTYPE wx[2] = { 1 - fx, fx };
	DTYPE wy[2] = { 1 - fy, fy };
	DTYPE wz[2] = { 1 - fz, fz };

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	for (int k = 0; k < 2; k++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int i = 0; i < 2; i++)
			{
				int ix_off = ix + j;
				int iy_off = iy + i;
				int iz_off = iz + k;
				int idx_x = clamp(ix_off, 0, dim.x - 1);
				int idx_y = clamp(iy_off, 0, dim.y - 1);
				int idx_z = clamp(iz_off, 0, dim.z - 1);

				int idx = idx_y + idx_x * dim.y + idx_z * slice;

				DTYPE cf = 0.f;
				DTYPE cf_w = 0.f;
				if (ix_off >= 0 && ix_off < dim.x && iy_off >= 0 && iy_off < dim.y && iz_off >= 0 && iz_off < dim.z)
				{
					cf_w = wx[j] * wy[i] * wz[k];
					cf = vel_p * cf_w;
				}

				atomicAdd(&val[idx], cf);
				atomicAdd(&val_w[idx], cf_w);
			}
		}
	}
}

inline __host__ __device__ void InterpolateQuadratic3D_dxyz_ext(DTYPE3& res1, DTYPE3& res2, DTYPE* val1, DTYPE* val2, 
	DTYPE x, DTYPE y, DTYPE z, int3& dim, DTYPE inv_dx, char dire)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	int ix = floor(x - 0.5f);
	int iy = floor(y - 0.5f);
	int iz = floor(z - 0.5f);

	DTYPE fx = x - ix;
	DTYPE fy = y - iy;
	DTYPE fz = z - iz;

	DTYPE wx[3] = { 0.5f * my_pow2(1.5f - fx), 0.75f - my_pow2(fx - 1.f), 0.5f * my_pow2(fx - 0.5f) };
	DTYPE wy[3] = { 0.5f * my_pow2(1.5f - fy), 0.75f - my_pow2(fy - 1.f), 0.5f * my_pow2(fy - 0.5f) };
	DTYPE wz[3] = { 0.5f * my_pow2(1.5f - fz), 0.75f - my_pow2(fz - 1.f), 0.5f * my_pow2(fz - 0.5f) };

	DTYPE dwx[3] = { (fx - 1.5f) * inv_dx, -2.f * (fx - 1.f) * inv_dx, (fx - 0.5f) * inv_dx };
	DTYPE dwy[3] = { (fy - 1.5f) * inv_dx, -2.f * (fy - 1.f) * inv_dx, (fy - 0.5f) * inv_dx };
	DTYPE dwz[3] = { (fz - 1.5f) * inv_dx, -2.f * (fz - 1.f) * inv_dx, (fz - 0.5f) * inv_dx };

	int slice = dim.x * dim.y;
	for (int k = 0; k < 3; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				//DTYPE val_c = GetExtValue_linear(val, { ix + j, iy + i, iz + k }, dim, dire);
				DTYPE val_c = GetExtValue(val1, { ix + j, iy + i, iz + k }, dim, dire);
				res1.x += dwx[j] * wy[i] * wz[k] * val_c;
				res1.y += wx[j] * dwy[i] * wz[k] * val_c;
				res1.z += wx[j] * wy[i] * dwz[k] * val_c;

				DTYPE val_c2 = GetExtValue(val2, { ix + j, iy + i, iz + k }, dim, dire);
				res2.x += dwx[j] * wy[i] * wz[k] * val_c2;
				res2.y += wx[j] * dwy[i] * wz[k] * val_c2;
				res2.z += wx[j] * wy[i] * dwz[k] * val_c2;
			}
		}
	}
}


inline __host__ __device__ DTYPE InterpolateLinear3D_clamp(DTYPE* val, DTYPE expect, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	int ix, iy, iz;
	cuGetIx(ix, x, dim.x);
	cuGetIx(iy, y, dim.y);
	cuGetIx(iz, z, dim.z);

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE max_val = val[idx];
	DTYPE min_val = max_val;

	DTYPE res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	idx += slice;
	res = val[idx];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	return max(min(expect, max_val), min_val);
}

inline __host__ __device__ DTYPE InterpolateLinear3D_revert(DTYPE* val, DTYPE expect, DTYPE revert_val, DTYPE x, DTYPE y, DTYPE z, int3 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	int ix, iy, iz;
	cuGetIx(ix, x, dim.x);
	cuGetIx(iy, y, dim.y);
	cuGetIx(iz, z, dim.z);

	int slice = dim.x * dim.y;
	int idx = iy + ix * dim.y + iz * slice;
	DTYPE max_val = val[idx];
	DTYPE min_val = max_val;

	DTYPE res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	idx += slice;
	res = val[idx];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = expect;
	if (expect > max_val || expect < min_val)
	{
		res = revert_val;
	}

	return res;
}

inline __host__ __device__ DTYPE InterpolateLinear2D_clamp(DTYPE* val, DTYPE expect, DTYPE x, DTYPE y, int2 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	int ix, iy, iz;
	cuGetIx(ix, x, dim.x);
	cuGetIx(iy, y, dim.y);

	int idx = iy + ix * dim.y;
	DTYPE max_val = val[idx];
	DTYPE min_val = max_val;

	DTYPE res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	return max(min(expect, max_val), min_val);
}

inline __host__ __device__ DTYPE InterpolateLinear2D_revert(DTYPE* val, DTYPE expect, DTYPE revert_val, DTYPE x, DTYPE y, int2 dim)
{
	//x = clamp(x, CLAMP_L, dim.x - CLAMP_R);
	//y = clamp(y, CLAMP_L, dim.y - CLAMP_R);
	//z = clamp(z, CLAMP_L, dim.z - CLAMP_R);

	//int ix = floor(x);
	//int iy = floor(y);
	//int iz = floor(z);

	int ix, iy, iz;
	cuGetIx(ix, x, dim.x);
	cuGetIx(iy, y, dim.y);

	int idx = iy + ix * dim.y;
	DTYPE max_val = val[idx];
	DTYPE min_val = max_val;

	DTYPE res = val[idx + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = val[idx + dim.y + 1];
	max_val = max(max_val, res);
	min_val = min(min_val, res);

	res = expect;
	if (expect > max_val || expect < min_val)
	{
		res = revert_val;
	}

	return res;
}

enum class InterpType :unsigned int
{
	IT_LINEAR,
	IT_QUADRATIC
};

class Interpolation
{
public:
	Interpolation() = default;
	~Interpolation() = default;

	static Interpolation* GetInstance();

private:
	static std::auto_ptr<Interpolation> instance_;
};

inline __host__ __device__ DTYPE2 cuGetVelocity_Quadratic2D_dxyz_ext(DTYPE* val, const DTYPE2& pos, int2& dim, DTYPE inv_dx)
{
	DTYPE2 xg = pos * inv_dx;
	DTYPE2 res = InterpolateQuadratic2D_dxyz_ext(val, xg.x, xg.y, dim, inv_dx);
	return make_DTYPE2(res.y, -res.x);
}

inline __host__ __device__ DTYPE3 cuGetVelocity_Quadratic3D_dxyz_ext(DTYPE* qx, DTYPE* qy, DTYPE* qz, 
	const DTYPE3& pos, int3& dim, DTYPE inv_dx)
{
	int3 dim_qx = dim + make_int3(0, 1, 1);
	int3 dim_qy = dim + make_int3(1, 0, 1);
	int3 dim_qz = dim + make_int3(1, 1, 0);

	DTYPE3 xg = pos * inv_dx;
	DTYPE3 res;
	DTYPE3 xgx = { xg.x - 0.5f, xg.y, xg.z };
	DTYPE3 dFx = InterpolateQuadratic3D_dxyz_ext(qx, xgx.x, xgx.y, xgx.z, dim_qx, inv_dx, 'x');
	DTYPE3 xgy = { xg.x, xg.y - 0.5f, xg.z };
	DTYPE3 dFy = InterpolateQuadratic3D_dxyz_ext(qy, xgy.x, xgy.y, xgy.z, dim_qy, inv_dx, 'y');
	DTYPE3 xgz = { xg.x, xg.y, xg.z - 0.5f };
	DTYPE3 dFz = InterpolateQuadratic3D_dxyz_ext(qz, xgz.x, xgz.y, xgz.z, dim_qz, inv_dx, 'z');

	res.x = dFz.y - dFy.z;
	res.y = dFx.z - dFz.x;
	res.z = dFy.x - dFx.y;

	return res;
}

//Monotonic Cubic Interpolation
inline __host__ __device__ DTYPE cuGetExtValue(DTYPE* val, int ix, int iy, int2 dim, char ref_dire)
{
	DTYPE sign = 1.f;
	if (ref_dire == 'x')
	{
		if (ix < 0)
		{
			ix = 1;
			sign = -1.f;
		}
		if (ix >= dim.x)
		{
			ix = dim.x - 2;
			sign = -1.f;
		}
		iy = clamp(iy, 0, dim.y - 1);
	}
	else if (ref_dire == 'y')
	{
		if (iy < 0)
		{
			iy = 1;
			sign = -1.f;
		}
		if (iy >= dim.y)
		{
			iy = dim.y - 2;
			sign = -1.f;
		}
		ix = clamp(ix, 0, dim.x - 1);
	}
	return sign * val[INDEX2D(iy, ix, dim)];
}

inline __host__ __device__ DTYPE cuComputeMonotonicCubic(DTYPE mid_vals[4], DTYPE fx)
{
	
	DTYPE delta = mid_vals[2] - mid_vals[1];
	DTYPE sign_delta = 0.f;
	if (delta > 0.f)
		sign_delta = 1.f;
	else if (delta < 0.f)
		sign_delta = -1.f;


	DTYPE d0 = 0.5 * (mid_vals[2] - mid_vals[0]);
	if (d0 < 0.f)
		d0 = -d0;
	d0 *= sign_delta;

	DTYPE d1 = 0.5 * (mid_vals[3] - mid_vals[1]);
	if (d1 < 0.f)
		d1 = -d1;
	d1 *= sign_delta;

	DTYPE a3 = d0 + d1 - 2.f * delta;
	DTYPE a2 = 3.f * delta - 2.f * d0 - d1;
	
	return a3 * fx * fx * fx + a2 * fx * fx + d0 * fx + mid_vals[1];
}

inline __host__ __device__ DTYPE InterpolateMonotonicCubic2D(DTYPE* val, DTYPE x, DTYPE y, int2 dim, char ref_dire)
{
	int ix, iy;
	DTYPE fx, fy;
	//cuGetIxFx(ix, fx, x, dim.x);
	//cuGetIxFx(iy, fy, y, dim.y);
	ix = floor(x);
	iy = floor(y);
	fx = x - ix;
	fy = y - iy;

	DTYPE mid_vals[4];
	DTYPE tmp[4];

	for (int i = 0; i < 4; i++)
	{
		int iy_off = iy + i - 1;
		tmp[0] = cuGetExtValue(val, ix - 1, iy_off, dim, ref_dire);
		tmp[1] = cuGetExtValue(val, ix, iy_off, dim, ref_dire);
		tmp[2] = cuGetExtValue(val, ix + 1, iy_off, dim, ref_dire);
		tmp[3] = cuGetExtValue(val, ix + 2, iy_off, dim, ref_dire);
		mid_vals[i] = cuComputeMonotonicCubic(tmp, fx);
	}

	return cuComputeMonotonicCubic(mid_vals, fy);
}

inline __host__ __device__ DTYPE cuGetExtValue3D(DTYPE* val, int ix, int iy, int iz, int3 dim, char ref_dire)
{
	DTYPE sign = 1.f;
	if (ref_dire == 'x')
	{
		if (ix < 0)
		{
			ix = 1;
			sign = -1.f;
		}
		if (ix >= dim.x)
		{
			ix = dim.x - 2;
			sign = -1.f;
		}
		iz = clamp(iz, 0, dim.z - 1);
	}
	else if (ref_dire == 'y')
	{
		if (iy < 0)
		{
			iy = 1;
			sign = -1.f;
		}
		if (iy >= dim.y)
		{
			iy = dim.y - 2;
			sign = -1.f;
		}
		ix = clamp(ix, 0, dim.x - 1);
		iz = clamp(iz, 0, dim.z - 1);
	}
	else if (ref_dire == 'z')
	{
		if (iz < 0)
		{
			iz = 1;
			sign = -1.f;
		}
		if (iz >= dim.z)
		{
			iz = dim.z - 2;
			sign = -1.f;
		}
		ix = clamp(ix, 0, dim.x - 1);
		iy = clamp(iy, 0, dim.y - 1);
	}
	return sign * val[INDEX3D(iy, ix, iz, dim)];
}

inline __host__ __device__ DTYPE InterpolateMonotonicCubic3D(DTYPE* val, DTYPE x, DTYPE y, DTYPE z, int3 dim, char ref_dire)
{
	int ix, iy, iz;
	DTYPE fx, fy, fz;
	//cuGetIxFx(ix, fx, x, dim.x);
	//cuGetIxFx(iy, fy, y, dim.y);
	ix = floor(x);
	iy = floor(y);
	iz = floor(z);
	fx = x - ix;
	fy = y - iy;
	fz = z - iz;

	DTYPE mid_vals_2d[4][4];
	DTYPE mid_vals[4];
	DTYPE tmp[4];

	for (int k = 0; k < 4; k++)
	{
		for (int i = 0; i < 4; i++)
		{
			int iy_off = iy + i - 1;
			int iz_off = iz + k - 1;
			tmp[0] = cuGetExtValue3D(val, ix - 1, iy_off, iz_off, dim, ref_dire);
			tmp[1] = cuGetExtValue3D(val, ix, iy_off, iz_off, dim, ref_dire);
			tmp[2] = cuGetExtValue3D(val, ix + 1, iy_off, iz_off, dim, ref_dire);
			tmp[3] = cuGetExtValue3D(val, ix + 2, iy_off, iz_off, dim, ref_dire);
			mid_vals_2d[k][i] = cuComputeMonotonicCubic(tmp, fx);
		}
	}

	for (int i = 0; i < 4; i++)
	{
		mid_vals[i] = cuComputeMonotonicCubic(mid_vals_2d[i], fy);
	}

	return cuComputeMonotonicCubic(mid_vals, fz);
}


#endif