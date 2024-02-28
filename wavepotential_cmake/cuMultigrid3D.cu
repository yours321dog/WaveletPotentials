#include "cuMultigrid3D.cuh"
#include "cuMemoryManager.cuh"
#include "cudaMath.cuh"

CuMultigrid3D::CuMultigrid3D(int3 dim, DTYPE3 dx, char bc, MG3D_TYPE mg3d_type) : dim_(dim), max_level_(0), dx_(dx),
	bc_(bc), size_(dim.x * dim.y * dim.z), mg3d_type_(mg3d_type)
{
	int min_dim = std::min(std::min(dim.x, dim.y), dim.z);
	dx_inv_ = make_DTYPE3(1.f / dx.x, 1.f / dx.y, 1.f / dx.z);
	int ext = 0;

	if (mg3d_type == MG3D_TYPE::MG3D_2N)
	{
		dim_rs_from_dim_ = [](int3 dim_in) -> int3 { return make_int3(dim_in.x >> 1, dim_in.y >> 1, dim_in.z >> 1); };
	}
	else if (mg3d_type == MG3D_TYPE::MG3D_2NP1)
	{
		dim_rs_from_dim_ = [](int3 dim_in) -> int3 { return make_int3(((dim_in.x - 1) >> 1) + 1,
			((dim_in.y - 1) >> 1) + 1, ((dim_in.z - 1) >> 1) + 1); };
		ext = 1;
	}
	else if (mg3d_type == MG3D_TYPE::MG3D_2NM1)
	{
		dim_rs_from_dim_ = [](int3 dim_in) -> int3 { return make_int3(((dim_in.x + 1) >> 1) - 1,
			((dim_in.y + 1) >> 1) - 1, ((dim_in.z + 1) >> 1) - 1); };
		ext = 1;
	}
	else if (mg3d_type == MG3D_TYPE::MG3D_RN || mg3d_type == MG3D_TYPE::MG3D_RNC4)
	{
		dim_rs_from_dim_ = [](int3 dim_in) -> int3 { 
			int3 dim_out;
			if ((dim_in.x & 1) == 0)
			{
				dim_out.x = dim_in.x >> 1;
			}
			else
			{
				dim_out.x = ((dim_in.x + 1) >> 1) - 1;
			}
			if ((dim_in.y & 1) == 0)
			{
				dim_out.y = dim_in.y >> 1;
			}
			else
			{
				dim_out.y = ((dim_in.y + 1) >> 1) - 1;
			}
			if ((dim_in.z & 1) == 0)
			{
				dim_out.z = dim_in.z >> 1;
			}
			else
			{
				dim_out.z = ((dim_in.z + 1) >> 1) - 1;
			}
			return dim_out;
		};
		ext = 1;
	}
	int3 dim_rs = dim;
	while (min_dim > ext)
	{
		sl_n_ = std::max(std::max(dim_rs.x, dim_rs.y), dim_rs.z);
		int size_coarse = dim_rs.x * dim_rs.y * dim_rs.z;

		// create coarse grids for residual
		DTYPE* dev_f;
		checkCudaErrors(cudaMalloc((void**)&dev_f, size_coarse * sizeof(DTYPE)));
		checkCudaErrors(cudaMemset(dev_f, 0, sizeof(DTYPE) * size_coarse));
		f_coarses_.push_back(dev_f);

		// create coarse grids for resolution
		DTYPE* dev_v;
		checkCudaErrors(cudaMalloc((void**)&dev_v, size_coarse * sizeof(DTYPE)));
		checkCudaErrors(cudaMemset(dev_v, 0, sizeof(DTYPE) * size_coarse));
		v_coarses_.push_back(dev_v);

		dim_rs = dim_rs_from_dim_(dim_rs);
		min_dim = std::min(std::min(dim_rs.x, dim_rs.y), dim_rs.z);
		++max_level_;
	}
}

CuMultigrid3D::~CuMultigrid3D()
{
	for (DTYPE* ptr : f_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : v_coarses_)
	{
		cudaFree(ptr);
	}
}

__global__ void cuGetResidual3D_d_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
	}
}

__global__ void cuGetResidual3D_n_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuGetResidual3D_z_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		//int idx_xz = idx / dim.y;
		//int idx_y = idx & (dim.y - 1);
		//int idx_z = idx_xz / dim.x;
		//int idx_x = idx_xz & (dim.x - 1);
		//int slice_xy = dim.x * dim.y;

		//DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		//DTYPE off_d = 0.f;

		//// bottom
		//if (idx_y > 0)
		//	off_d += -src[idx - 1] * dxp2_inv.y;

		//// top
		//if (idx_y < dim.y - 1)
		//	off_d += -src[idx + 1] * dxp2_inv.y;

		//// left
		//if (idx_x > 0)
		//	off_d += -src[idx - dim.y] * dxp2_inv.x;

		//// right
		//if (idx_x < dim.x - 1)
		//	off_d += -src[idx + dim.y] * dxp2_inv.x;

		//// left
		//if (idx_z > 0)
		//	off_d += -src[idx - slice_xy] * dxp2_inv.z;

		//// right
		//if (idx_z < dim.z - 1)
		//	off_d += -src[idx + slice_xy] * dxp2_inv.z;

		//dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
		//if (max_size < 5)
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
	}
}

__global__ void cuGetResidual3D_d_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
	}
}

__global__ void cuGetResidual3D_n_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuGetResidual3D_z_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= 0;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= 0;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
	}
}

__global__ void cuGetResidual3D_d_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += coefs.y;
		else
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += coefs.y;
		else
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x == 0)
			a_diag += coefs.x;
		else
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += coefs.x;
		else
			off_d += -src[idx + dim.y] * coefs.x;

		// front
		if (idx_z == 0)
			a_diag += coefs.z;
		else
			off_d += -src[idx - slice_xy] * coefs.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += coefs.z;
		else
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
	}
}

__global__ void cuGetResidual3D_n_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= coefs.y;
		else
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= coefs.y;
		else
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x == 0)
			a_diag -= coefs.x;
		else
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= coefs.x;
		else
			off_d += -src[idx + dim.y] * coefs.x;

		// front
		if (idx_z == 0)
			a_diag -= coefs.z;
		else
			off_d += -src[idx - slice_xy] * coefs.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= coefs.z;
		else
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuGetResidual3D_z_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * coefs.x;

		// left
		if (idx_z > 0)
			off_d += -src[idx - slice_xy] * coefs.z;

		// right
		if (idx_z < dim.z - 1)
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = f[idx] - (a_diag * src[idx] + off_d);
		////if (max_size < 18)
		//if (max_size > 18 && max_size < 50)
		////if (max_size > 50)
		//	printf("lpx: %f, idx: %d, %d, %d, coefs: %f, %f %f, a_diag: %f\n", lpx, idx_x, idx_y, idx_z, coefs.x, coefs.y, coefs.z, a_diag);
	}
}

void CuMultigrid3D::Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	if (pre_v != nullptr)
	{
		cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * size_, cudaMemcpyDeviceToDevice);
	}
	else
	{
		cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * size_);
	}
	cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * size_, cudaMemcpyDeviceToDevice);
	if (mg3d_type_ == MG3D_TYPE::MG3D_2N)
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycle(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}
	else if (mg3d_type_ == MG3D_TYPE::MG3D_2NP1)
	{
		for (int i = 0; i < n_iters; i++)
		{
			//SolveOneVCycle(dim_, dx_, nu1, nu2, weight, bc_, 0);
			SolveOneVCycleP1(dim_, dx_, nu1, nu2, weight, 1.f, bc_, 0);
		}
	}
	else if (mg3d_type_ == MG3D_TYPE::MG3D_2NM1)
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycleM1(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}
	else if (mg3d_type_ == MG3D_TYPE::MG3D_RN)
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycleRN(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}
	else if (mg3d_type_ == MG3D_TYPE::MG3D_RNC4)
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycleRN_c4(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}
	cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * size_, cudaMemcpyDeviceToDevice);
}

void CuMultigrid3D::SolveOneVCycle(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	// do weightedjacobi solver
	CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 2)
	{
		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		int max_size = dim.x * dim.y * dim.z;
		int size_rs = dim_rs.x * dim_rs.y * dim_rs.z;
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", max_size);
		switch (bc)
		{
		case 'd':
			//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_d_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_z_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_n_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		CuRestrict3D::GetInstance()->Solve(f_coarses_[deep_level + 1], dev_res, dim);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * size_rs));
		SolveOneVCycle(dim_rs, make_DTYPE3(dx.x * 2.f, dx.y * 2.f, dx.z * 2.f), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuProlongate3D::GetInstance()->SolveAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim_rs);

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//printf("small level: %d, dim: %d, %d\n", deep_level, dim.x, dim.y);
		//if (sl_n_ == 1 && bc == 'n')
		//{
		//	checkCudaErrors(cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE)));
		//}
		//else
		//{
		//	checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
		//	checkCudaErrors(getrs_func(handle_, CUBLAS_OP_N, sl_n_, 1, dev_A_, sl_n_, d_pivot_, v_coarses_[deep_level], sl_n_, d_info_));
		//}
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);

	}
}

void CuMultigrid3D::SolveOneVCycleP1(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, DTYPE lpx, char bc, int deep_level)
{
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	// do weightedjacobi solver
	CuWeightedJacobi3D::GetInstance()->SolveP1(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, lpx, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		int max_size = dim.x * dim.y * dim.z;
		int size_rs = dim_rs.x * dim_rs.y * dim_rs.z;
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", max_size);
		switch (bc)
		{
		case 'd':
			//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_d_p1 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, lpx, max_size);
			break;
		case 'z':
			//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_z_p1 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, lpx, max_size);
			break;
		default:
			//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_n_p1 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, lpx, max_size);
			break;
		}
		//CudaPrintfMat(dev_res, dim);
		CuRestrict3D::GetInstance()->Solve(f_coarses_[deep_level + 1], dev_res, dim);
		//CudaPrintfMat(f_coarses_[deep_level + 1], dim_rs);


		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * size_rs));
		SolveOneVCycleP1(dim_rs, make_DTYPE3(dx.x * 2.f, dx.y * 2.f, dx.z * 2.f), nu1, nu2, weight, lpx * 0.5f, bc, deep_level + 1);

		// do prolongation

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuProlongate3D::GetInstance()->SolveAddP1(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim_rs);

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuWeightedJacobi3D::GetInstance()->SolveP1(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, lpx, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//printf("small level: %d, dim: %d, %d\n", deep_level, dim.x, dim.y);
		//if (sl_n_ == 1 && bc == 'n')
		//{
		//	checkCudaErrors(cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE)));
		//}
		//else
		//{
		//	checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
		//	checkCudaErrors(getrs_func(handle_, CUBLAS_OP_N, sl_n_, 1, dev_A_, sl_n_, d_pivot_, v_coarses_[deep_level], sl_n_, d_info_));
		//}
		CuWeightedJacobi3D::GetInstance()->SolveP1(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, lpx, bc);

	}
}

void CuMultigrid3D::SolveOneVCycleM1(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	// do weightedjacobi solver
	CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		int max_size = dim.x * dim.y * dim.z;
		int size_rs = dim_rs.x * dim_rs.y * dim_rs.z;
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", max_size);
		switch (bc)
		{
		case 'd':
			//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_d_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_z_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_n_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		CuRestrict3D::GetInstance()->SolveM1(f_coarses_[deep_level + 1], dev_res, dim);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * size_rs));
		SolveOneVCycleM1(dim_rs, make_DTYPE3(dx.x * 2.f, dx.y * 2.f, dx.z * 2.f), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuProlongate3D::GetInstance()->SolveAddM1(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim_rs);

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//printf("small level: %d, dim: %d, %d\n", deep_level, dim.x, dim.y);
		//if (sl_n_ == 1 && bc == 'n')
		//{
		//	checkCudaErrors(cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE)));
		//}
		//else
		//{
		//	checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
		//	checkCudaErrors(getrs_func(handle_, CUBLAS_OP_N, sl_n_, 1, dev_A_, sl_n_, d_pivot_, v_coarses_[deep_level], sl_n_, d_info_));
		//}
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);

	}
}

void CuMultigrid3D::SolveOneVCycleRN(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	// do weightedjacobi solver
	CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		int max_size = dim.x * dim.y * dim.z;
		int size_rs = dim_rs.x * dim_rs.y * dim_rs.z;
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", max_size);
		switch (bc)
		{
		case 'd':
			//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_d_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_z_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_n_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		CuRestrict3D::GetInstance()->SolveRN(f_coarses_[deep_level + 1], dev_res, dim);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * size_rs));
		SolveOneVCycleRN(dim_rs, make_DTYPE3(dx.x * 2.f, dx.y * 2.f, dx.z * 2.f), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuProlongate3D::GetInstance()->SolveAddRN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs);

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
	}
}

void CuMultigrid3D::SolveOneVCycleRN_c4(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	// do weightedjacobi solver
	CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		int max_size = dim.x * dim.y * dim.z;
		int size_rs = dim_rs.x * dim_rs.y * dim_rs.z;
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", max_size);
		switch (bc)
		{
		case 'd':
			//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_d_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_z_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			cuGetResidual3D_n_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		CuRestrict3D::GetInstance()->SolveRN_c4(f_coarses_[deep_level + 1], dev_res, dim);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * size_rs));
		SolveOneVCycleRN_c4(dim_rs, make_DTYPE3(dx.x * 2.f, dx.y * 2.f, dx.z * 2.f), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuProlongate3D::GetInstance()->SolveAddRN_c4(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs);

		//CudaPrintfMat(v_coarses_[deep_level], dim);
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		CuWeightedJacobi3D::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
	}
}

DTYPE CuMultigrid3D::GetRhs(DTYPE* v, DTYPE* f, int3 dim, DTYPE3 dx, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE3 dxp2_inv = make_DTYPE3(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y), 1.f / (dx.z * dx.z));

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidual3D_d_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	case 'z':
		//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidual3D_z_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	default:
		//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidual3D_n_common << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}