#include "cuRestrict3D.cuh"
#include "cuMemoryManager.cuh"

std::auto_ptr<CuRestrict3D> CuRestrict3D::instance_;

CuRestrict3D* CuRestrict3D::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuRestrict3D>(new CuRestrict3D); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

__global__ void cuRestrict3D_p2(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		
		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y * 2;
		int idx_z_src = idx_z * 2;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * slice_xy;
		int idx_src_z = idx_src + slice_xy;
		dst[idx] = 0.125f * (src[idx_src] + src[idx_src + 1] + src[idx_src + dim_src.y] + src[idx_src + dim_src.y + 1]
			+ src[idx_src_z] + src[idx_src_z + 1] + src[idx_src_z + dim_src.y] + src[idx_src_z + dim_src.y + 1]);
	}
}

__global__ void cuRestrict3D_common(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y * 2;
		int idx_z_src = idx_z * 2;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * slice_xy;
		int idx_src_z = idx_src + slice_xy;

		DTYPE val = src[idx_src];
		if (idx_y_src + 1 < dim_src.y)
		{
			val += src[idx_src + 1];
			if (idx_x_src + 1 < dim_src.x)
			{
				val += src[idx_src + dim_src.y + 1];
				val += src[idx_src + dim_src.y];
			}
		}
		else if (idx_x_src + 1 < dim_src.x)
		{
			val += src[idx_src + dim_src.y];
		}
		if (idx_z_src + 1 < dim_src.z)
		{
			val += src[idx_src_z];
			if (idx_y_src + 1 < dim_src.y)
			{
				val += src[idx_src_z + 1];
				if (idx_x_src + 1 < dim_src.x)
				{
					val += src[idx_src_z + dim_src.y + 1];
					val += src[idx_src_z + dim_src.y];
				}
			}
			else if (idx_x_src + 1 < dim_src.x)
			{
				val += src[idx_src_z + dim_src.y];
			}
		}
		val *= 0.125f;
		//val *= 0.5f;
		//val *= 0.0625f;
		dst[idx] = val;
		//printf("idx: %d, %d, %d, idx_src: %d, %d, %d, val: %f\n", idx_x, idx_y, idx_z, idx_x_src, idx_y_src, idx_z_src, val);
	}
}

__global__ void cuRestrict3D_m1(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = (idx_x + 1) * 2 - 1;
		int idx_y_src = (idx_y + 1) * 2 - 1;
		int idx_z_src = (idx_z + 1) * 2 - 1;

		DTYPE val = 0.f;

		for (int k = -1; k <= 1; k++)
		{
			DTYPE coef_k = ((2.f - abs(float(k))) * 0.25f);
			for (int j = -1; j <= 1; j++)
			{
				DTYPE coef_jk = coef_k * ((2.f - abs(float(j))) * 0.25f);
				for (int i = -1; i <= 1; i++)
				{
					DTYPE coef_ijk = ((2.f - abs(float(i))) * 0.25f) * coef_jk;
					int idx_src = (idx_y_src + i) + (idx_x_src + j) * dim_src.y + (idx_z_src + k) * slice_xy;
					val += src[idx_src] * coef_ijk;
				}
			}
		}
		dst[idx] = val;
	}
}

__global__ void cuRestrict3D_m1_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x;
		int idx_y_src = idx_y;
		int idx_z_src = idx_z * 2 + 1;
		int slice = dim_src.x * dim_src.y;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);
		dst[idx_dst] = 0.25f * (src[idx_src - slice] + src[idx_src + slice]) + 0.5f * src[idx_src];
	}
}

__global__ void cuRestrict3D_m1_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x_src, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);
		dst[idx_dst] = 0.25f * (src[idx_src - dim_src.y] + src[idx_src + dim_src.y]) + 0.5f * src[idx_src];
	}
}

__global__ void cuRestrict3D_m1_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);
		dst[idx_dst] = 0.25f * (src[idx_src - 1] + src[idx_src + 1]) + 0.5f * src[idx_src];
	}
}

__global__ void cuRestrict3D_p2_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		//int idx_x_src = idx_x;
		//int idx_y_src = idx_y;
		int idx_z_src = idx_z * 2 + 1;
		int slice = dim_src.x * dim_src.y;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src - slice] + src[idx_src]);
	}
}

__global__ void cuRestrict3D_p2_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2 + 1;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x_src, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src - dim_src.y] + src[idx_src]);
	}
}

__global__ void cuRestrict3D_p2_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);
		dst[idx_dst] = 0.5f * (src[idx_src - 1] + src[idx_src]);
	}
}

__global__ void cuRestrict3D_p2_c4_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		//int idx_x_src = idx_x;
		//int idx_y_src = idx_y;
		int idx_z_src = idx_z * 2 + 1;
		int slice = dim_src.x * dim_src.y;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);
		DTYPE res = 0.375f * (src[idx_src - slice] + src[idx_src]);
		if (idx_z > 0)
		{
			res += 0.125f * src[idx_src - 2 * slice];
		}
		if (idx_z < dim.z - 1)
		{
			res += 0.125f * src[idx_src + slice];
		}
		dst[idx_dst] = res;
	}
}

__global__ void cuRestrict3D_p2_c4_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2 + 1;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x_src, idx_z_src, dim_src);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z_src, dim_src);
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

__global__ void cuRestrict3D_p2_c4_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_x_src = idx_x * 2 + 1;
		int idx_y_src = idx_y * 2 + 1;
		int idx_z_src = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);
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

void CuRestrict3D::Solve(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int3 dim_dst = { dim.x >> 1, dim.y >> 1, dim.z >> 1 };
	int3 ext = { dim.x & 1, dim.y & 1, dim.z & 1 };
	dim_dst.x += ext.x;
	dim_dst.y += ext.y;
	dim_dst.z += ext.z;

	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;
	//cuRestrict3D_p2 << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	cuRestrict3D_common << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuRestrict3D::SolveP2(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		return dim >> 1;
	};
	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };

	int3 dim_thread = { dim.x, dim.y, dim_dst.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_p2_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	//CudaPrintfMat(tmp, dim);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_thread, dim, thread_size);
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y * dim_dst.z;
	cuRestrict3D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	//CudaPrintfMat(dst, dim_dst);
}

void CuRestrict3D::SolveM1(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		return ((dim + 1) >> 1) - 1;
	};
	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };

	int3 dim_thread = { dim.x, dim.y, dim_dst.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_m1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	//CudaPrintfMat(tmp, dim);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_thread, dim, thread_size);
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y * dim_dst.z;
	cuRestrict3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	//CudaPrintfMat(dst, dim_dst);
}

void CuRestrict3D::SolveM1Old(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	auto rs_dim = [](int dim) -> int {
		return ((dim + 1) >> 1) - 1;
	};
	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };

	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;
	cuRestrict3D_m1 << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuRestrict3D::SolveRN(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return ((dim + 1) >> 1) - 1;
	};

	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };
	int3 dim_thread = { dim.x, dim.y, dim_dst.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y * dim_dst.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim, thread_size);
	}
}

void CuRestrict3D::SolveRN_c4(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return ((dim + 1) >> 1) - 1;
	};

	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };
	int3 dim_thread = { dim.x, dim.y, dim_dst.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_c4_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_c4_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (src, tmp, dim_thread, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (src, tmp, dim_thread, dim, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	thread_size = dim_dst.x * dim_dst.y * dim_dst.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, src, dim_dst, dim, thread_size);
	}
	else
	{
		cuRestrict3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, src, dim_dst, dim, thread_size);
	}
	cudaCheckError(cudaGetLastError());
}

/***********************************************************************************************************/
__global__ void cuRestrict3D_p1_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z, dim_src);

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

__global__ void cuRestrict3D_p1_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x, idx_z, dim_src);

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

__global__ void cuRestrict3D_p1_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);

		DTYPE val = 0.5f * src[idx_src];
		if (idx_z < dim.z - 1)
		{
			val += 0.5f * src[idx_src + slice_xy];
		}
		else
		{
			val *= 2.f;
		}

		dst[idx_dst] = val;
	}
}

__global__ void cuRestrict3D_p1_main_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z, dim_src);

		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p1_main_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x, idx_z, dim_src);

		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p1_main_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);

		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p2_lesser_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z, dim_src);

		dst[idx_dst] = 0.5f * (src[idx_src] + src[idx_src + dim_src.y]);
	}
}

__global__ void cuRestrict3D_p2_lesser_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x, idx_z, dim_src);

		dst[idx_dst] = 0.5f * (src[idx_src + 1] + src[idx_src]);
	}
}

__global__ void cuRestrict3D_p2_lesser_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);

		dst[idx_dst] = 0.5f * (src[idx_src + slice_xy] + src[idx_src]);
	}
}

__global__ void cuRestrict3D_p2_main_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z, dim_src);

		if (idx_x == dim.x - 1)
		{
			idx_src -= dim_src.y;
			//idx_dst -= dim_src.y;
		}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p2_main_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x, idx_z, dim_src);

		if (idx_y == dim.y - 1)
		{
			idx_src -= 1;
		}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p2_main_z(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);

		if (idx_z == dim.z - 1)
		{
			idx_src -= slice_xy;
		}
		dst[idx_dst] = src[idx_src];
	}
}

__global__ void cuRestrict3D_p1_dx_x(DTYPE* dst, DTYPE* src, DTYPE* dx, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x_src, idx_z, dim_src);

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

__global__ void cuRestrict3D_p1_dy_y(DTYPE* dst, DTYPE* src, DTYPE* dy, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_y_src = idx_y * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x, idx_z, dim_src);

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

__global__ void cuRestrict3D_p1_dz_z(DTYPE* dst, DTYPE* src, DTYPE* dz, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int slice_xy = dim_src.x * dim_src.y;

		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z_src, dim_src);

		DTYPE val = dz[idx_z_src] * src[idx_src];
		DTYPE w = dz[idx_z_src];
		if (idx_z_src < dim_src.z - 1)
		{
			val += dz[idx_z_src + 1] * src[idx_src + slice_xy];
			w += dz[idx_z_src + 1];
		}

		dst[idx_dst] = val / w;
	}
}

void CuRestrict3D::SolveP1RN(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	//int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };
	int3 dim_src = dim;
	int3 dim_thread = { rs_dim(dim.x), dim.y, dim.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp, dim_thread);
	//CudaPrintfMat(tmp, dim);

	dim_src.x = dim_thread.x;
	dim_thread.y = rs_dim(dim.y);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp_2, dim_thread);

	dim_src.y = dim_thread.y;
	dim_thread.z = rs_dim(dim.z);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_lesser_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(dst, dim_thread);
}

void CuRestrict3D::SolveP1RN(DTYPE* dst, DTYPE* src, DTYPE* dx, DTYPE* dy, DTYPE* dz, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	//int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };
	int3 dim_src = dim;
	int3 dim_thread = { rs_dim(dim.x), dim.y, dim.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_p1_dx_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dx, dim_thread, dim_src, thread_size);
	//CudaPrintfMat(tmp, dim_thread);
	//CudaPrintfMat(tmp, dim);

	dim_src.x = dim_thread.x;
	dim_thread.y = rs_dim(dim.y);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_p1_dy_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dy, dim_thread, dim_src, thread_size);
	//CudaPrintfMat(tmp_2, dim_thread);

	dim_src.y = dim_thread.y;
	dim_thread.z = rs_dim(dim.z);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuRestrict3D_p1_dz_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dz, dim_thread, dim_src, thread_size);
}

void CuRestrict3D::SolveP1Ax(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		return (dim >> 1) + 1;
	};

	int3 dim_src = dim;
	int3 dim_thread = { rs_dim_main(dim.x), dim.y, dim.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	//CudaPrintfMat(src, dim);
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_main_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_main_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp, dim_thread);

	dim_src.x = dim_thread.x;
	dim_thread.y = rs_dim(dim.y);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp_2, dim_thread);

	dim_src.y = dim_thread.y;
	dim_thread.z = rs_dim(dim.z);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_lesser_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(dst, dim_thread);
}

void CuRestrict3D::SolveP1Ay(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		return (dim >> 1) + 1;
	};

	int3 dim_src = dim;
	int3 dim_thread = { rs_dim(dim.x), dim.y, dim.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	//CudaPrintfMat(src, dim);
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp, dim);

	dim_src.x = dim_thread.x;
	dim_thread.y = rs_dim_main(dim.y);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_main_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_main_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}

	dim_src.y = dim_thread.y;
	dim_thread.z = rs_dim(dim.z);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_lesser_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
}

void CuRestrict3D::SolveP1Az(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	auto rs_dim_main = [](int dim) -> int {
		return (dim >> 1) + 1;
	};

	int3 dim_src = dim;
	int3 dim_thread = { rs_dim(dim.x), dim.y, dim.z };
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	//CudaPrintfMat(src, dim);
	if ((dim.x & 1) == 0)
	{
		cuRestrict3D_p2_lesser_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp, dim_thread);

	dim_src.x = dim_thread.x;
	dim_thread.y = rs_dim(dim.y);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.y & 1) == 0)
	{
		cuRestrict3D_p2_lesser_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(tmp_2, dim_thread);

	dim_src.y = dim_thread.y;
	dim_thread.z = rs_dim_main(dim.z);
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	if ((dim.z & 1) == 0)
	{
		cuRestrict3D_p2_main_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	else
	{
		cuRestrict3D_p1_main_z << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_thread, dim_src, thread_size);
	}
	//CudaPrintfMat(dst, dim_thread);
}


/******************************************Local Index********************************************************/
__global__ void cuRestrict3D_p1_dx_whole(DTYPE* dst, DTYPE* src, DTYPE* dx, DTYPE* dy, DTYPE* dz, int3 dim, int3 dim_src, int* index, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x * 2;
		int idx_y_src = idx_y * 2;
		int idx_z_src = idx_z * 2;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);

		DTYPE wx_l = dx[idx_x_src];
		DTYPE wy_l = dy[idx_y_src];
		DTYPE wz_l = dz[idx_z_src];
		
		bool x_has_next = idx_x_src < dim_src.x - 1;
		bool y_has_next = idx_y_src < dim_src.y - 1;
		bool z_has_next = idx_z_src < dim_src.z - 1;

		DTYPE wx_r = x_has_next ? dx[idx_x_src + 1] : 0.f;
		DTYPE wy_r = y_has_next ? dy[idx_y_src + 1] : 0.f;
		DTYPE wz_r = z_has_next ? dz[idx_z_src + 1] : 0.f;

		int offset_next_x = x_has_next ? dim_src.y : 0;
		int offset_next_y = y_has_next ? 1 : 0;
		int offset_next_z = z_has_next ? dim_src.x * dim_src.y : 0;

		DTYPE val = wx_l * wy_l * wz_l * src[idx_src];
		val += wx_l * wy_r * wz_l * src[idx_src + offset_next_y];
		val += wx_r * wy_l * wz_l * src[idx_src + offset_next_x];
		val += wx_r * wy_r * wz_l * src[idx_src + offset_next_y + offset_next_x];
		
		idx_src += offset_next_z;
		val += wx_l * wy_l * wz_r * src[idx_src];
		val += wx_l * wy_r * wz_r * src[idx_src + offset_next_y];
		val += wx_r * wy_l * wz_r * src[idx_src + offset_next_x];
		val += wx_r * wy_r * wz_r * src[idx_src + offset_next_y + offset_next_x];

		dst[idx_dst] = val / ((wx_l + wx_r) * (wy_l + wy_r) * (wz_l + wz_r));
	}
}

void CuRestrict3D::SolveP1RN(DTYPE* dst, DTYPE* src, int* index, int size_idx, DTYPE* dx, DTYPE* dy, DTYPE* dz, int3 dim, char bc)
{
	auto rs_dim = [](int dim) -> int {
		if ((dim & 1) == 0)
			return dim >> 1;
		else
			return (dim >> 1) + 1;
	};

	//printf("in dim: %d, %d, %d\n", dim.x, dim.y, dim.z);

	int3 dim_dst = { rs_dim(dim.x), rs_dim(dim.y), rs_dim(dim.z) };
	cuRestrict3D_p1_dx_whole << <BLOCKS(size_idx), THREADS(size_idx) >> > (dst, src, dx, dy, dz, dim_dst, dim, index, size_idx);
	cudaCheckError(cudaGetLastError());
}