#include "cuProlongate3D.cuh"
#include "cuMemoryManager.cuh"

CuProlongate3D::CuProlongate3D() = default;
CuProlongate3D::~CuProlongate3D() = default;

std::auto_ptr<CuProlongate3D> CuProlongate3D::instance_;

CuProlongate3D* CuProlongate3D::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuProlongate3D>(new CuProlongate3D); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

__global__ void cuProlongate3D_p2_add(DTYPE * dst, DTYPE * src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);

		int idx_x_src = idx_x >> 1;
		int idx_y_src = idx_y >> 1;
		int idx_z_src = idx_z >> 1;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * dim_src.x * dim_src.y;
		dst[idx] += src[idx_src];
	}
}

__global__ void cuProlongate3D_p2(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);

		int idx_x_src = idx_x >> 1;
		int idx_y_src = idx_y >> 1;
		int idx_z_src = idx_z >> 1;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * dim_src.x * dim_src.y;
		dst[idx] = src[idx_src];
	}
}

__global__ void cuProlongate3D_common(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x >> 1;
		int idx_y_src = idx_y >> 1;
		int idx_z_src = idx_z >> 1;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * dim_src.x * dim_src.y;
		dst[idx] = src[idx_src];
	}
}

__global__ void cuProlongate3D_common_add(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;

		int idx_x_src = idx_x >> 1;
		int idx_y_src = idx_y >> 1;
		int idx_z_src = idx_z >> 1;

		int idx_src = idx_y_src + idx_x_src * dim_src.y + idx_z_src * dim_src.x * dim_src.y;
		dst[idx] += src[idx_src];
	}
}

__global__ void cuProlongate3D_m1_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y + 1);
		int idx_y = idx - idx_xz * (dim_src.y + 1);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);
		DTYPE val_0 = idx_y > 0 ? 0.5 * src[idx_src - 1] : 0.f;
		//DTYPE val_1 = 0.f;
		if (idx_y < dim_src.y)
		{
			DTYPE val_1 = src[idx_src];
			val_0 += 0.5 * val_1;
			dst[idx_dst + 1] = val_1;
		}
		dst[idx_dst] = val_0;
	}
}

__global__ void cuProlongate3D_m1_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x + 1);
		int idx_x = idx_xz - (dim_src.x + 1) * idx_z;

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		//int idx_src = INDEX3D(idx_y_dst, idx_x_dst + 1, idx_z_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = idx_x > 0 ? 0.5 * src[idx_src - 2 * dim.y] : 0.f;
		DTYPE val_1 = 0.f;
		if (idx_x < dim_src.x)
		{
			val_1 = src[idx_src];
			val_0 += 0.5 * val_1;
			dst[idx_dst + dim.y] = val_1;
		}
		dst[idx_dst] = val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_m1_z_add(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2;
		int slice = dim.x * dim.y;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		//int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);
		int idx_src = idx_dst + slice;
		DTYPE val_0 = idx_z > 0 ? 0.5 * src[idx_src - 2 * slice] : 0.f;
		DTYPE val_1 = 0.f;
		if (idx_z < dim_src.z)
		{
			val_1 = src[idx_src];
			val_0 += 0.5 * val_1;
			dst[idx_dst + slice] += val_1;
		}
		dst[idx_dst] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_p2_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);
		DTYPE val_0 = src[idx_src];

		dst[idx_dst + 1] = val_0;
		dst[idx_dst] = val_0;
	}
}

__global__ void cuProlongate3D_p2_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = idx_dst + dim.y;
		DTYPE val_0 = src[idx_src];
		dst[idx_dst + dim.y] = val_0;
		dst[idx_dst] = val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_p2_z_add(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2;
		int slice = dim.x * dim.y;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = idx_dst + slice;
		DTYPE val_0 = src[idx_src];
		dst[idx_dst + slice] += val_0;
		dst[idx_dst] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_p2_c4_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2 + 1;
		int idx_y_dst = idx_y * 2;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);
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

__global__ void cuProlongate3D_p2_c4_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2 + 1;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
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
		dst[idx_dst + dim.y] = val_p1;
		dst[idx_dst] = val_m1;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_p2_c4_z_add(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x;
		int idx_y_dst = idx_y;
		int idx_z_dst = idx_z * 2;
		int slice = dim.x * dim.y;

		int idx_dst = INDEX3D(idx_y_dst, idx_x_dst, idx_z_dst, dim);
		int idx_src = idx_dst + slice;
		//DTYPE val_0 = src[idx_src];
		DTYPE val_m1 = 0.75f * src[idx_src];
		DTYPE val_p1 = val_m1;

		if (idx_z > 0)
		{
			val_m1 += 0.25f * src[idx_src - 2 * slice];
		}
		if (idx_z < dim_src.z - 1)
		{
			val_p1 += 0.25f * src[idx_src + 2 * slice];
		}
		dst[idx_dst + slice] += val_p1;
		dst[idx_dst] += val_m1;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

__global__ void cuProlongate3D_p1_y(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_y_dst = idx_y * 2;

		int idx_dst = INDEX3D(idx_y_dst, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);

		DTYPE val_0 = src[idx_src];

		dst[idx_dst] = val_0;

		if (idx_y_dst + 1 < dim.y)
			dst[idx_dst + 1] = val_0;
	}
}

__global__ void cuProlongate3D_p1_x(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_x_dst = idx_x * 2;

		int idx_dst = INDEX3D(idx_y, idx_x_dst, idx_z, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);

		DTYPE val_0 = src[idx_src];
		dst[idx_dst] = val_0;

		if (idx_x_dst + 1 < dim.x)
			dst[idx_dst + dim.y] = val_0;
	}
}

__global__ void cuProlongate3D_p1_z_add(DTYPE* dst, DTYPE* src, int3 dim, int3 dim_src, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / (dim_src.y);
		int idx_y = idx - idx_xz * (dim_src.y);
		int idx_z = idx_xz / (dim_src.x);
		int idx_x = idx_xz - (dim_src.x) * idx_z;

		int idx_z_dst = idx_z * 2;
		int slice = dim.x * dim.y;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z_dst, dim);
		int idx_src = INDEX3D(idx_y, idx_x, idx_z, dim_src);

		DTYPE val_0 = src[idx_src];
		dst[idx_dst] += val_0;

		if (idx_z_dst + 1 < dim.z)
			dst[idx_dst + slice] += val_0;
		//printf("idx: %d, %d, %d, idx_dst: %d, %d, %d, val: %f, %f\n", idx_x, idx_y, idx_z, idx_x_dst, idx_y_dst, idx_z_dst, val_0, val_1);
	}
}

void CuProlongate3D::SolveAdd(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int3 dim_dst = { dim.x << 1, dim.y << 1, dim.z << 1 };
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);

	cuProlongate3D_common_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuProlongate3D::SolveAddP1(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int3 dim_dst = { ((dim.x - 1) << 1) + 1, ((dim.y - 1) << 1) + 1, ((dim.z - 1) << 1) + 1 };
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);

	cuProlongate3D_common_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuProlongate3D::Solve(DTYPE* dst, DTYPE* src, int3 dim)
{
	int3 dim_dst = { dim.x << 1, dim.y << 1, dim.z << 1 };
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2 << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	cuProlongate3D_common << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
}

void CuProlongate3D::SolveAddM1(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int3 dim_dst = { ((dim.x + 1) << 1) - 1, ((dim.y + 1) << 1) - 1, ((dim.z + 1) << 1) - 1 };
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	int3 dim_thread = dim;
	int thread_size = dim_thread.x * (dim_thread.y + 1) * dim_thread.z;
	cuProlongate3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.y = dim_dst.y;
	thread_size = (dim_thread.x + 1) * dim_thread.y * dim_thread.z;
	cuProlongate3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_dst, dim_thread, thread_size);
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * (dim_thread.z + 1);
	cuProlongate3D_m1_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
}

void CuProlongate3D::SolveAddP2(DTYPE* dst, DTYPE* src, int3 dim, char bc)
{
	int3 dim_dst = { dim.x << 1, dim.y << 1, dim.z << 1 };
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	int3 dim_thread = dim;
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.y = dim_dst.y;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_dst, dim_thread, thread_size);
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p2_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
}

void CuProlongate3D::SolveAddRN(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc)
{
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	int3 dim_thread = dim;
	int thread_size;
	if ((dim_dst.y & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * (dim_thread.y + 1) * dim_thread.z;
		cuProlongate3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.y = dim_dst.y;
	if ((dim_dst.x & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = (dim_thread.x + 1) * dim_thread.y * dim_thread.z;
		cuProlongate3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, tmp, dim_dst, dim_thread, thread_size);
	}
	//CudaPrintfMat(tmp, dim_dst);

	dim_thread.x = dim_dst.x;
	if ((dim_dst.z & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * dim_thread.y * (dim_thread.z + 1);
		cuProlongate3D_m1_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp, dim_dst, dim_thread, thread_size);
	}
}

void CuProlongate3D::SolveAddRN_c4(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc)
{
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	//cuProlongate3D_p2_add << <BLOCKS(max_size), THREADS(max_size) >> > (dst, src, dim_dst, dim, max_size);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);
	int3 dim_thread = dim;
	int thread_size;
	if ((dim_dst.y & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_c4_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * (dim_thread.y + 1) * dim_thread.z;
		cuProlongate3D_m1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst, dim_thread, thread_size);
	}
	//CudaPrintfMat(tmp, dim_dst);
	cudaCheckError(cudaGetLastError());

	dim_thread.y = dim_dst.y;
	if ((dim_dst.x & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_c4_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = (dim_thread.x + 1) * dim_thread.y * dim_thread.z;
		cuProlongate3D_m1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_dst, dim_thread, thread_size);
	}
	//CudaPrintfMat(tmp, dim_dst);
	cudaCheckError(cudaGetLastError());

	dim_thread.x = dim_dst.x;
	if ((dim_dst.z & 1) == 0)
	{
		thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
		cuProlongate3D_p2_c4_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_dst, dim_thread, thread_size);
	}
	else
	{
		thread_size = dim_thread.x * dim_thread.y * (dim_thread.z + 1);
		cuProlongate3D_m1_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_dst, dim_thread, thread_size);
	}
	cudaCheckError(cudaGetLastError());
}

void CuProlongate3D::SolveAddP1RN(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc)
{
	int max_size = dim_dst.x * dim_dst.y * dim_dst.z;

	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("tmp", max_size);
	DTYPE* tmp_2 = CuMemoryManager::GetInstance()->GetData("tmp_2", max_size);

	int3 dim_thread = dim;
	int3 dim_dst_com = dim;
	dim_dst_com.y = dim_dst.y;
	int thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p1_y << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp, src, dim_dst_com, dim_thread, thread_size);

	dim_thread.y = dim_dst.y;
	dim_dst_com.x = dim_dst.x;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p1_x << <BLOCKS(thread_size), THREADS(thread_size) >> > (tmp_2, tmp, dim_dst_com, dim_thread, thread_size);

	dim_thread.x = dim_dst.x;
	dim_dst_com.z = dim_dst.z;
	thread_size = dim_thread.x * dim_thread.y * dim_thread.z;
	cuProlongate3D_p1_z_add << <BLOCKS(thread_size), THREADS(thread_size) >> > (dst, tmp_2, dim_dst_com, dim_thread, thread_size);
}

/***************************************************Local Select********************************************************/
__global__ void cuProlongate3D_p1_whole_add(DTYPE* dst, DTYPE* src, int* index, int3 dim, int3 dim_src, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / (dim.y);
		int idx_y = idx - idx_xz * (dim.y);
		int idx_z = idx_xz / (dim.x);
		int idx_x = idx_xz - (dim.x) * idx_z;

		int idx_x_src = idx_x >> 1;
		int idx_y_src = idx_y >> 1;
		int idx_z_src = idx_z >> 1;
		int slice = dim.x * dim.y;

		int idx_dst = INDEX3D(idx_y, idx_x, idx_z, dim);
		int idx_src = INDEX3D(idx_y_src, idx_x_src, idx_z_src, dim_src);

		dst[idx_dst] += src[idx_src];
	}
}

void CuProlongate3D::SolveAddP1RN(DTYPE* dst, DTYPE* src, int* index, int size_idx, int3 dim_dst, int3 dim, char bc)
{
	cuProlongate3D_p1_whole_add << <BLOCKS(size_idx), THREADS(size_idx) >> > (dst, src, index, dim_dst, dim, size_idx);
	cudaCheckError(cudaGetLastError());
}