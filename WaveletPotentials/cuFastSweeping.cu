#include "cuFastSweeping.cuh"
#include "cudaMath.cuh"


__device__ inline float atomicMin(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ __forceinline__ double atomicMin(double* address, double val)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val < __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long*)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}



CuFastSweeping::CuFastSweeping() = default;
CuFastSweeping::~CuFastSweeping() = default;

std::auto_ptr<CuFastSweeping> CuFastSweeping::instance_;

CuFastSweeping* CuFastSweeping::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuFastSweeping>(new CuFastSweeping); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

/*
 * Solves the Eikonal equation at each point of the grid.
 * Arguments:
 *   DTYPE - current distance value
 *   DTYPE - minimum distance in the x-direction
 *   DTYPE - minimum distance in the y-direction
 *   DTYPE - minimum distance in the z-direction
 *   DTYPE - spacing in the x-direction
 *   DTYPE - spacing in the y-direction
 *   DTYPE - spacing in the z-direction
 */
__device__ DTYPE solve_eikonal(DTYPE cur_dist, DTYPE minX, DTYPE minY, DTYPE minZ, DTYPE dx, DTYPE dy, DTYPE dz) {
	DTYPE dist_new = 0;
	DTYPE m[] = { minX, minY, minZ };
	DTYPE d[] = { dx, dy, dz };

	// sort the mins
	for (int i = 1; i < 3; i++) {
		for (int j = 0; j < 3 - i; j++) {
			if (m[j] > m[j + 1]) {
				DTYPE tmp_m = m[j];
				DTYPE tmp_d = d[j];
				m[j] = m[j + 1]; d[j] = d[j + 1];
				m[j + 1] = tmp_m; d[j + 1] = tmp_d;
			}
		}
	}

	// simplifying the variables
	DTYPE m_0 = m[0], m_1 = m[1], m_2 = m[2];
	DTYPE d_0 = d[0], d_1 = d[1], d_2 = d[2];
	DTYPE m2_0 = m_0 * m_0, m2_1 = m_1 * m_1, m2_2 = m_2 * m_2;
	DTYPE d2_0 = d_0 * d_0, d2_1 = d_1 * d_1, d2_2 = d_2 * d_2;

	dist_new = m_0 + d_0;
	if (dist_new > m_1) {

		DTYPE s = sqrt(-m2_0 + 2 * m_0 * m_1 - m2_1 + d2_0 + d2_1);
		dist_new = (m_1 * d2_0 + m_0 * d2_1 + d_0 * d_1 * s) / (d2_0 + d2_1);

		if (dist_new > m_2) {

			DTYPE a = sqrt(-m2_0 * d2_1 - m2_0 * d2_2 + 2 * m_0 * m_1 * d2_2
				- m2_1 * d2_0 - m2_1 * d2_2 + 2 * m_0 * m_2 * d2_1
				- m2_2 * d2_0 - m2_2 * d2_1 + 2 * m_1 * m_2 * d2_0
				+ d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);

			dist_new = (m_2 * d2_0 * d2_1 + m_1 * d2_0 * d2_2 + m_0 * d2_1 * d2_2 + d_0 * d_1 * d_2 * a) /
				(d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
		}
	}

	return min(cur_dist, dist_new);
}

__device__ DTYPE solve_eikonal_nomin(DTYPE cur_dist, DTYPE minX, DTYPE minY, DTYPE minZ, DTYPE dx, DTYPE dy, DTYPE dz) {
	DTYPE dist_new = 0;
	DTYPE m[] = { minX, minY, minZ };
	DTYPE d[] = { dx, dy, dz };

	// sort the mins
	for (int i = 1; i < 3; i++) {
		for (int j = 0; j < 3 - i; j++) {
			if (m[j] > m[j + 1]) {
				DTYPE tmp_m = m[j];
				DTYPE tmp_d = d[j];
				m[j] = m[j + 1]; d[j] = d[j + 1];
				m[j + 1] = tmp_m; d[j + 1] = tmp_d;
			}
		}
	}

	// simplifying the variables
	DTYPE m_0 = m[0], m_1 = m[1], m_2 = m[2];
	DTYPE d_0 = d[0], d_1 = d[1], d_2 = d[2];
	DTYPE m2_0 = m_0 * m_0, m2_1 = m_1 * m_1, m2_2 = m_2 * m_2;
	DTYPE d2_0 = d_0 * d_0, d2_1 = d_1 * d_1, d2_2 = d_2 * d_2;

	dist_new = m_0 + d_0;
	if (dist_new > m_1) {

		DTYPE s = sqrt(-m2_0 + 2 * m_0 * m_1 - m2_1 + d2_0 + d2_1);
		dist_new = (m_1 * d2_0 + m_0 * d2_1 + d_0 * d_1 * s) / (d2_0 + d2_1);

		if (dist_new > m_2) {

			DTYPE a = sqrt(-m2_0 * d2_1 - m2_0 * d2_2 + 2 * m_0 * m_1 * d2_2
				- m2_1 * d2_0 - m2_1 * d2_2 + 2 * m_0 * m_2 * d2_1
				- m2_2 * d2_0 - m2_2 * d2_1 + 2 * m_1 * m_2 * d2_0
				+ d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);

			dist_new = (m_2 * d2_0 * d2_1 + m_1 * d2_0 * d2_2 + m_0 * d2_1 * d2_2 + d_0 * d_1 * d_2 * a) /
				(d2_0 * d2_1 + d2_0 * d2_2 + d2_1 * d2_2);
		}
	}

	return dist_new;
}

/*
 * Kernel for fast sweeping method
 * Arguments:
 *   cudaPitchedPtr [in/out] - pointer to distance array in
 *                             device memory
 *        SweepInfo [in]     - sweep information
 */
__global__ void fast_sweep_kernel(DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE center = ls[idx];                                                     // center distance
		DTYPE left = ls[idx - dim.y];                                                     // left distance
		DTYPE right = ls[idx + dim.y];                                                    // right distance
		DTYPE up = ls[idx + 1];    // upper distance
		DTYPE down = ls[idx - 1];   // lower distance
		DTYPE front = ls[idx - slice_xy]; // front distance
		DTYPE back = ls[idx + slice_xy];  // back distance

		DTYPE minX = min(left, right);
		DTYPE minY = min(up, down);
		DTYPE minZ = min(front, back);
		ls[idx] = solve_eikonal(center, minX, minY, minZ, dx, dx, dx);
	}
}

/*
 * Kernel for fast sweeping method
 * Arguments:
 *   cudaPitchedPtr [in/out] - pointer to distance array in
 *                             device memory
 *        SweepInfo [in]     - sweep information
 */
__global__ void fast_sweep_kernel_neg(DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size && ls[idx] < 0)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE center = ls[idx];                                                     // center distance
		DTYPE left = ls[idx - dim.y];                                                     // left distance
		DTYPE right = ls[idx + dim.y];                                                    // right distance
		DTYPE up = ls[idx + 1];    // upper distance
		DTYPE down = ls[idx - 1];   // lower distance
		DTYPE front = ls[idx - slice_xy]; // front distance
		DTYPE back = ls[idx + slice_xy];  // back distance

		DTYPE minX = min(left, right);
		DTYPE minY = min(up, down);
		DTYPE minZ = min(front, back);
		ls[idx] = solve_eikonal(center, minX, minY, minZ, dx, dx, dx);
	}
}

/*
 * Kernel for fast sweeping method
 * Arguments:
 *   cudaPitchedPtr [in/out] - pointer to distance array in
 *                             device memory
 *        SweepInfo [in]     - sweep information
 */
__global__ void fast_sweep_kernel_pos(DTYPE* ls, int3 dim, DTYPE dx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size && ls[idx] > -eps<DTYPE>)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * (dim.y);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - idx_z * (dim.x);
		int slice_xy = dim.x * dim.y;

		//DTYPE center = ls[idx];                                                     // center distance
		//DTYPE left = idx_x > 0 ? ls[idx - dim.y] : 0.f;                                                     // left distance
		//DTYPE right = idx_x < dim.x - 1 ? ls[idx + dim.y] : 0.f;                                                    // right distance
		//DTYPE up = idx_y < dim.y - 1 ? ls[idx + 1] : 0.f;    // upper distance
		//DTYPE down = idx_y > 0 ? ls[idx - 1] : 0.f;   // lower distance
		//DTYPE front = idx_z > 0 ? ls[idx - slice_xy] : 0.f; // front distance
		//DTYPE back = idx_z < dim.z - 1 ? ls[idx + slice_xy] : 0.f;  // back distance

		DTYPE max_dx = 100000000.f;
		DTYPE center = ls[idx];                                                     // center distance
		DTYPE left = idx_x > 0 ? ls[idx - dim.y] : max_dx;                                                     // left distance
		DTYPE right = idx_x < dim.x - 1 ? ls[idx + dim.y] : max_dx;                                                    // right distance
		DTYPE up = idx_y < dim.y - 1 ? ls[idx + 1] : max_dx;    // upper distance
		DTYPE down = idx_y > 0 ? ls[idx - 1] : max_dx;   // lower distance
		DTYPE front = idx_z > 0 ? ls[idx - slice_xy] : max_dx; // front distance
		DTYPE back = idx_z < dim.z - 1 ? ls[idx + slice_xy] : max_dx;  // back distance

		DTYPE minX = min(left, right);
		DTYPE minY = min(up, down);
		DTYPE minZ = min(front, back);
		ls[idx] = solve_eikonal_nomin(center, minX, minY, minZ, dx, dx, dx);
	}
}

void CuFastSweeping::Solve(DTYPE* ls, int3 dim, DTYPE dx, int n_iters)
{
	int size = getsize(dim);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
}

__global__ void cuParticlesToRoughLevelSet(DTYPE* ls, DTYPE3* pars, int3 dim, int n_pars, DTYPE dx, DTYPE radius)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n_pars)
	{
		DTYPE inv_dx = 1.f / dx;

		DTYPE x = pars[idx].x * inv_dx - 0.5f;
		DTYPE y = pars[idx].y * inv_dx - 0.5f;
		DTYPE z = pars[idx].z * inv_dx - 0.5f;

		int ix = floor(x);
		int iy = floor(y);
		int iz = floor(z);

		DTYPE fx = x - ix;
		DTYPE fy = y - iy;
		DTYPE fz = z - iz;

		DTYPE wx[2] = { fx * fx, (1.f - fx) * (1.f - fx) };
		DTYPE wy[2] = { fy * fy, (1.f - fy) * (1.f - fy) };
		DTYPE wz[2] = { fz * fz, (1.f - fz) * (1.f - fz) };

#pragma unroll
		for (int k = iz; k <= iz + 1; k++)
		{
#pragma unroll
			for (int j = ix; j <= ix + 1; j++)
			{
#pragma unroll
				for (int i = iy; i <= iy + 1; i++)
				{
					DTYPE dis = sqrt(wx[j - ix] + wy[i - iy] + wz[k - iz]) * dx;
					//DTYPE dis = 1.f;
					DTYPE rough_ls = dis - radius;
					int idx_ls = INDEX3D(i, j, k, dim);
					atomicMin(&(ls[idx_ls]), rough_ls);
				}
			}
		}
	}
}

__global__ void cuInvLevelSet(DTYPE* ls, DTYPE max_dis, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		DTYPE res = ls[idx];
		if (res > 0.f)
		{
			res *= -1.f;
		}
		else
		{
			res = max_dis;
		}
		ls[idx] = res;
	}
}

void CuFastSweeping::ParticlesToLs(DTYPE* ls, DTYPE3* x, int3 dim, int n_pars, DTYPE dx, DTYPE radius, DTYPE max_dis, int n_iters)
{
	int size = getsize(dim);
	CudaSetValue(ls, size, dx);
	cuParticlesToRoughLevelSet << <BLOCKS(n_pars), THREADS(n_pars) >> > (ls, x, dim, n_pars, dx, radius);
	//CudaPrintfMat(ls, dim);
	//CudaMatScale(ls, -1.f, size);
	cuInvLevelSet << <BLOCKS(size), THREADS(size) >> > (ls, max_dis, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
	//CudaPrintfMat(ls, dim);
	//CudaMatScale(ls, -1.f, size);
	cuInvLevelSet << <BLOCKS(size), THREADS(size) >> > (ls, max_dis, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
	CudaMatScale(ls, -1.f, size);
}

__global__ void cuInvLevelSet_nb(DTYPE* ls, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		DTYPE res = ls[idx];
		ls[idx] = -res;
	}
}

void CuFastSweeping::ParticlesToLsNarrow(DTYPE* ls, DTYPE3* x, int3 dim, int n_pars, DTYPE dx, DTYPE radius, DTYPE max_dis, int n_iters)
{
	int size = getsize(dim);
	//if (max_dis >= 19 && max_dis < 20)
	//	CudaPrintfMat(ls, dim);
	cuParticlesToRoughLevelSet << <BLOCKS(n_pars), THREADS(n_pars) >> > (ls, x, dim, n_pars, dx, radius);
	//if (max_dis >= 19 && max_dis < 20)
	//	CudaPrintfMat(ls, dim);
	//CudaPrintfMat(ls, dim);
	//CudaMatScale(ls, -1.f, size);
	//cuInvLevelSet << <BLOCKS(size), THREADS(size) >> > (ls, max_dis, size);
	//cuInvLevelSet_nb << <BLOCKS(size), THREADS(size) >> > (ls, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
	//if (max_dis >= 19 && max_dis < 20)
	//	CudaPrintfMat(ls, dim);
	//CudaPrintfMat(ls, dim);
	//CudaMatScale(ls, -1.f, size);
	//cuInvLevelSet << <BLOCKS(size), THREADS(size) >> > (ls, max_dis, size);
	cuInvLevelSet_nb << <BLOCKS(size), THREADS(size) >> > (ls, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
	//CudaMatScale(ls, -1.f, size);
	//if (max_dis >= 19 && max_dis < 20)
	//	CudaPrintfMat(ls, dim);
}

void CuFastSweeping::ReinitLevelSet(DTYPE* ls, int3 dim, DTYPE dx, int n_iters)
{
	int size = getsize(dim);
	CudaMatScale(ls, -1.f, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
	//CudaPrintfMat(ls, dim);
	CudaMatScale(ls, -1.f, size);
	for (int i = 0; i < n_iters; i++)
	{
		fast_sweep_kernel_pos << <BLOCKS(size), THREADS(size) >> > (ls, dim, dx, size);
	}
}