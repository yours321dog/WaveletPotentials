#include "cuWeightedJacobi.cuh"
#include "cuMemoryManager.cuh"
#include "cudaMath.cuh"

__global__ void cuWeightedJacobiOneIterDirichlet(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
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

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
	}
}

__global__ void cuWeightedJacobiOneIterNeumann(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
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

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx]);
		//		if (max_size < 9)
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobiOneIterDirichlet00(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx]);
	}
}

__global__ void cuWeightedJacobiDirichlet_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, DTYPE level_mc, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		DTYPE level_mc_inv_2 = 1 / (level_mc) * 2;

		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += level_mc * dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += (level_mc_inv_2 - 1) * dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += level_mc * dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += (level_mc_inv_2 - 1) * dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
	}
}

__global__ void cuWeightedJacobiNeumann_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, DTYPE level_mc, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= level_mc * dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= level_mc * dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobiDirichlet00_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, DTYPE level_mc_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * dxp2_inv.y;
		else
			a_diag += (level_mc_inv - 1) * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * dxp2_inv.x;
		else
			a_diag += (level_mc_inv - 1) * dxp2_inv.x;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//if (max_size < 5)
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi_frac_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}
			

		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWeightedJacobi_frac_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, 
	DTYPE* dx, DTYPE* dy, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWeightedJacobi_frac_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim,
	DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;
		
		// bottom
		DTYPE dxy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dxy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
			off_d += -cf * src[idx - 1];


		// top
		dxy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dxy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
			off_d += -cf * src[idx + 1];

		// left
		dxy_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dxy_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dxy_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dxy_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}


CuWeightedJacobi::CuWeightedJacobi() = default;
CuWeightedJacobi::~CuWeightedJacobi() = default;

std::auto_ptr<CuWeightedJacobi> CuWeightedJacobi::instance_;

CuWeightedJacobi* CuWeightedJacobi::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuWeightedJacobi>(new CuWeightedJacobi); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void CuWeightedJacobi::Solve(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dx, int n_iters, DTYPE weight, char bc, WeightedJacobiType wj_type, int level)
{
	int max_size = dim.x * dim.y;
	DTYPE2 dx_inv = make_DTYPE2(1.f / dx.x, 1.f / dx.y);
	DTYPE2 dxp2_inv = make_DTYPE2(dx_inv.x * dx_inv.x, dx_inv.y * dx_inv.y);

	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("lwt", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		//printf("solve wj-------------------\n");
		switch (wj_type)
		{
		//case WeightedJacobiType::WJ_2NM1:

		//	break;
		case WeightedJacobiType::WJ_2NP1:
			break;
		case WeightedJacobiType::WJ_2N:
		{
			DTYPE level_mc_inv = std::exp2(level);
			DTYPE level_mc = 1 / level_mc_inv;
			switch (bc)
			{
			case 'd':
				cuWeightedJacobiDirichlet_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, level_mc, max_size);
				break;
			case 'z':
				cuWeightedJacobiDirichlet00_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, level_mc_inv, max_size);
				break;
			default:
				//cuWeightedJacobiNeumann_p2 << <BLOCKS(max_size), NUM_THREADS >> > (result, sss, f, dim, dxp2_inv, weight, level_mc, max_size);
				cuWeightedJacobiOneIterNeumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			}
		}
			break;
		default:
			switch (bc)
			{
			case 'd':
				cuWeightedJacobiOneIterDirichlet << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
				break;
			case 'z':
				cuWeightedJacobiOneIterDirichlet00 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
				break;
			default:
				cuWeightedJacobiOneIterNeumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			}
			break;
		}
		SwapPointer((void**)&result, (void**)&sss);
		//cudaDeviceSynchronize();
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx,
	int n_iters, DTYPE weight, char bc, WeightedJacobiType wj_type, int level)
{
	int max_size = dim.x * dim.y;
	DTYPE2 dx_inv = make_DTYPE2(1.f / dx.x, 1.f / dx.y);
	DTYPE2 dxp2_inv = make_DTYPE2(dx_inv.x * dx_inv.x, dx_inv.y * dx_inv.y);

	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("lwt", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
			//case 'd':
			//	cuWeightedJacobiDirichlet_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, level_mc, max_size);
			//	break;
			//case 'z':
			//	cuWeightedJacobiDirichlet00_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, dim, dxp2_inv, weight, level_mc_inv, max_size);
			//	break;
		default:
			cuWeightedJacobi_frac_n << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, Ax, Ay, dim, dxp2_inv, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy,
	int n_iters, DTYPE weight, char bc, WeightedJacobiType wj_type, int level)
{
	int max_size = dim.x * dim.y;

	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("lwt", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		default:
			cuWeightedJacobi_frac_n << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, Ax, Ay, dim, dx, dy, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, 
	DTYPE2 dx_e, int n_iters, DTYPE weight,	char bc)
{
	int max_size = dim.x * dim.y;

	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("lwt", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'z':
			cuWeightedJacobi_frac_z << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, Ax, Ay, dim, dx, dy, dx_e, weight, max_size);
			break;
		default:
			cuWeightedJacobi_frac_n << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (result, sss, f, Ax, Ay, dim, dx, dy, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

__global__ void cuWjGetFracRes_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dxp2_inv, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			//val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
	}
}

__global__ void cuWjGetFracRes_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			//val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
	}
}

__global__ void cuWjGetFracRes_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1);

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE dxy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dxy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
			off_d += -cf * src[idx - 1];


		// top
		dxy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dxy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
			off_d += -cf * src[idx + 1];

		// left
		dxy_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dxy_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dxy_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dxy_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			//val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
	}
}

DTYPE CuWeightedJacobi::GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx, char bc)
{
	int max_size = dim.x * dim.y;
	DTYPE2 dxp2_inv = make_DTYPE2(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y));

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		//cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	case 'z':
		//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		//cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	default:
		//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuWjGetFracRes_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, dim, dxp2_inv, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}

DTYPE CuWeightedJacobi::GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc)
{
	int max_size = dim.x * dim.y;

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, dim, dx, dy, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}

double CuWeightedJacobi::GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc)
{
	int max_size = dim.x * dim.y;

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		cuWjGetFracRes_z << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, dim, dx, dy, make_DTYPE2(0.f), max_size);
		break;
	case 'z':
		cuWjGetFracRes_z << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, dim, dx, dy, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, dim, dx, dy, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	double sum_rhs = CudaArrayNormSum(dev_res, max_size);
	double sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}

void CuWeightedJacobi::GetFracRhs(DTYPE* res, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc)
{
	int max_size = dim.x * dim.y;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_n << <BLOCKS(max_size), NUM_THREADS >> > (res, v, f, Ax, Ay, dim, dx, dy, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi::GetFracRhs(DTYPE* res, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc)
{
	int max_size = dim.x * dim.y;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		cuWjGetFracRes_z << <BLOCKS(max_size), NUM_THREADS >> > (res, v, f, Ax, Ay, dim, dx, dy, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_n << <BLOCKS(max_size), NUM_THREADS >> > (res, v, f, Ax, Ay, dim, dx, dy, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}