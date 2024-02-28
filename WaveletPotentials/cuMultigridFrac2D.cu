#include "cuMultigridFrac2D.cuh"
#include "cuMemoryManager.cuh"

__global__ void cuRestrictDx(DTYPE* dst_dx, DTYPE* src_dx, int dim_dst, int dim)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < dim_dst)
	{
		int idx_src = idx * 2;
		dst_dx[idx] = src_dx[idx_src];
		idx_src += 1;
		if (idx_src < dim)
		{
			dst_dx[idx] += src_dx[idx_src];
		}
	}
}

__global__ void cuInitDx(DTYPE* dst_dx, DTYPE dx, int dim)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < dim)
	{
		dst_dx[idx] = dx;
	}
}

CuMultigridFrac2D::CuMultigridFrac2D(DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx, char bc) : dim_(dim), max_level_(0), dx_(dx), bc_(bc)
{
	dx_e_ = bc == 'd' ? make_DTYPE2(0.f, 0.f) : dx;
	bc_ = bc == 'd' ? 'z' : bc;
	int min_dim = std::min(dim.x, dim.y);
	dx_inv_ = make_DTYPE2(1.f / dx.x, 1.f / dx.y);
	dim_rs_from_dim_ = [](int2 dim_in) -> int2 {
		int2 dim_out;
		if ((dim_in.x & 1) == 0)
		{
			dim_out.x = dim_in.x >> 1;
		}
		else
		{
			dim_out.x = (dim_in.x >> 1) + 1;
		}
		if ((dim_in.y & 1) == 0)
		{
			dim_out.y = dim_in.y >> 1;
		}
		else
		{
			dim_out.y = (dim_in.y >> 1) + 1;
		}
		return dim_out;
	};

	int2 dim_rs = dim;
	int2 last_rs = dim;
	while (min_dim > 1)
	{
		sl_n_ = std::max(dim_rs.x, dim_rs.y);
		int size_coarse = dim_rs.x * dim_rs.y;

		// create coarse grids for residual
		DTYPE* dev_f;
		checkCudaErrors(cudaMalloc((void**)&dev_f, size_coarse * sizeof(DTYPE)));
		checkCudaErrors(cudaMemset(dev_f, 0, sizeof(DTYPE) * size_coarse));
		f_coarses_.push_back(dev_f);

		// create coarse grids for result
		DTYPE* dev_v;
		checkCudaErrors(cudaMalloc((void**)&dev_v, size_coarse * sizeof(DTYPE)));
		checkCudaErrors(cudaMemset(dev_v, 0, sizeof(DTYPE) * size_coarse));
		v_coarses_.push_back(dev_v);

		// create coarse grids for ax & create coarse grids for ax
		DTYPE* dev_ax;
		int2 dim_ax_rs = { last_rs.x + 1, last_rs.y };
		int size_ax = dim_ax_rs.x * dim_ax_rs.y;
		checkCudaErrors(cudaMalloc((void**)&dev_ax, size_ax * sizeof(DTYPE)));

		DTYPE* dev_ay;
		int2 dim_ay_rs = { last_rs.x, last_rs.y + 1 };
		int size_ay = dim_ay_rs.y * dim_ay_rs.x;
		checkCudaErrors(cudaMalloc((void**)&dev_ay, size_ay * sizeof(DTYPE)));

		if (max_level_ == 0)
		{
			checkCudaErrors(cudaMemcpy(dev_ax, Ax, size_ax * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(dev_ay, Ay, size_ay * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
		}
		else
		{
			CuRestriction2D::GetInstance()->SolveP1Ax(dev_ax, ax_coarses_[max_level_ - 1], dim_ax_rs, bc);
			CuRestriction2D::GetInstance()->SolveP1Ay(dev_ay, ay_coarses_[max_level_ - 1], dim_ay_rs, bc);
		}
		ax_coarses_.push_back(dev_ax);
		ay_coarses_.push_back(dev_ay);

		// create coarse grids for ax & create coarse grids for ax
		DTYPE* dev_dx;
		checkCudaErrors(cudaMalloc((void**)&dev_dx, dim_rs.x * sizeof(DTYPE)));
		DTYPE* dev_dy;
		checkCudaErrors(cudaMalloc((void**)&dev_dy, dim_rs.y * sizeof(DTYPE)));
		if (max_level_ == 0)
		{
			cuInitDx << <BLOCKS(dim.x), THREADS(dim.x) >> > (dev_dx, dx.x, dim.x);
			cuInitDx << <BLOCKS(dim.y), THREADS(dim.y) >> > (dev_dy, dx.y, dim.y);
		}
		else
		{
			cuRestrictDx << <BLOCKS(dim_rs.x), THREADS(dim_rs.x) >> > (dev_dx, dx_coarses_[max_level_ - 1], dim_rs.x, last_rs.x);
			cuRestrictDx << <BLOCKS(dim_rs.y), THREADS(dim_rs.y) >> > (dev_dy, dy_coarses_[max_level_ - 1], dim_rs.y, last_rs.y);
		}
		dx_coarses_.push_back(dev_dx);
		dy_coarses_.push_back(dev_dy);

		last_rs = dim_rs;
		dim_rs = dim_rs_from_dim_(dim_rs);
		min_dim = std::min(dim_rs.x, dim_rs.y);
		++max_level_;
	}
}

CuMultigridFrac2D::~CuMultigridFrac2D()
{
	for (DTYPE* ptr : f_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : v_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : ax_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : ay_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : dx_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : dy_coarses_)
	{
		cudaFree(ptr);
	}
}

void CuMultigridFrac2D::Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	if (pre_v != nullptr)
	{
		cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * dim_.x * dim_.y, cudaMemcpyDeviceToDevice);
	}
	else
	{
		cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * dim_.x * dim_.y);
	}
	cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * dim_.x * dim_.y, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycle(dim_, nu1, nu2, weight, bc_, 0);
	cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * dim_.x * dim_.y, cudaMemcpyDeviceToDevice);
}

void CuMultigridFrac2D::SolveOneVCycle(int2 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{	
	if (deep_level < max_level_ - 1)
	{
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], 
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], nu1, weight, bc);
		CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
			ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dx_e_, nu1, weight, bc);
		

		// do restriction 
		int2 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg2d", dim.x * dim.y);

		// get residual
		//CuWeightedJacobi::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc);
		CuWeightedJacobi::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level],
			ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dx_e_, bc);

		//CuRestriction2D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		CuRestriction2D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y));
		SolveOneVCycle(dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		CuProlongation2D::GetInstance()->SolveP1RNAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], nu2, weight, bc);
		CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
			ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dx_e_, nu2, weight, bc);
	}
	else
	{
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
	 	CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
			ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dx_e_, 10, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int2(dim.x + 1, dim.y));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int2(dim.x, dim.y + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

void CuMultigridFrac2D::RestrictDx1D(DTYPE* dst_dx, DTYPE* src_dx, int dim_dst, int dim)
{
	cuRestrictDx << <BLOCKS(dim_dst), THREADS(dim_dst) >> > (dst_dx, src_dx, dim_dst, dim);
}

void CuMultigridFrac2D::InitDx1D(DTYPE* dst_dx, DTYPE dx, int dim)
{
	cuInitDx << <BLOCKS(dim), THREADS(dim) >> > (dst_dx, dx, dim);
}