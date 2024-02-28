#include "cuMultigrid2D.cuh"

#include "cuMemoryManager.cuh"
#include <vector>

#include "cudaMath.cuh"

constexpr unsigned int switch_pair(MultigridType a, char f) {
	return (static_cast<unsigned int>(a) << 16) + f;
}

CuMultigrid2D::CuMultigrid2D(int2 dim, DTYPE2 dx, char bc, MultigridType mg_type) : dim_(dim), mg_type_(mg_type), max_level_(0), dx_(dx), bc_(bc)
{
	int min_dim = std::min(dim.x, dim.y);
	dx_inv_ = make_DTYPE2(1.f / dx.x, 1.f / dx.y);
	int end_dim = 0;
	switch (mg_type)
	{
	case MultigridType::MG_2NP1:
		break;
	case MultigridType::MG_2N:
		dim_rs_from_dim_ = [](int2 dim_in) -> int2 { return make_int2(dim_in.x / 2, dim_in.y / 2); };
		rs_type_ = RestrictionType::RT_2N;
		pg_type_ = ProlongationType::PT_2N;
		wj_type_ = WeightedJacobiType::WJ_2N;
		break;
	case MultigridType::MG_2N_C4:
		dim_rs_from_dim_ = [](int2 dim_in) -> int2 { return make_int2(dim_in.x / 2, dim_in.y / 2); };
		rs_type_ = RestrictionType::RT_2N_C4;
		pg_type_ = ProlongationType::PT_2N_C4;
		wj_type_ = WeightedJacobiType::WJ_2N;
		break;
	case MultigridType::MG_OTHER:
		break;
	case MultigridType::MG_RN:
		dim_rs_from_dim_ = [](int2 dim_in) -> int2 {
			int2 dim_out;
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
			return dim_out;
		};
		rs_type_ = RestrictionType::RT_2N_C4;
		pg_type_ = ProlongationType::PT_2N_C4;
		wj_type_ = WeightedJacobiType::WJ_2N;
		break;
	case MultigridType::MG_2NM1_RN:
		dim_rs_from_dim_ = [](int2 dim_in) -> int2 {
			int2 dim_out;
			dim_out.x = (dim_in.x >> 1) + 1;
			dim_out.y = (dim_in.y >> 1) + 1;
			return dim_out;
		};
		rs_type_ = RestrictionType::RT_2NM1_RN;
		pg_type_ = ProlongationType::PT_2NM1_RN;
		wj_type_ = WeightedJacobiType::WJ_2N;
		end_dim = 2;
		break;
	case MultigridType::MG_2NM1:
	default:
		dim_rs_from_dim_ = [](int2 dim_in) -> int2 { return make_int2((dim_in.x + 1) / 2 - 1, (dim_in.y + 1) / 2 - 1); };
		rs_type_ = RestrictionType::RT_2NM1;
		pg_type_ = ProlongationType::PT_2NM1;
		wj_type_ = WeightedJacobiType::WJ_2NM1;
		break;
	}

	int2 dim_rs = dim;
	while (min_dim > end_dim)
	{
		sl_n_ = std::max(dim_rs.x, dim_rs.y);
		int size_coarse = dim_rs.x * dim_rs.y;

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
		min_dim = std::min(dim_rs.x, dim_rs.y);
		++max_level_;
	}

	// prepare for the smallest levels linear algebra solver
	int a_sl_size = sl_n_ * sl_n_;
	std::vector<DTYPE> host_a_sl(a_sl_size, 0);
	DTYPE2 dx_sl = make_DTYPE2(dx.x * std::exp2(max_level_ - 1), dx.y * std::exp2(max_level_ - 1));
	DTYPE2 dxp2_inv_sl = make_DTYPE2(1.f / (dx_sl.x * dx_sl.x), 1.f / (dx_sl.y * dx_sl.y));

	int2 dim_sl = dim.x > dim.y ? make_int2(sl_n_, 1) : make_int2(1, sl_n_);

	DTYPE a_diag_mid = 2 * dxp2_inv_sl.x + 2 * dxp2_inv_sl.y;
	int idx_a_row = 0;

	switch (bc)
	{
	case 'd':
		for (int j = 0; j < dim_sl.x; j++)
		{
			for (int i = 0; i < dim_sl.y; i++)
			{
				int idx_a = idx_a_row + idx_a_row * sl_n_;

				a_diag_mid = 2 * dxp2_inv_sl.x + 2 * dxp2_inv_sl.y;


				if (i == 0)
					a_diag_mid += dxp2_inv_sl.y;
				else
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.y;

				if (i == dim_sl.y - 1)
					a_diag_mid += dxp2_inv_sl.y;
				else
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.y;

				if (j == 0)
					a_diag_mid += dxp2_inv_sl.x;
				else
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.x;

				if (j == dim_sl.x - 1)
					a_diag_mid += dxp2_inv_sl.x;
				else
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.x;

				host_a_sl[idx_a] = a_diag_mid;
				++idx_a_row;
			}
		}
		break;
	case 'z':
		for (int j = 0; j < dim_sl.x; j++)
		{
			for (int i = 0; i < dim_sl.y; i++)
			{
				int idx_a = idx_a_row + idx_a_row * sl_n_;

				a_diag_mid = 2 * dxp2_inv_sl.x + 2 * dxp2_inv_sl.y;


				if (i > 0)
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.y;
				if (i < dim_sl.y - 1)
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.y;

				if (j > 0)
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.x;
				if (j < dim_sl.x - 1)
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.x;

				host_a_sl[idx_a] = a_diag_mid;
				++idx_a_row;
			}
		}
		break;
	default:
		for (int j = 0; j < dim_sl.x; j++)
		{
			for (int i = 0; i < dim_sl.y; i++)
			{
				int idx_a = idx_a_row + idx_a_row * sl_n_;

				a_diag_mid = 2 * dxp2_inv_sl.x + 2 * dxp2_inv_sl.y;


				if (i == 0)
					a_diag_mid -= dxp2_inv_sl.y;
				else
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.y;

				if (i == dim_sl.y - 1)
					a_diag_mid -= dxp2_inv_sl.y;
				else
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.y;

				if (j == 0)
					a_diag_mid -= dxp2_inv_sl.x;
				else
					host_a_sl[idx_a - 1] = -dxp2_inv_sl.x;

				if (j == dim_sl.x - 1)
					a_diag_mid -= dxp2_inv_sl.x;
				else
					host_a_sl[idx_a + 1] = -dxp2_inv_sl.x;

				host_a_sl[idx_a] = a_diag_mid;
				++idx_a_row;
			}
		}
	}

	if (bc != 'n')
	{
		//checkCudaErrors(cudaMalloc((void**)&dev_A_, sizeof(DTYPE) * a_sl_size));
		//checkCudaErrors(cudaMalloc((void**)&d_pivot_, sl_n_ * sizeof(int)));
		//checkCudaErrors(cudaMalloc((void**)&d_info_, sizeof(int)));

		//checkCudaErrors(cudaMemcpy(dev_A_, host_a_sl.data(), a_sl_size * sizeof(DTYPE),
		//	cudaMemcpyHostToDevice));

		//checkCudaErrors(cusolverDnCreate(&handle_));

		//int Lwork;
		//checkCudaErrors(buffer_size_func(handle_, sl_n_, sl_n_, dev_A_, sl_n_, &Lwork));
		//DTYPE* d_Work;
		//checkCudaErrors(cudaMalloc((void**)&d_Work, Lwork * sizeof(DTYPE)));
		//checkCudaErrors(getrf_func(handle_, sl_n_, sl_n_, dev_A_, sl_n_, d_Work, d_pivot_, d_info_));
		//int aa;
		//checkCudaErrors(cudaMemcpy(&aa, d_info_, sizeof(int), cudaMemcpyDeviceToHost));
		//printf("d_info: %d\n", aa);
	}
}

CuMultigrid2D::~CuMultigrid2D()
{
	for (DTYPE* ptr : f_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : v_coarses_)
	{
		cudaFree(ptr);
	}
	//if (bc_ != 'n')
	//{
	//	cudaFree(dev_A_);
	//	cudaFree(d_pivot_);
	//	cudaFree(d_info_);
	//	checkCudaErrors(cusolverDnDestroy(handle_));
	//}
}

__global__ void cuGetResidualDirichlet(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, int max_size)
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
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += dxp2_inv.y;
		else
			off_d += -v[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += dxp2_inv.x;
		else
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += dxp2_inv.x;
		else
			off_d += -v[idx + dim.y] * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
	}
}

__global__ void cuGetResidualNeumann(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, int max_size)
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
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -v[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= dxp2_inv.x;
		else
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -v[idx + dim.y] * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
	}
}

__global__ void cuGetResidualDirichlet00(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, int max_size)
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
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -v[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -v[idx + dim.y] * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
	}
}

__global__ void cuGetResidualDirichlet_p2(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE level_mc, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_x = idx / dim.y;
		int idx_y = idx - idx_x * dim.y;

		DTYPE level_mc_inv_2 = 1 / (level_mc) * 2;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += level_mc * dxp2_inv.y;
		else
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += (level_mc_inv_2 - 1) * dxp2_inv.y;
		else
			off_d += -v[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += level_mc * dxp2_inv.x;
		else
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += (level_mc_inv_2 - 1) * dxp2_inv.x;
		else
			off_d += -v[idx + dim.y] * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
	}
}

__global__ void cuGetResidualNeumann_p2(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE level_mc, int max_size)
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
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -v[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= level_mc * dxp2_inv.x;
		else
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -v[idx + dim.y] * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], v[idx], f[idx], a_diag);
	}
}

__global__ void cuGetResidualDirichlet00_p2(DTYPE* dst, DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dxp2_inv, DTYPE level_mc_inv, int max_size)
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
			off_d += -v[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -v[idx + 1] * dxp2_inv.y;
		else
			a_diag += (level_mc_inv - 1) * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -v[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -v[idx + dim.y] * dxp2_inv.x;
		else
			a_diag += (level_mc_inv - 1) * dxp2_inv.x;

		dst[idx] = f[idx] - (a_diag * v[idx] + off_d);
	}
}

__global__ void cuRecoverPNuemann(DTYPE* dst, DTYPE* src, DTYPE dx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		dst[idx] = src[idx];
		//printf("f_val: %f\n", dst[idx]);
		__syncthreads();
		if (idx == 0)
		{
			dst[0] = dst[idx] * dx;
			for (int i = 1; i < max_size; i++)
			{
				dst[i] = dst[i - 1] + dst[i] * dx;
			}
			for (int i = max_size - 2; i >= 0; i--)
			{
				dst[i] = dst[i] * dx + dst[i + 1];
			}
		}
	}
}

void CuMultigrid2D::SolveOneVCycle(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE2 dxp2_inv = make_DTYPE2(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y));

	// do weightedjacobi solver

	if (deep_level < max_level_ - 3)
	{
		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc, wj_type_, deep_level);
		// do restriction 
		int2 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg2d", dim.x * dim.y);
		int max_size = dim.x * dim.y;
		//switch (bc)
		//{
		//case 'd':
		//	cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//case 'z':
		//	cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//default:
		//	cuGetResidualNeumann << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//}

		DTYPE level_mc_inv = std::exp2(deep_level);
		DTYPE level_mc = 1 / level_mc_inv;

		switch (switch_pair(mg_type_, bc))
		{
		case switch_pair(MultigridType::MG_2NM1, 'd'):
			cuGetResidualDirichlet << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case switch_pair(MultigridType::MG_2NM1, 'z'):
			cuGetResidualDirichlet00 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case switch_pair(MultigridType::MG_2NM1, 'n'):
			cuGetResidualNeumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case switch_pair(MultigridType::MG_2N, 'd'):
			cuGetResidualDirichlet_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, level_mc, max_size);
			break;
		case switch_pair(MultigridType::MG_2N, 'z'):
			cuGetResidualDirichlet00_p2 << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, level_mc_inv, max_size);
			break;
		case switch_pair(MultigridType::MG_2N, 'n'):
		case switch_pair(MultigridType::MG_2N_C4, 'n'):
			//cuGetResidualNeumann_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, level_mc, max_size);
			cuGetResidualNeumann << <BLOCKS(max_size), std::min(NUM_THREADS, max_size) >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		//switch (bc)
		//{
		//case 'd':
		//	cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//case 'z':
		//	cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//default:
		//	cuGetResidualNeumann << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
		//	break;
		//}

		CuRestriction2D::GetInstance()->Solve(f_coarses_[deep_level + 1], dev_res, dim, rs_type_, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y));
		SolveOneVCycle(dim_rs, make_DTYPE2(dx.x * 2, dx.y * 2), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		CuProlongation2D::GetInstance()->SolveAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim_rs, pg_type_, bc);
		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc, wj_type_, deep_level);
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
		//if (switch_pair(MultigridType::MG_2NM1, 'n') == switch_pair(mg_type_, bc) && sl_n_ == 1)
		//if (bc == 'n')
		//{
		//	//checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
		//	checkCudaErrors(cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE)));
		//	//RecoverNeumannPSmallLevel();
		//	//static DTYPE2 dx_sl = make_DTYPE2(dx_.x * std::exp2(max_level_ - 1), dx_.y * std::exp2(max_level_ - 1));
		//	//static DTYPE dx_l = dim_.x > dim_.y ? dx_sl.x : dx_sl.y;
		//	//cuRecoverPNuemann << <1, sl_n_ >> > (v_coarses_[deep_level], f_coarses_[deep_level], dx_l, sl_n_);
		//}
		//else
		{
			//checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
			//checkCudaErrors(cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE)));
			CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, 10, weight, bc, wj_type_, deep_level);
			//checkCudaErrors(cudaMemcpy(v_coarses_[deep_level], f_coarses_[deep_level], sizeof(DTYPE) * sl_n_, cudaMemcpyDeviceToDevice));
			//checkCudaErrors(getrs_func(handle_, CUBLAS_OP_N, sl_n_, 1, dev_A_, sl_n_, d_pivot_, v_coarses_[deep_level], sl_n_, d_info_));
			//int aa;
			//checkCudaErrors(cudaMemcpy(&aa, d_info_, sizeof(int), cudaMemcpyDeviceToHost));
			//printf("d_info: %d\n", aa);
			
		}
	}
}

void CuMultigrid2D::SolveOneVCycleP2(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE2 dxp2_inv = make_DTYPE2(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y));

	// do weightedjacobi solver
	auto dim_rs_from_dim = [](int k) { return (k + 1) / 2 - 1; };
	CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int2 dim_rs = make_int2(dim_rs_from_dim(dim.x), dim_rs_from_dim(dim.y));
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg2d", dim.x * dim.y);
		int max_size = dim.x * dim.y;
		switch (bc)
		{
		case 'd':
			cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			cuGetResidualNeumann << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		CuRestriction2D::GetInstance()->Solve(f_coarses_[deep_level + 1], dev_res, dim);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y));
		SolveOneVCycle(dim_rs, make_DTYPE2(dx.x * 2, dx.y * 2), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		CuProlongation2D::GetInstance()->SolveAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim_rs);
		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
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

		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
	}
}

void CuMultigrid2D::SolveOneVCycleRN(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	DTYPE2 dxp2_inv = make_DTYPE2(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y));

	// do weightedjacobi solver
	CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu1, weight, bc);

	if (deep_level < max_level_ - 1)
	{
		// do restriction 
		int2 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg2d", dim.x * dim.y);
		int max_size = dim.x * dim.y;
		switch (bc)
		{
		case 'd':
			cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		case 'z':
			cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		default:
			cuGetResidualNeumann << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v_coarses_[deep_level], f_coarses_[deep_level], dim, dxp2_inv, max_size);
			break;
		}

		if (mg_type_ == MultigridType::MG_2NM1_RN)
		{
			CuRestriction2D::GetInstance()->Solve2NM1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		}
		else
			CuRestriction2D::GetInstance()->SolveRN(f_coarses_[deep_level + 1], dev_res, dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y));
		SolveOneVCycleRN(dim_rs, make_DTYPE2(dx.x * 2, dx.y * 2), nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		if (mg_type_ == MultigridType::MG_2NM1_RN)
		{
			CuProlongation2D::GetInstance()->Solve2NM1RNAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		}
		else
			CuProlongation2D::GetInstance()->SolveRNAdd(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, nu2, weight, bc);
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

		CuWeightedJacobi::GetInstance()->Solve(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], dim, dx, 10, weight, bc);
	}
}

void CuMultigrid2D::Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
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

	if (mg_type_ == MultigridType::MG_RN || mg_type_ == MultigridType::MG_2NM1_RN)
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycleRN(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}
	else
	{
		for (int i = 0; i < n_iters; i++)
		{
			SolveOneVCycle(dim_, dx_, nu1, nu2, weight, bc_, 0);
		}
	}

	cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * dim_.x * dim_.y, cudaMemcpyDeviceToDevice);
}

void CuMultigrid2D::RecoverNeumannPSmallLevel()
{
}

DTYPE CuMultigrid2D::GetRhs(DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dx, char bc)
{
	int max_size = dim.x * dim.y ;
	DTYPE2 dxp2_inv = make_DTYPE2(1.f / (dx.x * dx.x), 1.f / (dx.y * dx.y));

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		//cuGetResidual3D_d_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidualDirichlet << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	case 'z':
		//cuGetResidual3D_z_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidualDirichlet00 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	default:
		//cuGetResidual3D_n_p2 << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		cuGetResidualNeumann << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, dim, dxp2_inv, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}