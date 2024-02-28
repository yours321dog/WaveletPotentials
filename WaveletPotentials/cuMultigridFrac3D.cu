#include "cuMultigridFrac3D.cuh"
#include "cuMultigridFrac2D.cuh"
#include "CuMemoryManager.cuh"

#include "PrefixSum.cuh"

#include "cuBoundSelector.cuh"

CuMultigridFrac3D::CuMultigridFrac3D(DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE3 dx, char bc) : dim_(dim), max_level_(0), dx_(dx), bc_(bc),
v_index_(0), stop_dim_(1)
{
	dx_e_ = bc == 'd' ? make_DTYPE3(0.f, 0.f, 0.f) : dx;
	bc_ = bc == 'd' ? 'z' : bc;
	int min_dim = std::min(std::min(dim.x, dim.y), dim.z);
	//dx_inv_ = make_DTYPE2(1.f / dx.x, 1.f / dx.y);
	auto dim_rs_from_dim_1d = [](int dim_in) -> int {
		int dim_out;
		if ((dim_in & 1) == 0)
			dim_out = dim_in >> 1;
		else
			dim_out = (dim_in >> 1) + 1;
		return dim_out;
	};
	dim_rs_from_dim_ = [&dim_rs_from_dim_1d](int3 dim_in) -> int3 {
		int3 dim_out = { dim_rs_from_dim_1d(dim_in.x), dim_rs_from_dim_1d(dim_in.y),
			dim_rs_from_dim_1d(dim_in.z) };
		return dim_out;
	};

	int3 dim_rs = dim;
	int3 last_rs = dim;
	while (min_dim > stop_dim_)
	{
		sl_n_ = std::max(dim_rs.x, std::max(dim_rs.y, dim_rs.z));
		int size_coarse = dim_rs.x * dim_rs.y * dim_rs.z;

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
		int3 dim_ax_rs = { dim_rs.x + 1, dim_rs.y, dim_rs.z };
		int size_ax = dim_ax_rs.x * dim_ax_rs.y * dim_ax_rs.z;
		checkCudaErrors(cudaMalloc((void**)&dev_ax, size_ax * sizeof(DTYPE)));

		DTYPE* dev_ay;
		int3 dim_ay_rs = { dim_rs.x, dim_rs.y + 1, dim_rs.z };
		int size_ay = dim_ay_rs.y * dim_ay_rs.x * dim_ay_rs.z;
		checkCudaErrors(cudaMalloc((void**)&dev_ay, size_ay * sizeof(DTYPE)));

		DTYPE* dev_az;
		int3 dim_az_rs = { dim_rs.x, dim_rs.y, dim_rs.z + 1 };
		int size_az = dim_az_rs.y * dim_az_rs.x * dim_az_rs.z;
		checkCudaErrors(cudaMalloc((void**)&dev_az, size_az * sizeof(DTYPE)));

		if (max_level_ == 0)
		{
			checkCudaErrors(cudaMemcpy(dev_ax, Ax, size_ax * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(dev_ay, Ay, size_ay * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(dev_az, Az, size_az * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
		}
		else
		{
			CuRestrict3D::GetInstance()->SolveP1Ax(dev_ax, ax_coarses_[max_level_ - 1], make_int3(last_rs.x + 1, last_rs.y, last_rs.z), bc);
			CuRestrict3D::GetInstance()->SolveP1Ay(dev_ay, ay_coarses_[max_level_ - 1], make_int3(last_rs.x, last_rs.y + 1, last_rs.z), bc);
			CuRestrict3D::GetInstance()->SolveP1Az(dev_az, az_coarses_[max_level_ - 1], make_int3(last_rs.x, last_rs.y, last_rs.z + 1), bc);
		}
		//CudaPrintfMat(dev_ax, dim_ax_rs);
		//CudaPrintfMat(dev_ay, dim_ay_rs);
		//CudaPrintfMat(dev_az, dim_az_rs);
		ax_coarses_.push_back(dev_ax);
		ay_coarses_.push_back(dev_ay);
		az_coarses_.push_back(dev_az);

		/***************************************************************************************************/
		checkCudaErrors(cudaMalloc((void**)&dev_ax, size_ax * sizeof(DTYPE)));
		checkCudaErrors(cudaMalloc((void**)&dev_ay, size_ay * sizeof(DTYPE)));
		checkCudaErrors(cudaMalloc((void**)&dev_az, size_az * sizeof(DTYPE)));

		if (max_level_ == 0)
		{
			checkCudaErrors(cudaMemcpy(dev_ax, Ax, size_ax * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(dev_ay, Ay, size_ay * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(dev_az, Az, size_az * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
		}
		else
		{
			CuRestrict3D::GetInstance()->SolveP1Ax(dev_ax, ax_coarses_[max_level_ - 1], make_int3(last_rs.x + 1, last_rs.y, last_rs.z), bc);
			CuRestrict3D::GetInstance()->SolveP1Ay(dev_ay, ay_coarses_[max_level_ - 1], make_int3(last_rs.x, last_rs.y + 1, last_rs.z), bc);
			CuRestrict3D::GetInstance()->SolveP1Az(dev_az, az_coarses_[max_level_ - 1], make_int3(last_rs.x, last_rs.y, last_rs.z + 1), bc);
		}
		ax_df_coarses_.push_back(dev_ax);
		ay_df_coarses_.push_back(dev_ay);
		az_df_coarses_.push_back(dev_az);
		/***************************************************************************************************/

		// create coarse grids for ax & create coarse grids for ax
		DTYPE* dev_dx;
		checkCudaErrors(cudaMalloc((void**)&dev_dx, dim_rs.x * sizeof(DTYPE)));
		DTYPE* dev_dy;
		checkCudaErrors(cudaMalloc((void**)&dev_dy, dim_rs.y * sizeof(DTYPE)));
		DTYPE* dev_dz;
		checkCudaErrors(cudaMalloc((void**)&dev_dz, dim_rs.z * sizeof(DTYPE)));
		if (max_level_ == 0)
		{
			CuMultigridFrac2D::InitDx1D(dev_dx, dx.x, dim.x);
			CuMultigridFrac2D::InitDx1D(dev_dy, dx.y, dim.y);
			CuMultigridFrac2D::InitDx1D(dev_dz, dx.z, dim.z);
		}
		else
		{
			CuMultigridFrac2D::RestrictDx1D(dev_dx, dx_coarses_[max_level_ - 1], dim_rs.x, last_rs.x);
			CuMultigridFrac2D::RestrictDx1D(dev_dy, dy_coarses_[max_level_ - 1], dim_rs.y, last_rs.y);
			CuMultigridFrac2D::RestrictDx1D(dev_dz, dz_coarses_[max_level_ - 1], dim_rs.z, last_rs.z);
		}
		dx_coarses_.push_back(dev_dx);
		dy_coarses_.push_back(dev_dy);
		dz_coarses_.push_back(dev_dz);

		last_rs = dim_rs;
		dim_rs = dim_rs_from_dim_(dim_rs);
		min_dim = std::min(dim_rs.x, std::min(dim_rs.y, dim_rs.z));
		++max_level_;
	}
}

CuMultigridFrac3D::~CuMultigridFrac3D()
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
	for (DTYPE* ptr : az_coarses_)
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
	for (DTYPE* ptr : dz_coarses_)
	{
		cudaFree(ptr);
	}
	for (int* ptr : v_index_)
	{
		cudaFree(ptr);
	}

	for (DTYPE* ptr : ax_df_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : ay_df_coarses_)
	{
		cudaFree(ptr);
	}
	for (DTYPE* ptr : az_df_coarses_)
	{
		cudaFree(ptr);
	}
}

void CuMultigridFrac3D::InitIndex()
{
	if (v_index_.size() == 0)
	{
		auto dim_rs_from_dim_1d = [](int dim_in) -> int {
			int dim_out;
			if ((dim_in & 1) == 0)
				dim_out = dim_in >> 1;
			else
				dim_out = (dim_in >> 1) + 1;
			return dim_out;
		};


		int size_0 = getsize(dim_) + 1;
		int* index_select;
		int* index_sum;

		checkCudaErrors(cudaMalloc((void**)&index_select, sizeof(int) * size_0));
		checkCudaErrors(cudaMalloc((void**)&index_sum, sizeof(int) * size_0));

		checkCudaErrors(cudaMemset(index_select, 0, sizeof(int) * (size_0)));

		int3 dim = dim_;
		for (int i = 0; i < v_coarses_.size(); i++)
		{
			//printf("init dim: %d, %d, %d\n", dim.x, dim.y, dim.z);
			int size_coarse = getsize(dim);
			int size_x = getsize(dim) + 1;

			CuBoundSelector::SelectByFrac(index_select, ax_coarses_[i], ay_coarses_[i], az_coarses_[i], dim);
			//CudaPrintfMat<int>(index_select, dim);
			PrefixSum pfs_index(index_sum, index_select, size_x);
			pfs_index.Do();
			int index_size = 0;
			//CudaPrintfMat<int>(index_sum, dim);
			checkCudaErrors(cudaMemcpy(&index_size, &index_sum[size_coarse], sizeof(int), cudaMemcpyDeviceToHost));
			v_index_size_.push_back(index_size);

			int* dev_index;
			checkCudaErrors(cudaMalloc((void**)&dev_index, index_size * sizeof(int)));
			CuBoundSelector::SelectToIndex(dev_index, index_sum, size_x);
			v_index_.push_back(dev_index);

			int last_idx;
			if (index_size > 1)
			{
				checkCudaErrors(cudaMemcpy(&last_idx, &dev_index[index_size - 1], sizeof(int), cudaMemcpyDeviceToHost));
			}
			printf("init dim: %d, %d, %d, index_size: %d, last_idx: %d, total_dim: %d\n", dim.x, dim.y, dim.z, index_size, last_idx, size_coarse);
			//CudaPrintfMat(index_sum, make_int2(1, index_size));

			dim = dim_rs_from_dim_(dim);
		}

		checkCudaErrors(cudaFree(index_select));
		checkCudaErrors(cudaFree(index_sum));
	}
}

void CuMultigridFrac3D::Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	int max_size = dim_.x * dim_.y * dim_.z;
	if (pre_v != nullptr)
	{
		cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
	else
	{
		cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * max_size);
	}
	cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycle(dim_, nu1, nu2, weight, bc_, 0);
	cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
}

void CuMultigridFrac3D::SolveOneVCycle(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	if (deep_level < max_level_ - 1)
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", dim.x * dim.y * dim.z);

		// get residual
		//CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], bc);
		CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, bc);
		//CudaPrintfMat(dev_res, dim);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y * dim_rs.z));
		SolveOneVCycle(dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE) * dim.x * dim.y * dim.z);
		//CudaPrintfMat(v_coarses_[deep_level + 1], dim_rs);
		CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], nu2, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim); CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, 10, weight, bc);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(dz_coarses_[deep_level], make_int2(1, dim.z));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int3(dim.x + 1, dim.y, dim.z));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int3(dim.x, dim.y + 1, dim.z));
		//CudaPrintfMat(az_coarses_[deep_level], make_int3(dim.x, dim.y, dim.z + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

DTYPE CuMultigridFrac3D::GetResdualError(DTYPE* dst, DTYPE* f)
{
	return CuWeightedJacobi3D::GetFracRhs(dst, f, ax_coarses_[0], ay_coarses_[0], az_coarses_[0], dim_, dx_coarses_[0],
		dy_coarses_[0], dz_coarses_[0], dx_e_, bc_);
}

DTYPE CuMultigridFrac3D::GetResdualErrorInt(DTYPE* dst, DTYPE* f)
{
	return CuWeightedJacobi3D::GetFracRhsInt(dst, f, ax_coarses_[0], ay_coarses_[0], az_coarses_[0], nullptr, dim_, dx_coarses_[0],
		dy_coarses_[0], dz_coarses_[0], dx_, bc_);
}

DTYPE CuMultigridFrac3D::GetResdualErrorIntLs(DTYPE* dst, DTYPE* f, DTYPE* ls)
{
	return CuWeightedJacobi3D::GetFracRhsIntLs(dst, f, ax_coarses_[0], ay_coarses_[0], az_coarses_[0], ls, dim_, dim_, dx_coarses_[0],
		dy_coarses_[0], dz_coarses_[0], dx_, dx_.x, bc_);
}

void CuMultigridFrac3D::Solve_index(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	InitIndex();
	int max_size = dim_.x * dim_.y * dim_.z;
	if (pre_v != nullptr)
	{
		checkCudaErrors(cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * max_size));
	}
	cudaCheckError(cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycle_index(dim_, nu1, nu2, weight, bc_, 0);
	cudaCheckError(cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
}

void CuMultigridFrac3D::SolveOneVCycle_index(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	//printf("deep_level: %d, dim: %d, %d, %d, v_index_size_[deep_level]: %d, v_index_size_[deep_level + 1]: %d\n", deep_level, dim.x, dim.y, dim.z, v_index_size_[deep_level],
	//	v_index_size_[deep_level + 1]);
	if (deep_level < max_level_ - 1 && v_index_size_[deep_level + 1] > 0)
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], v_index_[deep_level], v_index_size_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 
			dz_coarses_[deep_level], dx_e_, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", dim.x * dim.y * dim.z);

		// get residual
		//CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], bc);
		CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], v_index_[deep_level], v_index_size_[deep_level], dim, 
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, bc);
		//CudaPrintfMat(dev_res, dim);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);
		CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, v_index_[deep_level + 1], v_index_size_[deep_level + 1],
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y * dim_rs.z));
		SolveOneVCycle_index(dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE) * dim.x * dim.y * dim.z);
		//CudaPrintfMat(v_coarses_[deep_level + 1], dim_rs);
		//CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1],  
			v_index_[deep_level], v_index_size_[deep_level], dim, dim_rs, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], nu2, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu2, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], v_index_[deep_level], v_index_size_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level],
			dz_coarses_[deep_level], dx_e_, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim); CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_e_, 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(dz_coarses_[deep_level], make_int2(1, dim.z));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int3(dim.x + 1, dim.y, dim.z));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int3(dim.x, dim.y + 1, dim.z));
		//CudaPrintfMat(az_coarses_[deep_level], make_int3(dim.x, dim.y, dim.z + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

/**************************************************SolveOneVCycleInt***********************************************************/
void CuMultigridFrac3D::ResetFrac(DTYPE* Ax, DTYPE* Ay, DTYPE* Az)
{
	int3& dim = dim_;
	int min_dim = std::min(std::min(dim.x, dim.y), dim.z);
	int3 dim_rs = dim;
	int3 last_rs = dim;
	int max_level = 0;
	while (min_dim > stop_dim_)
	{
		sl_n_ = std::max(dim_rs.x, std::max(dim_rs.y, dim_rs.z));
		int size_coarse = dim_rs.x * dim_rs.y * dim_rs.z;

		// create coarse grids for ax & create coarse grids for ax
		int3 dim_ax_rs = { dim_rs.x + 1, dim_rs.y, dim_rs.z };
		int size_ax = dim_ax_rs.x * dim_ax_rs.y * dim_ax_rs.z;

		int3 dim_ay_rs = { dim_rs.x, dim_rs.y + 1, dim_rs.z };
		int size_ay = dim_ay_rs.y * dim_ay_rs.x * dim_ay_rs.z;

		int3 dim_az_rs = { dim_rs.x, dim_rs.y, dim_rs.z + 1 };
		int size_az = dim_az_rs.y * dim_az_rs.x * dim_az_rs.z;

		if (max_level == 0)
		{
			checkCudaErrors(cudaMemcpy(ax_coarses_[max_level], Ax, size_ax * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(ay_coarses_[max_level], Ay, size_ay * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(az_coarses_[max_level], Az, size_az * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
		}
		else
		{
			CuRestrict3D::GetInstance()->SolveP1Ax(ax_coarses_[max_level], ax_coarses_[max_level - 1], make_int3(last_rs.x + 1, last_rs.y, last_rs.z), bc_);
			CuRestrict3D::GetInstance()->SolveP1Ay(ay_coarses_[max_level], ay_coarses_[max_level - 1], make_int3(last_rs.x, last_rs.y + 1, last_rs.z), bc_);
			CuRestrict3D::GetInstance()->SolveP1Az(az_coarses_[max_level], az_coarses_[max_level - 1], make_int3(last_rs.x, last_rs.y, last_rs.z + 1), bc_);
		}

		last_rs = dim_rs;
		dim_rs = dim_rs_from_dim_(dim_rs);
		min_dim = std::min(dim_rs.x, std::min(dim_rs.y, dim_rs.z));
		max_level++;
	}
}

void CuMultigridFrac3D::SolveInt(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	int max_size = dim_.x * dim_.y * dim_.z;
	if (pre_v != nullptr)
	{
		checkCudaErrors(cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * max_size));
	}
	cudaCheckError(cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycleInt(dim_, nu1, nu2, weight, bc_, 0);
	cudaCheckError(cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
}

void CuMultigridFrac3D::SolveOneVCycleInt(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	//printf("deep_level: %d, dim: %d, %d, %d, v_index_size_[deep_level]: %d, v_index_size_[deep_level + 1]: %d\n", deep_level, dim.x, dim.y, dim.z, v_index_size_[deep_level],
	//	v_index_size_[deep_level + 1]);
	//printf("deep_level: % d\n", deep_level);
	int size = getsize(dim);
	//printf("size: %d\n", size);
	if (deep_level < max_level_ - 1)
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		if (size > NUM_THREADS)
			CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
				az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu1, weight, bc);
		else
			CuWeightedJacobi3D::GetInstance()->SolveFracIntKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
				az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", dim.x * dim.y * dim.z);

		// get residual
		//CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], bc);
		CuWeightedJacobi3D::GetFracRhsInt(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, bc);
		//CudaPrintfMat(dev_res, dim);
		CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, v_index_[deep_level + 1], v_index_size_[deep_level + 1],
		//	dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y * dim_rs.z));
		SolveOneVCycleInt(dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE) * dim.x * dim.y * dim.z);
		//CudaPrintfMat(v_coarses_[deep_level + 1], dim_rs);
		CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		//CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1],
		//	v_index_[deep_level], v_index_size_[deep_level], dim, dim_rs, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], nu2, weight, bc);
		if (size > NUM_THREADS)
			CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
				az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu2, weight, bc);
		else
			CuWeightedJacobi3D::GetInstance()->SolveFracIntKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
				az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim); CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracIntKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuConjugateGradient::GetInstance()->SolveFracPcg_int(v_coarses_[deep_level], v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, bc, 1e-6f, 10);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(dz_coarses_[deep_level], make_int2(1, dim.z));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int3(dim.x + 1, dim.y, dim.z));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int3(dim.x, dim.y + 1, dim.z));
		//CudaPrintfMat(az_coarses_[deep_level], make_int3(dim.x, dim.y, dim.z + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

/**************************************************SolveOneVCycleIntLs***********************************************************/
void CuMultigridFrac3D::SolveIntLs(DTYPE* dst, DTYPE* pre_v, DTYPE* f, DTYPE* ls, int n_iters, int nu1, int nu2, DTYPE weight)
{
	int max_size = dim_.x * dim_.y * dim_.z;
	if (pre_v != nullptr)
	{
		checkCudaErrors(cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * max_size));
	}
	cudaCheckError(cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycleIntLs(ls, dim_, nu1, nu2, weight, bc_, 0);
	cudaCheckError(cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
}

void CuMultigridFrac3D::SolveOneVCycleIntLs(DTYPE* ls, int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	//printf("deep_level: %d, dim: %d, %d, %d, v_index_size_[deep_level]: %d, v_index_size_[deep_level + 1]: %d\n", deep_level, dim.x, dim.y, dim.z, v_index_size_[deep_level],
	//	v_index_size_[deep_level + 1]);
	//printf("deep_level: % d\n", deep_level);
	if (deep_level < max_level_ - 1)
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracIntLs(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ls, dim, dim_, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, dx_.x, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", dim.x * dim.y * dim.z);

		// get residual
		//CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], bc);
		CuWeightedJacobi3D::GetFracRhsIntLs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ls, dim, dim_, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, dx_.x, bc);
		//CudaPrintfMat(dev_res, dim);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, v_index_[deep_level + 1], v_index_size_[deep_level + 1],
		//	dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y * dim_rs.z));
		SolveOneVCycleIntLs(ls, dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE) * dim.x * dim.y * dim.z);
		//CudaPrintfMat(v_coarses_[deep_level + 1], dim_rs);
		CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		//CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1],
		//	v_index_[deep_level], v_index_size_[deep_level], dim, dim_rs, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], nu2, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracIntLs(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ls, dim, dim_, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, dx_.x, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim); CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracIntLs(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ls, dim, dim_, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, dx_.x, 10, weight, bc);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(dz_coarses_[deep_level], make_int2(1, dim.z));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int3(dim.x + 1, dim.y, dim.z));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int3(dim.x, dim.y + 1, dim.z));
		//CudaPrintfMat(az_coarses_[deep_level], make_int3(dim.x, dim.y, dim.z + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

/*********************************************************************************************************************************/
__global__ void cuBound_df(DTYPE* ax_out, DTYPE* ay_out, DTYPE* az_out, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			// bottom
			ay_out[idx_ay] = 1.f - Ay[idx_ay];
			// top
			ay_out[idx_ay + 1] = 1.f - Ay[idx_ay + 1];
			// left
			ax_out[idx_ax] = 1.f - Ax[idx_ax];
			// right
			ax_out[idx_ax + dim.y] = 1.f - Ax[idx_ax + dim.y];
			// front
			az_out[idx_az] = 1.f - Az[idx_az];
			// back
			az_out[idx_az + slice_xy] = 1.f - Az[idx_az + slice_xy];
		}
		else
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			// bottom
			ay_out[idx_ay] =  0.f;
			// top
			ay_out[idx_ay + 1] = 0.f;
			// left
			ax_out[idx_ax] = 0.f;
			// right
			ax_out[idx_ax + dim.y] = 0.f;
			// front
			az_out[idx_az] = 0.f;
			// back
			az_out[idx_az + slice_xy] = 0.f;
		}
	}
}

void CuMultigridFrac3D::SolveInt_df(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters, int nu1, int nu2, DTYPE weight)
{
	int max_size = dim_.x * dim_.y * dim_.z;
	if (pre_v != nullptr)
	{
		checkCudaErrors(cudaMemcpy(v_coarses_[0], pre_v, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
	}
	else
	{
		checkCudaErrors(cudaMemset(v_coarses_[0], 0, sizeof(DTYPE) * max_size));
	}
	cudaCheckError(cudaMemcpy(f_coarses_[0], f, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < n_iters; i++)
		SolveOneVCycleInt_df(dim_, nu1, nu2, weight, bc_, 0);
	cudaCheckError(cudaMemcpy(dst, v_coarses_[0], sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
}

void CuMultigridFrac3D::SolveOneVCycleInt_df(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level)
{
	if (deep_level < max_level_ - 1)
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu1, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ax_df_coarses_[deep_level], ay_df_coarses_[deep_level], az_df_coarses_[deep_level], nullptr, dim,
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu1, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);

		// do restriction 
		int3 dim_rs = dim_rs_from_dim_(dim);
		DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("mg3d", dim.x * dim.y * dim.z);

		// get residual
		//CuWeightedJacobi3D::GetFracRhs(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], bc);
		//CuWeightedJacobi3D::GetFracRhsInt(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, bc);
		CuWeightedJacobi3D::GetFracRhsInt(dev_res, v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ax_df_coarses_[deep_level], ay_df_coarses_[deep_level], az_df_coarses_[deep_level], nullptr, dim,
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, bc);
		//CudaPrintfMat(dev_res, dim);
		CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);
		//CuRestrict3D::GetInstance()->SolveP1RN(f_coarses_[deep_level + 1], dev_res, v_index_[deep_level + 1], v_index_size_[deep_level + 1],
		//	dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dim, bc);

		// next vcycle
		checkCudaErrors(cudaMemset(v_coarses_[deep_level + 1], 0, sizeof(DTYPE) * dim_rs.x * dim_rs.y * dim_rs.z));
		SolveOneVCycleInt_df(dim_rs, nu1, nu2, weight, bc, deep_level + 1);

		// do prolongation
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//cudaMemset(v_coarses_[deep_level], 0, sizeof(DTYPE) * dim.x * dim.y * dim.z);
		//CudaPrintfMat(v_coarses_[deep_level + 1], dim_rs);
		CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1], dim, dim_rs, bc);
		//CuProlongate3D::GetInstance()->SolveAddP1RN(v_coarses_[deep_level], v_coarses_[deep_level + 1],
		//	v_index_[deep_level], v_index_size_[deep_level], dim, dim_rs, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim);
		//CuWeightedJacobi3D::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], nu2, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu2, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ax_df_coarses_[deep_level], ay_df_coarses_[deep_level], az_df_coarses_[deep_level], nullptr, dim,
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, nu2, weight, bc);
		//CudaPrintfMat(v_coarses_[deep_level], dim); CudaPrintfMat(v_coarses_[deep_level], dim);
	}
	else
	{
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracKernel(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		CuWeightedJacobi3D::GetInstance()->SolveFracInt(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
			az_coarses_[deep_level], ax_df_coarses_[deep_level], ay_df_coarses_[deep_level], az_df_coarses_[deep_level], nullptr, dim,
			dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, 10, weight, bc);
		//CuConjugateGradient::GetInstance()->SolveFracPcg_int(v_coarses_[deep_level], v_coarses_[deep_level], f_coarses_[deep_level], ax_coarses_[deep_level], ay_coarses_[deep_level],
		//	az_coarses_[deep_level], nullptr, dim, dx_coarses_[deep_level], dy_coarses_[deep_level], dz_coarses_[deep_level], dx_, bc, 1e-6f, 10);
		//CuWeightedJacobi::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], 5, weight, bc);
		//CudaPrintfMat(dx_coarses_[deep_level], make_int2(1, dim.x));
		//CudaPrintfMat(dy_coarses_[deep_level], make_int2(1, dim.y));
		//CudaPrintfMat(dz_coarses_[deep_level], make_int2(1, dim.z));
		//CudaPrintfMat(ax_coarses_[deep_level], make_int3(dim.x + 1, dim.y, dim.z));
		//CudaPrintfMat(ay_coarses_[deep_level], make_int3(dim.x, dim.y + 1, dim.z));
		//CudaPrintfMat(az_coarses_[deep_level], make_int3(dim.x, dim.y, dim.z + 1));
		//CuConjugateGradient::GetInstance()->SolveFrac(v_coarses_[deep_level], nullptr, f_coarses_[deep_level], ax_coarses_[deep_level],
		//	ay_coarses_[deep_level], dim, dx_coarses_[deep_level], dy_coarses_[deep_level], bc, 1e-5f, 100);
	}
}

void CuMultigridFrac3D::ResetFracBound(DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound)
{
	int3& dim = dim_;
	int min_dim = std::min(std::min(dim.x, dim.y), dim.z);
	int3 dim_rs = dim;
	int3 last_rs = dim;
	int max_level = 0;

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("is_bound", getsize(dim));
		CuWeightedJacobi3D::GetBound(is_bound, Ax, Ay, Az, dim);
	}

	while (min_dim > stop_dim_)
	{
		sl_n_ = std::max(dim_rs.x, std::max(dim_rs.y, dim_rs.z));
		int size_coarse = dim_rs.x * dim_rs.y * dim_rs.z;

		// create coarse grids for ax & create coarse grids for ax
		int3 dim_ax_rs = { dim_rs.x + 1, dim_rs.y, dim_rs.z };
		int size_ax = dim_ax_rs.x * dim_ax_rs.y * dim_ax_rs.z;

		int3 dim_ay_rs = { dim_rs.x, dim_rs.y + 1, dim_rs.z };
		int size_ay = dim_ay_rs.y * dim_ay_rs.x * dim_ay_rs.z;

		int3 dim_az_rs = { dim_rs.x, dim_rs.y, dim_rs.z + 1 };
		int size_az = dim_az_rs.y * dim_az_rs.x * dim_az_rs.z;

		if (max_level == 0)
		{
			//checkCudaErrors(cudaMemcpy(ax_df_coarses_[max_level], Ax, size_ax * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			//checkCudaErrors(cudaMemcpy(ay_coarses_[max_level], Ay, size_ay * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			//checkCudaErrors(cudaMemcpy(az_coarses_[max_level], Az, size_az * sizeof(DTYPE), cudaMemcpyDeviceToDevice));
			cuBound_df << <BLOCKS(size_coarse), THREADS(size_coarse) >> > (ax_df_coarses_[max_level], ay_df_coarses_[max_level],
				az_df_coarses_[max_level], Ax, Ay, Az, is_bound, dim, size_coarse);
		}
		else
		{
			CuRestrict3D::GetInstance()->SolveP1Ax(ax_df_coarses_[max_level], ax_df_coarses_[max_level - 1], make_int3(last_rs.x + 1, last_rs.y, last_rs.z), bc_);
			CuRestrict3D::GetInstance()->SolveP1Ay(ay_df_coarses_[max_level], ay_df_coarses_[max_level - 1], make_int3(last_rs.x, last_rs.y + 1, last_rs.z), bc_);
			CuRestrict3D::GetInstance()->SolveP1Az(az_df_coarses_[max_level], az_df_coarses_[max_level - 1], make_int3(last_rs.x, last_rs.y, last_rs.z + 1), bc_);
		}

		last_rs = dim_rs;
		dim_rs = dim_rs_from_dim_(dim_rs);
		min_dim = std::min(dim_rs.x, std::min(dim_rs.y, dim_rs.z));
		max_level++;
	}
}