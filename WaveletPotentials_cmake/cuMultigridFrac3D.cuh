#ifndef __CUMULTIGRIDFRAC3D_CUH__
#define __CUMULTIGRIDFRAC3D_CUH__

#include "cudaGlobal.cuh"
#include <vector>
#include <functional>

#include "cuProlongate3D.cuh"
#include "cuRestrict3D.cuh"
#include "cuWeightedJacobi3D.cuh"

class CuMultigridFrac3D
{
public:
	CuMultigridFrac3D(DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE3 dx, char bc = 'n');
	~CuMultigridFrac3D();
	void Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);
	DTYPE GetResdualError(DTYPE* dst, DTYPE* f);
	DTYPE GetResdualErrorInt(DTYPE* dst, DTYPE* f);
	DTYPE GetResdualErrorIntLs(DTYPE* dst, DTYPE* f, DTYPE* ls);

	void Solve_index(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);

	void SolveInt(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);
	void SolveIntLs(DTYPE* dst, DTYPE* pre_v, DTYPE* f, DTYPE* ls, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);

	void InitIndex();
	void ResetFrac(DTYPE* Ax, DTYPE* Ay, DTYPE* Az);

	void ResetFracBound(DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound);
	void SolveInt_df(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);

private:
	void SolveOneVCycle(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycle_index(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleInt(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleIntLs(DTYPE* ls, int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);


	void SolveOneVCycleInt_df(int3 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);


	std::vector<DTYPE*> f_coarses_;
	std::vector<DTYPE*> v_coarses_;
	std::vector<DTYPE*> ax_coarses_;
	std::vector<DTYPE*> ay_coarses_;
	std::vector<DTYPE*> az_coarses_;
	std::vector<DTYPE*> dx_coarses_;
	std::vector<DTYPE*> dy_coarses_;
	std::vector<DTYPE*> dz_coarses_;

	std::vector<DTYPE*> ax_df_coarses_;
	std::vector<DTYPE*> ay_df_coarses_;
	std::vector<DTYPE*> az_df_coarses_;

	std::vector<int*> v_index_;
	std::vector<int> v_index_size_;

	int3 dim_;
	int max_level_;
	DTYPE3 dx_;
	DTYPE3 dx_e_;
	char bc_;
	int sl_n_;

	int stop_dim_;

	std::function<int3(int3)> dim_rs_from_dim_;
};

#endif