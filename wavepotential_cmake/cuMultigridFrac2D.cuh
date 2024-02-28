#ifndef __CUMULTIGRIDFRAC_CUH__
#define __CUMULTIGRIDFRAC_CUH__

#include "cudaGlobal.cuh"
#include <vector>
#include <functional>

#include "cuProlongation2D.cuh"
#include "cuRestriction2D.cuh"
#include "cuWeightedJacobi.cuh"

class CuMultigridFrac2D
{
public:
	CuMultigridFrac2D(DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx, char bc = 'n');
	~CuMultigridFrac2D();
	void Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8);

	static void RestrictDx1D(DTYPE* dst_dx, DTYPE* src_dx, int dim_dst, int dim);
	static void InitDx1D(DTYPE* dst_dx, DTYPE dx, int dim);

private:
	void SolveOneVCycle(int2 dim, int nu1, int nu2, DTYPE weight, char bc, int deep_level);

	std::vector<DTYPE*> f_coarses_;
	std::vector<DTYPE*> v_coarses_;
	std::vector<DTYPE*> ax_coarses_;
	std::vector<DTYPE*> ay_coarses_;
	std::vector<DTYPE*> dx_coarses_;
	std::vector<DTYPE*> dy_coarses_;

	int2 dim_;
	int max_level_;
	DTYPE2 dx_inv_;
	DTYPE2 dx_;
	DTYPE2 dx_e_;
	char bc_;
	int sl_n_;

	std::function<int2(int2)> dim_rs_from_dim_;
};

#endif