#ifndef __CUMULTIGRID3D_CUH__
#define __CUMULTIGRID3D_CUH__

#include "cudaGlobal.cuh"
#include <vector>
#include <functional>

#include "cuProlongate3D.cuh"
#include "cuRestrict3D.cuh"
#include "cuWeightedJacobi3D.cuh"

enum class MG3D_TYPE :unsigned int {
	MG3D_2NM1,
	MG3D_2NP1,
	MG3D_2N,
	MG3D_RN,
	MG3D_RNC4,
	MG3D_OTHER
};

class CuMultigrid3D
{
public:
	CuMultigrid3D(int3 dim, DTYPE3 dx, char bc = 'd', MG3D_TYPE mg3d_type = MG3D_TYPE::MG3D_2N);
	~CuMultigrid3D();
	void Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.8f);
	static DTYPE GetRhs(DTYPE* v, DTYPE* f, int3 dim, DTYPE3 dx, char bc);

private:
	void SolveOneVCycle(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleP1(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, DTYPE lpx, char bc, int deep_level);
	void SolveOneVCycleM1(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleRN(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleRN_c4(int3 dim, DTYPE3 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);

	std::vector<DTYPE*> f_coarses_;
	std::vector<DTYPE*> v_coarses_;
	int3 dim_;
	int size_;
	int max_level_;
	DTYPE3 dx_inv_;
	DTYPE3 dx_;
	char bc_;

	std::function<int3(int3)> dim_rs_from_dim_;
	int sl_n_;
	MG3D_TYPE mg3d_type_;
};

#endif