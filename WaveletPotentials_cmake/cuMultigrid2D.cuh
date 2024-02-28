#ifndef __CUMULTIGRID2D_CUH__
#define __CUMULTIGRID2D_CUH__

#include <cusolverDn.h>

#include "cudaGlobal.cuh"
#include <vector>
#include <functional>

#include "cuProlongation2D.cuh"
#include "cuRestriction2D.cuh"
#include "cuWeightedJacobi.cuh"

enum class MultigridType :unsigned int {
	MG_2NM1,
	MG_2NP1,
	MG_2N,
	MG_2N_C4,
	MG_RN,
	MG_2NM1_RN,
	MG_OTHER
};


class CuMultigrid2D
{
public:
	CuMultigrid2D(int2 dim, DTYPE2 dx, char bc = 'z', MultigridType mg_type = MultigridType::MG_2NM1);
	~CuMultigrid2D();
	void Solve(DTYPE* dst, DTYPE* pre_v, DTYPE* f, int n_iters = 10, int nu1 = 2, int nu2 = 2, DTYPE weight = 0.75);

	static DTYPE GetRhs(DTYPE* v, DTYPE* f, int2 dim, DTYPE2 dx, char bc);

private:
	void SolveOneVCycle(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleP2(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);
	void SolveOneVCycleRN(int2 dim, DTYPE2 dx, int nu1, int nu2, DTYPE weight, char bc, int deep_level);

	void RecoverNeumannPSmallLevel();


	MultigridType mg_type_;
	std::vector<DTYPE *> f_coarses_;
	std::vector<DTYPE *> v_coarses_;
	int2 dim_;
	int max_level_;
	DTYPE2 dx_inv_;
	DTYPE2 dx_;
	char bc_;

	std::function<int2(int2)> dim_rs_from_dim_;

	RestrictionType rs_type_;
	ProlongationType pg_type_;
	WeightedJacobiType wj_type_;

	//// solve the smallest level with cusolver
	//DTYPE* dev_A_;
	//cusolverDnHandle_t handle_;
	//int* d_pivot_;
	//int* d_info_;
	int sl_n_;
};

#endif