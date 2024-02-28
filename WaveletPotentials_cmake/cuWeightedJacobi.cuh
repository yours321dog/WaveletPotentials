#ifndef __CUWEIGHTEDJACOBI_CUH__
#define __CUWEIGHTEDJACOBI_CUH__

#include "cudaGlobal.cuh"

enum class WeightedJacobiType {
	WJ_2NM1,
	WJ_2NP1,
	WJ_2N,
	WJ_OTHER
};

class CuWeightedJacobi
{
public:
	CuWeightedJacobi();
	~CuWeightedJacobi();
	void Solve(DTYPE* dst, DTYPE* src, DTYPE* f, int2 dim, DTYPE2 dx, int n_iters, DTYPE weight,
		char bc = 'z', WeightedJacobiType wj_type = WeightedJacobiType::WJ_OTHER, int level = 0);
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx, int n_iters, DTYPE weight,
		char bc = 'z', WeightedJacobiType wj_type = WeightedJacobiType::WJ_OTHER, int level = 0);
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, int n_iters, DTYPE weight,
		char bc = 'z', WeightedJacobiType wj_type = WeightedJacobiType::WJ_OTHER, int level = 0);
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, int n_iters, DTYPE weight,
		char bc = 'z');

	static DTYPE GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE2 dx, char bc);

	static DTYPE GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc);
	static double GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc);
	static void GetFracRhs(DTYPE* res, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, char bc);
	static void GetFracRhs(DTYPE* res, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, int2 dim, DTYPE* dx, DTYPE* dy, DTYPE2 dx_e, char bc);

	static CuWeightedJacobi* GetInstance();
private:
	static std::auto_ptr<CuWeightedJacobi> instance_;
};

#endif