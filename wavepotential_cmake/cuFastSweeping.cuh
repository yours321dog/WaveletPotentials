#ifndef __CUFASTSWEEPING_CUH__
#define __CUFASTSWEEPING_CUH__

#include "cudaGlobal.cuh"

class CuFastSweeping
{
public:
	CuFastSweeping();
	~CuFastSweeping();
	static CuFastSweeping* GetInstance();

	static void Solve(DTYPE* ls, int3 dim, DTYPE dx, int n_iters = 5);
	static void ParticlesToLs(DTYPE* ls, DTYPE3* x, int3 dim, int n_pars, DTYPE dx, DTYPE radius, DTYPE max_dis, int n_iter = 5);
	static void ParticlesToLsNarrow(DTYPE* ls, DTYPE3* x, int3 dim, int n_pars, DTYPE dx, DTYPE radius, DTYPE max_dis, int n_iter = 5);
	static void ReinitLevelSet(DTYPE* ls, int3 dim, DTYPE dx, int n_iter = 5);

private:
	static std::auto_ptr<CuFastSweeping> instance_;
};
#endif