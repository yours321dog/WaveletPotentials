#ifndef __CUWEIGHTJACOBI3D_CUH__
#define __CUWEIGHTJACOBI3D_CUH__

#include "cudaGlobal.cuh"

class CuWeightedJacobi3D
{
public:
	CuWeightedJacobi3D();
	~CuWeightedJacobi3D();
	void Solve(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dx, int n_iters, DTYPE weight,
		char bc = 'z');
	void SolveP1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dx, int n_iters, DTYPE weight, DTYPE lpx,
		char bc = 'z');
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE3 dx,
		int n_iters, DTYPE weight, char bc = 'z');
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz,
		int n_iters, DTYPE weight, char bc = 'z');
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');

	void SolveFracKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz,
		int n_iters, DTYPE weight, char bc = 'z');
	void SolveFracKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');
	static DTYPE GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc);
	static void GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc);

	// interp
	void SolveFracInt(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound, 
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');
	void SolveFracIntKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');
	static DTYPE GetFracRhsInt(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);
	static void GetFracRhsInt(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);

	// interp ls
	void SolveFracIntLs(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
		int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls,
		int n_iters, DTYPE weight, char bc = 'z');
	static DTYPE GetFracRhsIntLs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
		int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls, char bc);
	static void GetFracRhsIntLs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
		int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls, char bc);

	// interp df
	void SolveFracInt(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');
	static DTYPE GetFracRhsInt(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);
	static void GetFracRhsInt(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);

	// Local Index
	void SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index, int size_idx, 
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
		int n_iters, DTYPE weight, char bc = 'z');
	static void GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index, int size_idx, 
		int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);

	static DTYPE GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);
	static void GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc);
	static void BoundB(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim);
	static void GetBound(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim);
	static void GetBound_any(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim);
	static CuWeightedJacobi3D* GetInstance();
private:
	static std::auto_ptr<CuWeightedJacobi3D> instance_;
};

#endif