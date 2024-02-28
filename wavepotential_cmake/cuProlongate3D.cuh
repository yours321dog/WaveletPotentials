#ifndef __CUPROPOGATE3D_CUH__
#define __CUPROPOGATE3D_CUH__

#include "cudaGlobal.cuh"

class CuProlongate3D
{
public:
	CuProlongate3D();
	~CuProlongate3D();
	void Solve(DTYPE* dst, DTYPE* src, int3 dim);
	void SolveAdd(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveAddP1(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveAddM1(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveAddP2(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveAddRN(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc = 'd');
	void SolveAddRN_c4(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc = 'd');
	void SolveAddP1RN(DTYPE* dst, DTYPE* src, int3 dim_dst, int3 dim, char bc = 'd');

	void SolveAddP1RN(DTYPE* dst, DTYPE* src, int* index, int size_idx, int3 dim_dst, int3 dim,  char bc = 'd');

	static CuProlongate3D* GetInstance();
private:
	static std::auto_ptr<CuProlongate3D> instance_;
};

#endif