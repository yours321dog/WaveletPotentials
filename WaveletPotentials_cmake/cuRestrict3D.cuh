#ifndef __CURESTRIC3D_CUH__
#define __CURESTRIC3D_CUH__

#include "cudaGlobal.cuh"

class CuRestrict3D
{
public:
	CuRestrict3D() = default;
	~CuRestrict3D() = default;
	void Solve(DTYPE * dst, DTYPE * src, int3 dim, char bc = 'd');
	void SolveP2(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveM1(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveM1Old(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveRN(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveRN_c4(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');

	void SolveP1RN(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveP1RN(DTYPE* dst, DTYPE* src, DTYPE* dx, DTYPE* dy, DTYPE* dz, int3 dim, char bc = 'd');
	void SolveP1Ax(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveP1Ay(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');
	void SolveP1Az(DTYPE* dst, DTYPE* src, int3 dim, char bc = 'd');

	void SolveP1RN(DTYPE* dst, DTYPE* src, int* index, int size_idx, DTYPE* dx, DTYPE* dy, DTYPE* dz, int3 dim, char bc = 'd');

	static CuRestrict3D* GetInstance();
private:
	static std::auto_ptr<CuRestrict3D> instance_;
};

#endif