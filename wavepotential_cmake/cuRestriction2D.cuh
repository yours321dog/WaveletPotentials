#ifndef __CURESTRICTION2D_CUH__
#define __CURESTRICTION2D_CUH__

#include "cudaGlobal.cuh"

enum class RestrictionType {
	RT_2NM1,
	RT_2NP1,
	RT_2N,
	RT_2N_C4,
	RT_2NM1_RN,
	RT_OTHER
};

class CuRestriction2D
{
public:
	CuRestriction2D();
	~CuRestriction2D();
	void Solve(DTYPE* dst, DTYPE* src, int2 dim, RestrictionType rs_type = RestrictionType::RT_2NM1, char bc = 'd');
	void SolveRN(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');
	void Solve2NM1RN(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');
	void SolveP1RN(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');

	void SolveP1RN(DTYPE* dst, DTYPE* src, DTYPE* dx, DTYPE* dy, int2 dim, char bc = 'd');

	void SolveAx(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');
	void SolveAy(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');

	void SolveP1Ax(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');
	void SolveP1Ay(DTYPE* dst, DTYPE* src, int2 dim, char bc = 'd');
	static CuRestriction2D* GetInstance();
private:
	static std::auto_ptr<CuRestriction2D> instance_;
};

#endif