#ifndef __CUPROLONGATION2D_CUH__
#define __CUPROLONGATION2D_CUH__

#include "cudaGlobal.cuh"

enum class ProlongationType {
	PT_2NM1,
	PT_2NP1,
	PT_2N,
	PT_2N_C4,
	PT_2NM1_RN,
	PT_OTHER
};

class CuProlongation2D
{
public:
	CuProlongation2D();
	~CuProlongation2D();
	void Solve(DTYPE* dst, DTYPE* src, int2 dim, ProlongationType rs_type = ProlongationType::PT_2NM1);
	void SolveAdd(DTYPE* dst, DTYPE* src, int2 dim, ProlongationType rs_type = ProlongationType::PT_2NM1, char bc = 'd');
	void SolveRNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc = 'd');
	void Solve2NM1RNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc = 'd');
	void SolveP1RNAdd(DTYPE* dst, DTYPE* src, int2 dim_dst, int2 dim, char bc = 'd');
	static CuProlongation2D* GetInstance();
private:
	static std::auto_ptr<CuProlongation2D> instance_;
};

#endif