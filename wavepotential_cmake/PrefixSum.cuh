#ifndef __PREFIX_SUM_CUH__
#define __PREFIX_SUM_CUH__

#include "Utils.h"

class PrefixSum
{
public:
	PrefixSum(int* data_out = nullptr, int* data = nullptr, int size = 1);
	~PrefixSum();

	int Do();
	int Do(int size);
	int Resize(int n[3]);
	int*& data = data_;

private:
	void scanSmallDeviceArray(int* d_out, int* d_in, int length);
	void scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, int level);
	void scanLargeDeviceArray(int* d_out, int* d_in, int length, int level);

	int* data_;
	int* data_out_;

	bool is_data_free_;
	bool is_data_out_free_;

	int* incr_[2];
	int* sums_[2];

	int block_num_[2];

	int size_;
	int total_size_;

	int THREADS_PER_BLOCK;
	int ELEMENTS_PER_BLOCK;
};

#endif