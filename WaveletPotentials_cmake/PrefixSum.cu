#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <iostream>
#include "PrefixSum.cuh"

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void prescan_arbitrary(int* output, int* input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_arbitrary_unoptimized(int* output, int* input, int n, int powerOfTwo) 
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}

__global__ void prescan_large(int* output, int* input, int n, int* sums) 
{
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_large_unoptimized(int* output, int* input, int n, int* sums) 
{
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

__global__ void add(int* output, int length, int* n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int* output, int length, int* n1, int* n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

PrefixSum::PrefixSum(int* data_out, int* data_in, int size) :
	size_(size), data_out_(data_out), data_(data_in), is_data_free_(false), is_data_out_free_(false), THREADS_PER_BLOCK(512), ELEMENTS_PER_BLOCK(1024), total_size_(size)
{
	int blocks = size / ELEMENTS_PER_BLOCK;
	block_num_[0] = blocks;

	if (blocks > ELEMENTS_PER_BLOCK)
	{
		blocks /= ELEMENTS_PER_BLOCK;
		block_num_[1] = blocks;
	}
	else
	{
		block_num_[1] = 0;
	}

	cudaError_t cudaStatus;
	if (data_ == nullptr)
	{
		cudaStatus = cudaMalloc((void**)&data_, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		is_data_free_ = true;
	}

	if (data_out_ == nullptr)
	{
		cudaStatus = cudaMalloc((void**)&data_out_, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		is_data_out_free_ = true;
	}

	for (int i = 0; i < 2; i++)
	{
		if (block_num_[i] > 0)
		{
			cudaStatus = cudaMalloc((void**)&incr_[i], block_num_[i] * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc incr_ failed!");
			}

			cudaStatus = cudaMalloc((void**)&sums_[i], block_num_[i] * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc sums_ failed!");
			}
		}
		else
		{
			incr_[i] = nullptr;
			sums_[i] = nullptr;
		}
	}
}

PrefixSum::~PrefixSum()
{
	if (is_data_free_)
	{
		cudaFree(data_);
	}

	if (is_data_out_free_)
	{
		cudaFree(data_out_);
	}

	for (int i = 0; i < 2; i++)
	{
		if (incr_[i] != nullptr)
		{
			cudaFree(incr_[i]);
			cudaFree(sums_[i]);
		}
	}
}

int PrefixSum::Resize(int n[3])
{
	int size = n[0] * n[1] * n[2];
	size_ = size;
	int blocks = size / ELEMENTS_PER_BLOCK;
	int new_blocks[2];
	new_blocks[0] = blocks;
	cudaError_t cudaStatus;

	if (blocks > ELEMENTS_PER_BLOCK)
	{
		blocks /= ELEMENTS_PER_BLOCK;
		new_blocks[1] = blocks;
	}
	else
	{
		new_blocks[1] = 0;
	}

	for (int i = 0; i < 2; i++)
	{
		if (new_blocks[i] > block_num_[i])
		{
			printf("in prefixsum resize\n");
			block_num_[i] = new_blocks[i];
			if (incr_[i] != nullptr)
			{
				cudaFree(incr_[i]);
				cudaFree(sums_[i]);
			}
			cudaStatus = cudaMalloc((void**)&incr_[i], block_num_[i] * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc incr_ failed!");
				return 1;
			}

			cudaStatus = cudaMalloc((void**)&sums_[i], block_num_[i] * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc sums_ failed!");
				return 1;
			}
		}
	}

	return 0;
}

void PrefixSum::scanSmallDeviceArray(int* d_out, int* d_in, int length)
{
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> > (d_out, d_in, length, powerOfTwo);
}

void PrefixSum::scanLargeEvenDeviceArray(int* d_out, int* d_in, int length, int level)
{
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int* d_sums = sums_[level];
	int* d_incr = incr_[level];

	prescan_large << <blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize >> > (d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		scanLargeDeviceArray(d_incr, d_sums, blocks, level + 1);
	}
	else {
		scanSmallDeviceArray(d_incr, d_sums, blocks);
	}

	add << <blocks, ELEMENTS_PER_BLOCK >> > (d_out, ELEMENTS_PER_BLOCK, d_incr);
}

void PrefixSum::scanLargeDeviceArray(int* d_out, int* d_in, int length, int level) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, level);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, level);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int* startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

		add << <1, remainder >> > (startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

int PrefixSum::Do()
{
	if (size_ > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(data_out_, data_, size_, 0);
	}
	else {
		scanSmallDeviceArray(data_out_, data_, size_);
	}

	return 0;
}


int PrefixSum::Do(int size)
{
	if (size > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(data_out_, data_, size, 0);
	}
	else {
		scanSmallDeviceArray(data_out_, data_, size);
	}

	return 0;
}