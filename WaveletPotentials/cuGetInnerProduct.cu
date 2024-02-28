#include "cuGetInnerProduct.cuh"
#include "cuMemoryManager.cuh"
#include "cuWHHDForward.cuh"
#include "cudaMath.cuh"

CuGetInnerProduct::CuGetInnerProduct() = default;
CuGetInnerProduct::~CuGetInnerProduct() = default;

std::auto_ptr<CuGetInnerProduct> CuGetInnerProduct::instance_;

__global__ void cuSelectIpPart(DTYPE * dst, DTYPE * src, int dim)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < dim)
	{
		int idx_src = dim * idx + idx;
		int idx_dst = idx;
		dst[idx] = src[idx_src];
		idx_dst += dim;
		if (idx > 1)
		{
			dst[idx_dst] = src[idx_src - 2];
		}
		else
		{
			dst[idx_dst] = 0.f;
		}

		idx_dst += dim;
		if (idx > 0)
		{
			dst[idx_dst] = src[idx_src - 1];
		}
		else
		{
			dst[idx_dst] = 0.f;
		}

		idx_dst += dim;
		if (idx < dim - 1)
		{
			dst[idx_dst] = src[idx_src + 1];
		}
		else
		{
			dst[idx_dst] = 0;
		}

		idx_dst += dim;
		if (idx < dim - 2)
		{
			dst[idx_dst] = src[idx_src + 2];
		}
		else
		{
			dst[idx_dst] = 0.f;
		}

		int ext = dim & 1;
		int nei_idx = ((idx - ext) >> 1) + ext;
		idx_dst += dim;
		idx_src = dim * idx + nei_idx;
		
		if (nei_idx > 0)
		{
			dst[idx_dst] = src[idx_src - 1];
			//printf("idx: %d, nei_idx: %d, src: %f\n", idx, nei_idx, src[idx_src - 1]);
		}

		idx_dst += dim;
		dst[idx_dst] = src[idx_src];

		idx_dst += dim;
		dst[idx_dst] = src[idx_src + 1];
	}
}

CuGetInnerProduct* CuGetInnerProduct::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuGetInnerProduct>(new CuGetInnerProduct); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

DTYPE* CuGetInnerProduct::GetIp(std::string str, int dim)
{
	std::string dstr(str + std::to_string(dim));
	return ips_map_.find(dstr) == ips_map_.end() ? nullptr : ips_map_[dstr];
}

DTYPE* CuGetInnerProduct::GenerateIp(std::string str, int dim)
{
	int size = dim * dim;
	std::string dstr(str + std::to_string(dim));

	if (ips_map_.find(dstr) != ips_map_.end())
	{
		return ips_map_[dstr];
	}

	DTYPE* data = CuMemoryManager::GetInstance()->GetData(dstr, dim * 8);
	DTYPE* tmp = CuMemoryManager::GetInstance()->GetData("getip_tmp", size);
	DTYPE* tmp_t = CuMemoryManager::GetInstance()->GetData("getip_tmp_t", size);
	DTYPE* tmp_mp = CuMemoryManager::GetInstance()->GetData("getip_tmp_mp", size);

	std::unique_ptr<DTYPE[]> host_v(new DTYPE[size]);
	memset(host_v.get(), 0, sizeof(DTYPE) * size);
	for (int i = 0; i < dim; i++)
	{
		host_v[i * dim + i] = 1;
	}
	int level = std::log2(dim);
	checkCudaErrors(cudaMemcpy(tmp, host_v.get(), sizeof(DTYPE) * size, cudaMemcpyHostToDevice));
	CuWHHDForward::GetInstance()->Solve1D(tmp, tmp, { dim, dim, 1 }, level, 'y', str[0], type_from_string(str), false);
	//CudaPrintfMat(tmp, size);
	//CudaPrintfMat(tmp, { dim, dim });
	CudaMatrixTranspose(tmp_t, tmp, { dim, dim });
	CudaMultiplySquare(tmp_mp, tmp_t, tmp, dim);
	cuSelectIpPart << <BLOCKS(dim), THREADS(dim) >> > (data, tmp_mp, dim);

	ips_map_[dstr] = data;
	CuMemoryManager::GetInstance()->FreeData("getip_tmp");
	CuMemoryManager::GetInstance()->FreeData("getip_tmp_t");
	return data;
}

DTYPE* CuGetInnerProduct::GenerateProjCoef(int& length, char direction)
{
	std::string dstr("Data\\coef_template_");
	dstr += direction;
	dstr += "v.csv";
	std::ifstream fp_input(dstr.c_str());
	std::vector<DTYPE> user_arr;
	int user_dim[3] = { 0, 0, 1 };
	ReadCsv(user_arr, fp_input, user_dim);

	int size_pc = user_dim[0] * user_dim[1];
	DTYPE* data = CuMemoryManager::GetInstance()->GetData(dstr, size_pc);
	checkCudaErrors(cudaMemcpy(data, user_arr.data(), sizeof(DTYPE) * size_pc, cudaMemcpyHostToDevice));
	length = user_dim[1];
	return data;
}

DTYPE* CuGetInnerProduct::GenerateProjQCoef(int& length, char direction)
{
	std::string dstr("Data\\coef_template_q_");
	dstr += direction;
	dstr += "v.csv";
	std::ifstream fp_input(dstr.c_str());
	std::vector<DTYPE> user_arr;
	int user_dim[3] = { 0, 0, 1 };
	ReadCsv(user_arr, fp_input, user_dim);

	int size_pc = user_dim[0] * user_dim[1];
	DTYPE* data = CuMemoryManager::GetInstance()->GetData(dstr, size_pc);
	checkCudaErrors(cudaMemcpy(data, user_arr.data(), sizeof(DTYPE) * size_pc, cudaMemcpyHostToDevice));
	length = user_dim[1];
	return data;
}

DTYPE* CuGetInnerProduct::GetIp(WaveletType type, char bc, int dim)
{
	return GetIp(string_from_type(type, bc), dim);
}

DTYPE* CuGetInnerProduct::GenerateIp(WaveletType type, char bc, int dim)
{
	return GenerateIp(string_from_type(type, bc), dim);
}