#include "cuWeightedjacobi3D.cuh"
#include "cuMemoryManager.cuh"
#include "cudaMath.cuh"
#include "Interpolation.cuh"

CuWeightedJacobi3D::CuWeightedJacobi3D() = default;
CuWeightedJacobi3D::~CuWeightedJacobi3D() = default;

std::auto_ptr<CuWeightedJacobi3D> CuWeightedJacobi3D::instance_;

CuWeightedJacobi3D* CuWeightedJacobi3D::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuWeightedJacobi3D>(new CuWeightedJacobi3D); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

__global__ void cuWeightedJacobi3D_d_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
	}
}

__global__ void cuWeightedJacobi3D_n_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_z_p2(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx & (dim.y - 1);
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz & (dim.x - 1);
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// left
		if (idx_z > 0)
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// right
		if (idx_z < dim.z - 1)
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//if (max_size < 5)
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_d_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
	}
}

__global__ void cuWeightedJacobi3D_n_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= dxp2_inv.y;
		else
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x == 0)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= dxp2_inv.x;
		else
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// front
		if (idx_z == 0)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= dxp2_inv.z;
		else
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_z_common(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE a_diag = 2.f * dxp2_inv.x + 2.f * dxp2_inv.y + 2.f * dxp2_inv.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * dxp2_inv.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * dxp2_inv.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * dxp2_inv.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * dxp2_inv.x;

		// left
		if (idx_z > 0)
			off_d += -src[idx - slice_xy] * dxp2_inv.z;

		// right
		if (idx_z < dim.z - 1)
			off_d += -src[idx + slice_xy] * dxp2_inv.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//if (max_size < 5)
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_d_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag += coefs.y;
		else
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag += coefs.y;
		else
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x == 0)
			a_diag += coefs.x;
		else
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag += coefs.x;
		else
			off_d += -src[idx + dim.y] * coefs.x;

		// front
		if (idx_z == 0)
			a_diag += coefs.z;
		else
			off_d += -src[idx - slice_xy] * coefs.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag += coefs.z;
		else
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//if (max_size < 18)
		//	printf("idx: %d, %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx_x, idx_y, idx_z, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_n_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y == 0)
			a_diag -= coefs.y;
		else
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y == dim.y - 1)
			a_diag -= coefs.y;
		else
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x == 0)
			a_diag -= coefs.x;
		else
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x == dim.x - 1)
			a_diag -= coefs.x;
		else
			off_d += -src[idx + dim.y] * coefs.x;

		// front
		if (idx_z == 0)
			a_diag -= coefs.z;
		else
			off_d += -src[idx - slice_xy] * coefs.z;

		// back
		if (idx_z == dim.z - 1)
			a_diag -= coefs.z;
		else
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWeightedJacobi3D_z_p1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dxp2_inv, DTYPE weight, DTYPE lpx, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		DTYPE3 coefs = dxp2_inv;
		//coefs.x *= idx_x == dim.x - 1 ? lpx : 1.f;
		//coefs.y *= idx_y == dim.y - 1 ? lpx : 1.f;
		//coefs.z *= idx_z == dim.z - 1 ? lpx : 1.f;
		//if (max_size < 18)
		//	printf("lpx: %f, idx: %d, %d, %d, coefs: %f, %f %f, lpx: %f\n", lpx, idx_x, idx_y, idx_z, coefs.x, coefs.y, coefs.z, lpx);
		if (idx_x == dim.x - 1)
		{
			coefs.y *= lpx;
			coefs.z *= lpx;
		}
		if (idx_y == dim.y - 1)
		{
			coefs.x *= lpx;
			coefs.z *= lpx;
		}
		if (idx_z == dim.z - 1)
		{
			coefs.x *= lpx;
			coefs.y *= lpx;
		}

		DTYPE a_diag = 2.f * coefs.x + 2.f * coefs.y + 2.f * coefs.z;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
			off_d += -src[idx - 1] * coefs.y;

		// top
		if (idx_y < dim.y - 1)
			off_d += -src[idx + 1] * coefs.y;

		// left
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * coefs.x;

		// right
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * coefs.x;

		// left
		if (idx_z > 0)
			off_d += -src[idx - slice_xy] * coefs.z;

		// right
		if (idx_z < dim.z - 1)
			off_d += -src[idx + slice_xy] * coefs.z;

		dst[idx] = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		//if (max_size < 18)
		////if (max_size > 18 && max_size < 50)
		////if (max_size > 50)
		//	printf("lpx: %f, idx: %d, %d, %d, coefs: %f, %f %f, a_diag: %f\n", lpx, idx_x, idx_y, idx_z, coefs.x, coefs.y, coefs.z, a_diag);
	}
}

__global__ void cuWjFrac3D_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] * dxp2_inv.y;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] * dxp2_inv.x;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] * dxp2_inv.z;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] * dxp2_inv.z;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_n_dy(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}
		else
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += 2.f * cf;
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;

			//printf("idx: %d, %d, %d, a_diag: %f, cf: %f, cf_val: %f\n", idx_x, idx_y, idx_z, a_diag, cf, src[idx - dim.y]);
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		//printf("idx: %d, %d, %d, a_diag: %f, cf: %f\n", idx_x, idx_y, idx_z, a_diag, off_d);
		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE3 dxp2_inv, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE cf = Ay[idx_ay] * dxp2_inv.y;
		if (idx_y > 0)
		{
			off_d += -cf * src[idx - 1];
		}
		a_diag += cf;


		// top
		cf = Ay[idx_ay + 1] * dxp2_inv.y;
		if (idx_y < dim.y - 1)
		{
			off_d += -cf * src[idx + 1];
		}
		a_diag += cf;

		// left
		cf = Ax[idx_ax] * dxp2_inv.x;
		if (idx_x > 0)
		{
			off_d += -src[idx - dim.y] * cf;
		}
		a_diag += cf;

		// right
		cf = Ax[idx_ax + dim.y] * dxp2_inv.x;
		if (idx_x < dim.x - 1)
		{
			off_d += -src[idx + dim.y] * cf;
		}
		a_diag += cf;

		// front
		cf = Az[idx_az] * dxp2_inv.z;
		if (idx_z > 0)
		{
			off_d += -src[idx - slice_xy] * cf;
		}
		a_diag += cf;

		// back
		cf = Az[idx_az + slice_xy] * dxp2_inv.z;
		if (idx_z < dim.z - 1)
		{
			off_d += -src[idx + slice_xy] * cf;
		}
		a_diag += cf;

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
		{
			off_d += -cf * src[idx - 1];
		}


		// top
		dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
		{
			off_d += -cf * src[idx + 1];
		}

		// left
		DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		// front
		DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
		cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z > 0)
		{
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
		cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z < dim.z - 1)
		{
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

void CuWeightedJacobi3D::Solve(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dx, int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE3 dx_inv = make_DTYPE3(1.f / dx.x, 1.f / dx.y, 1.f / dx.z);
	DTYPE3 dxp2_inv = make_DTYPE3(dx_inv.x * dx_inv.x, dx_inv.y * dx_inv.y, dx_inv.z * dx_inv.z);
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			//cuWeightedJacobi3D_d_p2 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			cuWeightedJacobi3D_d_common << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			break;
		case 'z':
			//cuWeightedJacobi3D_z_p2 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			cuWeightedJacobi3D_z_common << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			break;
		default:
			//cuWeightedJacobi3D_n_p2 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
			cuWeightedJacobi3D_n_common << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi3D::SolveP1(DTYPE* dst, DTYPE* src, DTYPE* f, int3 dim, DTYPE3 dx, int n_iters,
	DTYPE weight, DTYPE lpx, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE3 dx_inv = make_DTYPE3(1.f / dx.x, 1.f / dx.y, 1.f / dx.z);
	DTYPE3 dxp2_inv = make_DTYPE3(dx_inv.x * dx_inv.x, dx_inv.y * dx_inv.y, dx_inv.z * dx_inv.z);
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			cuWeightedJacobi3D_d_p1 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, lpx, max_size);
			break;
		case 'z':
			cuWeightedJacobi3D_z_p1 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, lpx, max_size);
			break;
		default:
			cuWeightedJacobi3D_n_p1 << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, dim, dxp2_inv, weight, lpx, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi3D::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE3 dx,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE3 dxp2_inv = { 1.f / dx.x / dx.x, 1.f / dx.y / dx.y, 1.f / dx.z / dx.z };

	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			break;
		case 'z':
			cuWjFrac3D_z << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dxp2_inv, weight, max_size);
			break;
		default:
			cuWjFrac3D_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dxp2_inv, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi3D::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			break;
		case 'z':
			break;
		default:
			cuWjFrac3D_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

void CuWeightedJacobi3D::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			break;
		case 'z':
			cuWjFrac3D_z << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, dx_e, weight, max_size);
			break;
		default:
			cuWjFrac3D_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

__global__ void cuWjGetFracRes_3d_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}

		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWjGetFracRes_3d_n_dy(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}
		else
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += 2.f * cf;
			//off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWjGetFracRes_3d_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
		{
			off_d += -cf * src[idx - 1];
		}


		// top
		dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
		{
			off_d += -cf * src[idx + 1];
		}

		// left
		DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		// front
		DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
		cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z > 0)
		{
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
		cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z < dim.z - 1)
		{
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

DTYPE CuWeightedJacobi3D::GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);
}

void CuWeightedJacobi3D::GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

DTYPE CuWeightedJacobi3D::GetFracRhs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);

	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi3D::GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

/********************************************************************************************************************/
__device__ void cuWj_3d_n(DTYPE* res, DTYPE* x, DTYPE* f, int3 dim, DTYPE ax_l, DTYPE ax_r, DTYPE ay_b, DTYPE ay_t, 
	DTYPE az_f, DTYPE az_b, DTYPE weight, int3 idx3d, int idx)
{
	DTYPE off_d = 0.f;
	DTYPE a_diag = 0.f;
	int slice = dim.x * dim.y;
	if (idx3d.x > 0)
	{
		a_diag += ax_l;
		off_d += -ax_l * x[idx - dim.y];
	}
	if (idx3d.x < dim.x - 1)
	{
		a_diag += ax_r;
		off_d += -ax_r * x[idx + dim.y];
	}
	if (idx3d.y > 0)
	{
		a_diag += ay_b;
		off_d += -ay_b * x[idx - 1];
	}
	if (idx3d.y < dim.y - 1)
	{
		a_diag += ay_t;
		off_d += -ay_t * x[idx + 1];
	}
	if (idx3d.z > 0)
	{
		a_diag += az_f;
		off_d += -az_f * x[idx - slice];
	}
	if (idx3d.z < dim.z - 1)
	{
		a_diag += az_b;
		off_d += -az_b * x[idx + slice];
	}
	DTYPE val = 0.f;
	if (a_diag > 0.f)
	{
		val = (1 - weight) * x[idx] + weight * (f[idx] - off_d) / a_diag;
	}

	res[idx] = val;
}

__global__ void cuWjFrac3D_kernel_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size, int block_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ DTYPE r[];
	DTYPE* p = &r[block_size];
	DTYPE* z = &p[block_size];

	int idx_xz = idx / dim.y;
	int idx_y = idx - idx_xz * dim.y;
	int idx_z = idx_xz / dim.x;
	int idx_x = idx_xz - dim.x * idx_z;
	int slice = dim.x * dim.y;

	int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
	int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
	int idx_az = idx;

	int3 idx3d = { idx_x, idx_y, idx_z };
	DTYPE res = 0.f;

	DTYPE dxdydz;

	DTYPE ax_l;
	DTYPE ax_r;
	DTYPE ay_b;
	DTYPE ay_t;
	DTYPE az_f;
	DTYPE az_b;

	if (idx < max_size)
	{
		dxdydz = dx[idx_x] * dy[idx_y] * dz[idx_z];
		DTYPE dydz = dy[idx_y] * dz[idx_z];
		DTYPE dxdy = dx[idx_x] * dy[idx_y];
		DTYPE dxdz = dx[idx_x] * dz[idx_z];
		ax_l = Ax[idx_ax] / (dx[max(idx_x - 1, 0)] + dx[idx_x]) * dydz * 2.f;
		ax_r = Ax[idx_ax + dim.y] / (dx[min(idx_x + 1, dim.x - 1)] + dx[idx_x]) * dydz * 2.f;
		ay_b = Ay[idx_ay] / (dy[max(idx_y - 1, 0)] + dy[idx_y]) * dxdz * 2.f;
		ay_t = Ay[idx_ay + 1] / (dy[min(idx_y + 1, dim.y - 1)] + dy[idx_y]) * dxdz * 2.f;
		az_f = Az[idx_az] / (dz[max(idx_z - 1, 0)] + dz[idx_z]) * dxdy * 2.f;
		az_b = Az[idx_az + slice] / (dz[min(idx_z + 1, dim.z - 1)] + dz[idx_z]) * dxdy * 2.f;
		p[idx] = src[idx];
		r[idx] = f[idx] * dxdydz;
	}
	else
	{
		r[idx] = 0.f;
		p[idx] = 0.f;
	}
	z[idx] = 0.f;

	__syncthreads();

	//for (int i = 0; i < CG_MAX_ITERS; i++)
#pragma unroll
	for (int i = 0; i < 50; i++)
	{
		if (idx < max_size)
			cuWj_3d_n(z, p, r, dim, ax_l, ax_r, ay_b, ay_t, az_f, az_b, weight, idx3d, idx);

		__syncthreads();

		p[idx] = z[idx];
		__syncthreads();
	}

	res = p[idx];
	if (idx < max_size)
	{
		dst[idx] = res;
	}
}

__device__ void cuWj_3d_z(DTYPE* res, DTYPE* x, DTYPE* f, int3 dim, DTYPE ax_l, DTYPE ax_r, DTYPE ay_b, DTYPE ay_t,
	DTYPE az_f, DTYPE az_b, DTYPE weight, int3 idx3d, int idx)
{
	DTYPE off_d = 0.f;
	DTYPE a_diag = 0.f;
	int slice = dim.x * dim.y;
	if (idx3d.x > 0)
	{
		off_d += -ax_l * x[idx - dim.y];
	}
	a_diag += ax_l;
	if (idx3d.x < dim.x - 1)
	{
		off_d += -ax_r * x[idx + dim.y];
	}
	a_diag += ax_r;
	if (idx3d.y > 0)
	{
		off_d += -ay_b * x[idx - 1];
	}
	a_diag += ay_b;
	if (idx3d.y < dim.y - 1)
	{
		off_d += -ay_t * x[idx + 1];
	}
	a_diag += ay_t;
	if (idx3d.z > 0)
	{
		off_d += -az_f * x[idx - slice];
	}
	a_diag += az_f;
	if (idx3d.z < dim.z - 1)
	{
		off_d += -az_b * x[idx + slice];
	}
	a_diag += az_b;
	DTYPE val = 0.f;
	if (a_diag > 0.f)
	{
		val = (1 - weight) * x[idx] + weight * (f[idx] - off_d) / a_diag;
	}

	res[idx] = val;
}

__global__ void cuWjFrac3D_kernel_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE weight, int max_size, int block_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ DTYPE r[];
	DTYPE* p = &r[block_size];
	DTYPE* z = &p[block_size];

	int idx_xz = idx / dim.y;
	int idx_y = idx - idx_xz * dim.y;
	int idx_z = idx_xz / dim.x;
	int idx_x = idx_xz - dim.x * idx_z;
	int slice = dim.x * dim.y;

	int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
	int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
	int idx_az = idx;

	int3 idx3d = { idx_x, idx_y, idx_z };
	DTYPE res = 0.f;

	DTYPE dxdydz;

	DTYPE ax_l;
	DTYPE ax_r;
	DTYPE ay_b;
	DTYPE ay_t;
	DTYPE az_f;
	DTYPE az_b;

	if (idx < max_size)
	{
		dxdydz = dx[idx_x] * dy[idx_y] * dz[idx_z];
		DTYPE dydz = dy[idx_y] * dz[idx_z];
		DTYPE dxdy = dx[idx_x] * dy[idx_y];
		DTYPE dxdz = dx[idx_x] * dz[idx_z];

		DTYPE dx_l = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		DTYPE dx_r = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		DTYPE dy_b = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE dy_t = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		DTYPE dz_f = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
		DTYPE dz_b = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;

		ax_l = Ax[idx_ax] / (dx_l + dx[idx_x]) * dydz * 2.f;
		ax_r = Ax[idx_ax + dim.y] / (dx_r + dx[idx_x]) * dydz * 2.f;
		ay_b = Ay[idx_ay] / (dy_b + dy[idx_y]) * dxdz * 2.f;
		ay_t = Ay[idx_ay + 1] / (dy_t + dy[idx_y]) * dxdz * 2.f;
		az_f = Az[idx_az] / (dz_f + dz[idx_z]) * dxdy * 2.f;
		az_b = Az[idx_az + slice] / (dz_b + dz[idx_z]) * dxdy * 2.f;
		p[idx] = src[idx];
		r[idx] = f[idx] * dxdydz;
	}
	else
	{
		r[idx] = 0.f;
		p[idx] = 0.f;
	}
	z[idx] = 0.f;

	__syncthreads();

	//for (int i = 0; i < CG_MAX_ITERS; i++)
#pragma unroll
	for (int i = 0; i < 20; i++)
	{
		if (idx < max_size)
			cuWj_3d_z(z, p, r, dim, ax_l, ax_r, ay_b, ay_t, az_f, az_b, weight, idx3d, idx);

		__syncthreads();

		p[idx] = z[idx];
		__syncthreads();
	}

	res = p[idx];
	if (idx < max_size)
	{
		dst[idx] = res;
	}
}

void CuWeightedJacobi3D::SolveFracKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	int sm_mem = 3 * NUM_THREADS * sizeof(DTYPE);

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjFrac3D_kernel_n << <BLOCKS(max_size), THREADS(max_size), sm_mem >> > (dst, src, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size, NUM_THREADS);
	}
}

void CuWeightedJacobi3D::SolveFracKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz,
	DTYPE3 dx_e, int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	int sm_mem = 3 * NUM_THREADS * sizeof(DTYPE);

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		cuWjFrac3D_kernel_z << <BLOCKS(max_size), THREADS(max_size), sm_mem >> > (dst, src, f, Ax, Ay, Az, dim, dx, dy, dz, dx_e, weight, max_size, NUM_THREADS);
		break;
	default:
		cuWjFrac3D_kernel_n << <BLOCKS(max_size), THREADS(max_size), sm_mem >> > (dst, src, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size, NUM_THREADS);
	}
}

/****************************************************Interior Dirichlet**************************************************************/
__global__ void cuWjFrac3D_int_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
			DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y > 0)
			{
				off_d += -cf * src[idx - 1] * is_interbound[idx - 1];
			}


			// top
			dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
			cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y < dim.y - 1)
			{
				off_d += -cf * src[idx + 1] * is_interbound[idx + 1];
			}

			// left
			DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
			cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x > 0)
				off_d += -src[idx - dim.y] * cf * is_interbound[idx - dim.y];

			// right
			dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
			cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x < dim.x - 1)
				off_d += -src[idx + dim.y] * cf * is_interbound[idx + dim.y];

			// front
			DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
			cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z > 0)
			{
				off_d += -src[idx - slice_xy] * cf * is_interbound[idx - slice_xy];
			}

			// back
			dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
			cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z < dim.z - 1)
			{
				off_d += -src[idx + slice_xy] * cf * is_interbound[idx + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjFrac3D_int_n_test(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		// inner boundary
		{
			a_diag -= (Ay[idx_ay + 1] - Ay[idx_ay]) / dy[idx_y] / dy[idx_y] * 2.f +
				(Ax[idx_ax + dim.y] - Ax[idx_ax]) / dx[idx_x] / dx[idx_x] * 2.f +
				(Az[idx_az + slice_xy] - Az[idx_az]) / dz[idx_z] / dz[idx_z] * 2.f;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_int_n_old2(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf;
				cf *= Ay[idx_ay];
				off_d += -cf * src[idx - 1];
			}


			// top

			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf;
				cf *= Ay[idx_ay + 1];
				off_d += -cf * src[idx + 1];
			}

			// left

			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf;
				cf *= Ax[idx_ax];
				off_d += -src[idx - dim.y] * cf;
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf;
				cf *= Ax[idx_ax + dim.y];
				off_d += -src[idx + dim.y] * cf;
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf;
				cf *= Az[idx_az];
				off_d += -src[idx - slice_xy] * cf;
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf;
				cf *= Az[idx_az + slice_xy];
				off_d += -src[idx + slice_xy] * cf;
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjFrac3D_int_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;
			DTYPE d_cf = 2.f;
			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay] + d_cf * (1.f - Ay[idx_ay]));
				off_d += -cf_base * Ay[idx_ay] * src[idx - 1];
			}

			// top
			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay + 1] + d_cf * (1.f - Ay[idx_ay + 1]));
				off_d += -cf_base * Ay[idx_ay + 1] * src[idx + 1];
			}

			// left
			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax] + d_cf * (1.f - Ax[idx_ax]));
				off_d += -src[idx - dim.y] * Ax[idx_ax] * cf_base;
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax + dim.y] + d_cf * (1.f - Ax[idx_ax + dim.y]));
				off_d += -src[idx + dim.y] * cf_base * Ax[idx_ax + dim.y];
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az] + d_cf * (1.f - Az[idx_az]));
				off_d += -src[idx - slice_xy] * cf_base * Az[idx_az];
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az + slice_xy] + d_cf * (1.f - Az[idx_az + slice_xy]));
				off_d += -src[idx + slice_xy] * cf_base * Az[idx_az + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjFrac3D_int_n_old(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf;
				off_d += -cf * src[idx - 1] * is_interbound[idx - 1];
			}


			// top

			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf;
				off_d += -cf * src[idx + 1] * is_interbound[idx + 1];
			}

			// left

			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf;
				off_d += -src[idx - dim.y] * cf * is_interbound[idx - dim.y];
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf;
				off_d += -src[idx + dim.y] * cf * is_interbound[idx + dim.y];
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf;
				off_d += -src[idx - slice_xy] * cf * is_interbound[idx - slice_xy];
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf;
				off_d += -src[idx + slice_xy] * cf * is_interbound[idx + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjGetInteriorBound(DTYPE* is_bound, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * dim.y * (dim.x + 1);
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * (dim.y + 1) * dim.x;
		int idx_az = idx_y + idx_x * dim.y + idx_z * dim.x * dim.y;

		//if (ax[idx_ax] < eps<DTYPE> || ax[idx_ax + dim.y] < eps<DTYPE> ||
		//	ay[idx_ay] < eps<DTYPE> || ay[idx_ay + 1] < eps<DTYPE> ||
		//	az[idx_az] < eps<DTYPE> || az[idx_az + dim.x * dim.y] < eps<DTYPE>)
		if (ax[idx_ax] < eps<DTYPE> && ax[idx_ax + dim.y] < eps<DTYPE> &&
			ay[idx_ay] < eps<DTYPE> && ay[idx_ay + 1] < eps<DTYPE> &&
			az[idx_az] < eps<DTYPE> && az[idx_az + dim.x * dim.y] < eps<DTYPE>)
		{
			is_bound[idx] = 0.f;
		}
		else
		{
			is_bound[idx] = 1.f;
		}
	}
}

__global__ void cuWjGetInteriorBound_any(DTYPE* is_bound, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * dim.y * (dim.x + 1);
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * (dim.y + 1) * dim.x;
		int idx_az = idx_y + idx_x * dim.y + idx_z * dim.x * dim.y;

		if (ax[idx_ax] < eps<DTYPE> || ax[idx_ax + dim.y] < eps<DTYPE> ||
			ay[idx_ay] < eps<DTYPE> || ay[idx_ay + 1] < eps<DTYPE> ||
			az[idx_az] < eps<DTYPE> || az[idx_az + dim.x * dim.y] < eps<DTYPE>)
		//if (ax[idx_ax] < eps<DTYPE> && ax[idx_ax + dim.y] < eps<DTYPE> &&
		//	ay[idx_ay] < eps<DTYPE> && ay[idx_ay + 1] < eps<DTYPE> &&
		//	az[idx_az] < eps<DTYPE> && az[idx_az + dim.x * dim.y] < eps<DTYPE>)
		{
			is_bound[idx] = 0.f;
		}
		else
		{
			is_bound[idx] = 1.f;
		}
	}
}

__global__ void cuWjGetFracRes_3d_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
			DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y > 0)
			{
				off_d += -cf * src[idx - 1] * is_interbound[idx - 1];
			}


			// top
			dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
			cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y < dim.y - 1)
			{
				off_d += -cf * src[idx + 1] * is_interbound[idx + 1];
			}

			// left
			DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
			cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x > 0)
				off_d += -src[idx - dim.y] * cf * is_interbound[idx - dim.y];

			// right
			dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
			cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x < dim.x - 1)
				off_d += -src[idx + dim.y] * cf * is_interbound[idx + dim.y];

			// front
			DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
			cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z > 0)
			{
				off_d += -src[idx - slice_xy] * cf * is_interbound[idx - slice_xy];
			}

			// back
			dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
			cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z < dim.z - 1)
			{
				off_d += -src[idx + slice_xy] * cf * is_interbound[idx + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = f[idx] - (a_diag * src[idx] + off_d);
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWjGetFracRes_3d_n_old(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		// inner boundary
		{
			a_diag -= (Ay[idx_ay + 1] - Ay[idx_ay]) / dy[idx_y] / dy[idx_y] * 2.f +
				(Ax[idx_ax + dim.y] - Ax[idx_ax]) / dx[idx_x] / dx[idx_x] * 2.f +
				(Az[idx_az + slice_xy] - Az[idx_az]) / dz[idx_z] / dz[idx_z] * 2.f;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
	}
}

__global__ void cuWjGetFracRes_3d_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;
			DTYPE d_cf = 2.f;

			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay] + d_cf * (1.f - Ay[idx_ay]));
				off_d += -cf_base * Ay[idx_ay] * src[idx - 1];
			}

			// top
			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay + 1] + d_cf * (1.f - Ay[idx_ay + 1]));
				off_d += -cf_base * Ay[idx_ay + 1] * src[idx + 1];
			}

			// left
			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax] + d_cf * (1.f - Ax[idx_ax]));
				off_d += -src[idx - dim.y] * Ax[idx_ax] * cf_base;
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax + dim.y] + d_cf * (1.f - Ax[idx_ax + dim.y]));
				off_d += -src[idx + dim.y] * cf_base * Ax[idx_ax + dim.y];
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az] + d_cf * (1.f - Az[idx_az]));
				off_d += -src[idx - slice_xy] * cf_base * Az[idx_az];
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az + slice_xy] + d_cf * (1.f - Az[idx_az + slice_xy]));
				off_d += -src[idx + slice_xy] * cf_base * Az[idx_az + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = f[idx] - (a_diag * src[idx] + off_d);
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

void CuWeightedJacobi3D::SolveFracInt(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
		//CudaPrintfMat(is_bound, dim);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			//CudaPrintfMat(sss, dim);
			cuWjFrac3D_n_dy << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size);
			break;
		case 'z':
			cuWjFrac3D_int_z << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, is_bound, 
				dim, dx, dy, dz, dx_e, weight, max_size);
			break;
		default:
			cuWjFrac3D_int_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, is_bound,
				dim, dx, dy, dz, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

__device__ void cuWj_3d_n(DTYPE* res, DTYPE* x, DTYPE* f, int3 dim, DTYPE a_diag, DTYPE ax_l, DTYPE ax_r, DTYPE ay_b, DTYPE ay_t,
	DTYPE az_f, DTYPE az_b, DTYPE weight, int3 idx3d, int idx)
{
	DTYPE off_d = 0.f;
	int slice = dim.x * dim.y;
	if (idx3d.x > 0)
	{
		off_d += -ax_l * x[idx - dim.y];
	}
	if (idx3d.x < dim.x - 1)
	{
		off_d += -ax_r * x[idx + dim.y];
	}
	if (idx3d.y > 0)
	{
		off_d += -ay_b * x[idx - 1];
	}
	if (idx3d.y < dim.y - 1)
	{
		off_d += -ay_t * x[idx + 1];
	}
	if (idx3d.z > 0)
	{
		off_d += -az_f * x[idx - slice];
	}
	if (idx3d.z < dim.z - 1)
	{
		off_d += -az_b * x[idx + slice];
	}
	DTYPE val = 0.f;
	if (a_diag > 0.f)
	{
		val = (1 - weight) * x[idx] + weight * (f[idx] - off_d) / a_diag;
	}

	res[idx] = val;
}

__global__ void cuWjFrac3D_int_kernel_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size, int block_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ DTYPE r[];
	DTYPE* p = &r[block_size];
	DTYPE* z = &p[block_size];

	int idx_xz = idx / dim.y;
	int idx_y = idx - idx_xz * dim.y;
	int idx_z = idx_xz / dim.x;
	int idx_x = idx_xz - dim.x * idx_z;
	int slice = dim.x * dim.y;

	int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
	int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
	int idx_az = idx;

	int3 idx3d = { idx_x, idx_y, idx_z };
	DTYPE res = 0.f;

	DTYPE dxdydz;

	DTYPE ax_l = 0.f;
	DTYPE ax_r = 0.f;
	DTYPE ay_b = 0.f;
	DTYPE ay_t = 0.f;
	DTYPE az_f = 0.f;
	DTYPE az_b = 0.f;
	DTYPE a_diag = 0.f;

	if (idx < max_size && is_interbound[idx] > 0.5f)
	{
		DTYPE d_cf = 2.f;
		// bottom
		if (idx_y > 0)
		{
			DTYPE dy_nei = dy[idx_y - 1];
			DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf_base * (Ay[idx_ay] + d_cf * (1.f - Ay[idx_ay]));
			ay_b = cf_base * Ay[idx_ay];
		}

		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE dy_nei = dy[idx_y + 1];
			DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf_base * (Ay[idx_ay + 1] + d_cf * (1.f - Ay[idx_ay + 1]));
			ay_t = cf_base * Ay[idx_ay + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE dx_nei = dx[idx_x - 1];
			DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf_base * (Ax[idx_ax] + d_cf * (1.f - Ax[idx_ax]));
			ax_l = Ax[idx_ax] * cf_base;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE dx_nei = dx[idx_x + 1];
			DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf_base * (Ax[idx_ax + dim.y] + d_cf * (1.f - Ax[idx_ax + dim.y]));
			ax_r = cf_base * Ax[idx_ax + dim.y];
		}

		// front
		if (idx_z > 0)
		{
			DTYPE dz_nei = dz[idx_z - 1];
			DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf_base * (Az[idx_az] + d_cf * (1.f - Az[idx_az]));
			az_f = cf_base * Az[idx_az];
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE dz_nei = dz[idx_z + 1];
			DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf_base * (Az[idx_az + slice] + d_cf * (1.f - Az[idx_az + slice]));
			az_b = cf_base * Az[idx_az + slice];
		}

		p[idx] = src[idx];
		r[idx] = f[idx];
	}
	else
	{
		r[idx] = 0.f;
		p[idx] = 0.f;
	}
	z[idx] = 0.f;

#pragma unroll
	for (int i = 0; i < 10; i++)
	{
		if (idx < max_size)
			cuWj_3d_n(z, p, r, dim, a_diag, ax_l, ax_r, ay_b, ay_t, az_f, az_b, weight, idx3d, idx);

		__syncthreads();

		p[idx] = z[idx];
		__syncthreads();
	}

	res = p[idx];
	if (idx < max_size)
	{
		dst[idx] = res;
	}
}

void CuWeightedJacobi3D::SolveFracIntKernel(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
		//CudaPrintfMat(is_bound, dim);
	}

	int sm_mem = 3 * NUM_THREADS * sizeof(DTYPE);

	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			//CudaPrintfMat(sss, dim);
			//cuWjFrac3D_n_dy << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size);
			break;
		case 'z':
			//cuWjFrac3D_int_z << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, is_bound,
			//	dim, dx, dy, dz, dx_e, weight, max_size);
			break;
		default:
			cuWjFrac3D_int_kernel_n << <BLOCKS(max_size), THREADS(max_size), sm_mem >> > (dst, src, f, Ax, Ay, Az, is_bound,
				dim, dx, dy, dz, weight, max_size, NUM_THREADS);
		}
	}
}

DTYPE CuWeightedJacobi3D::GetFracRhsInt(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
	}

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		cuWjGetFracRes_3d_n_dy << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);

	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi3D::GetFracRhsInt(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
	}

	switch (bc)
	{
	case 'd':
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi3D::BoundB(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);

	cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
	CudaMatMul(dst, dst, is_bound, max_size);
}

void CuWeightedJacobi3D::GetBound(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);

	cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (dst, Ax, Ay, Az, dim, max_size);
}

void CuWeightedJacobi3D::GetBound_any(DTYPE* dst, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int3 dim)
{
	int max_size = dim.x * dim.y * dim.z;
	DTYPE* is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);

	cuWjGetInteriorBound_any << <BLOCKS(max_size), THREADS(max_size) >> > (dst, Ax, Ay, Az, dim, max_size);
}

/***********************************************************Local Index********************************************************/
__global__ void cuWjFrac3D_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjFrac3D_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE weight, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
		{
			off_d += -cf * src[idx - 1];
		}


		// top
		dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
		{
			off_d += -cf * src[idx + 1];
		}

		// left
		DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		// front
		DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
		cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z > 0)
		{
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
		cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z < dim.z - 1)
		{
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

void CuWeightedJacobi3D::SolveFrac(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index, int size_idx, 
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			break;
		case 'z':
			cuWjFrac3D_z << <BLOCKS(size_idx), THREADS(size_idx) >> > (result, sss, f, Ax, Ay, Az, index, dim, dx, dy, dz, dx_e, weight, size_idx);
			break;
		default:
			cuWjFrac3D_n << <BLOCKS(size_idx), THREADS(size_idx) >> > (result, sss, f, Ax, Ay, Az, index, dim, dx, dy, dz, weight, size_idx);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	cudaCheckError(cudaGetLastError());
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaCheckError(cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice));
	}
}

__global__ void cuWjGetFracRes_3d_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWjGetFracRes_3d_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, int max_size)
{
	int idx_thread = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx_thread < max_size)
	{
		int idx = index[idx_thread];
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
		DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y > 0)
		{
			off_d += -cf * src[idx - 1];
		}


		// top
		dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
		cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
		a_diag += cf;
		if (idx_y < dim.y - 1)
		{
			off_d += -cf * src[idx + 1];
		}

		// left
		DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
		cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x > 0)
			off_d += -src[idx - dim.y] * cf;

		// right
		dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
		cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
		a_diag += cf;
		if (idx_x < dim.x - 1)
			off_d += -src[idx + dim.y] * cf;

		// front
		DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
		cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z > 0)
		{
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
		cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
		a_diag += cf;
		if (idx_z < dim.z - 1)
		{
			off_d += -src[idx + slice_xy] * cf;
		}

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

void CuWeightedJacobi3D::GetFracRhs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, int* index, int size_idx, 
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(size_idx), THREADS(size_idx) >> > (dst, v, f, Ax, Ay, Az, index, dim, dx, dy, dz, dx_e, size_idx);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(size_idx), THREADS(size_idx) >> > (dst, v, f, Ax, Ay, Az, index, dim, dx, dy, dz, size_idx);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

/*************************************************************************************************************************/
__global__ void cuWjFrac3D_int_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
	int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE dx_ls, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		// inner boundary
		DTYPE dx_ls_inv = 1.f / dx_ls;
		DTYPE xgx = (fmin(DTYPE(1.f), dx[idx_x] * (idx_x + 1)) - dx[idx_x] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xgy = (fmin(DTYPE(1.f), dy[idx_y] * (idx_y + 1)) - dy[idx_y] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xgz = (fmin(DTYPE(1.f), dz[idx_z] * (idx_z + 1)) - dz[idx_z] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xg_ls_o = InterpolateQuadratic3D(ls, xgx, xgy, xgz, dim_ls);
		DTYPE xg_ls = abs(xg_ls_o);
		DTYPE3 d_xg_ls = InterpolateQuadratic3D_dxyz(ls, xgx, xgy, xgz, dim_ls, dx_ls_inv);
		DTYPE3 d_xg_norm = normalize(d_xg_ls);
		DTYPE sign_ls = -xg_ls_o / (xg_ls + 1e-8f);
		//DTYPE sign_ls = -1.f;
		DTYPE a_diag_ib = 0.f;
		if (xg_ls > 1e-8f)
			a_diag_ib = sign_ls * ((d_xg_norm.y * abs(Ay[idx_ay + 1] - Ay[idx_ay])) / xg_ls / dy[idx_y] * 2.f +
			(d_xg_norm.x * abs(Ax[idx_ax + dim.y] - Ax[idx_ax])) / xg_ls / dx[idx_x] * 2.f +
			(d_xg_norm.z * abs(Az[idx_az + slice_xy] - Az[idx_az])) / xg_ls / dz[idx_z] * 2.f);
		//if (xg_ls > 1e-8f)
		//	a_diag_ib = (sign_ls * (d_xg_norm.y * (Ay[idx_ay + 1] - Ay[idx_ay])) / xg_ls / dy[idx_y] * 2.f +
		//		sign_ls * (d_xg_norm.x * (Ax[idx_ax + dim.y] - Ax[idx_ax])) / xg_ls / dx[idx_x] * 2.f +
		//		sign_ls * (d_xg_norm.z * (Az[idx_az + slice_xy] - Az[idx_az])) / xg_ls / dz[idx_z] * 2.f);
		a_diag += a_diag_ib;
		//if (xg_ls < dx_ls)
			//printf("idx: %d, %d, %d, xg: %f, %f, %f, xg_ls: %f, a_diag_ib: %f, a_diag_o: %f\n", idx_x, idx_y, idx_z, xgx, xgy, xgz, xg_ls, a_diag_ib, a_diag - a_diag_ib);


		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE> && xg_ls > eps<DTYPE>)
		{
			val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
		}
		dst[idx] = val;
	}
}

__global__ void cuWjGetFracRes_3d_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
	int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE dx_ls, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		int idx_xz = idx / dim.y;
		int idx_y = idx - idx_xz * dim.y;
		int idx_z = idx_xz / dim.x;
		int idx_x = idx_xz - dim.x * idx_z;
		int slice_xy = dim.x * dim.y;

		int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
		int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
		int idx_az = idx;

		DTYPE a_diag = 0.f;
		DTYPE off_d = 0.f;

		// bottom
		if (idx_y > 0)
		{
			DTYPE cf = Ay[idx_ay] / (dy[idx_y - 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx - 1];
		}


		// top
		if (idx_y < dim.y - 1)
		{
			DTYPE cf = Ay[idx_ay + 1] / (dy[idx_y + 1] + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			off_d += -cf * src[idx + 1];
		}

		// left
		if (idx_x > 0)
		{
			DTYPE cf = Ax[idx_ax] / (dx[idx_x - 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx - dim.y] * cf;
		}

		// right
		if (idx_x < dim.x - 1)
		{
			DTYPE cf = Ax[idx_ax + dim.y] / (dx[idx_x + 1] + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			off_d += -src[idx + dim.y] * cf;
		}

		// front
		if (idx_z > 0)
		{
			DTYPE cf = Az[idx_az] / (dz[idx_z - 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx - slice_xy] * cf;
		}

		// back
		if (idx_z < dim.z - 1)
		{
			DTYPE cf = Az[idx_az + slice_xy] / (dz[idx_z + 1] + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			off_d += -src[idx + slice_xy] * cf;
		}

		// inner boundary
		DTYPE dx_ls_inv = 1.f / dx_ls;
		DTYPE xgx = (fmin(DTYPE(1.f), dx[idx_x] * (idx_x + 1)) - dx[idx_x] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xgy = (fmin(DTYPE(1.f), dy[idx_y] * (idx_y + 1)) - dy[idx_y] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xgz = (fmin(DTYPE(1.f), dz[idx_z] * (idx_z + 1)) - dz[idx_z] * 0.5f) * dx_ls_inv - 0.5f;
		DTYPE xg_ls_o = InterpolateQuadratic3D(ls, xgx, xgy, xgz, dim_ls);
		DTYPE xg_ls = abs(xg_ls_o);
		DTYPE3 d_xg_ls = InterpolateQuadratic3D_dxyz(ls, xgx, xgy, xgz, dim_ls, dx_ls_inv);
		DTYPE3 d_xg_norm = normalize(d_xg_ls);
		DTYPE sign_ls = -xg_ls_o / (xg_ls + 1e-8f);
		//DTYPE sign_ls = -1.f;
		DTYPE a_diag_ib = 0.f;
		if (xg_ls > 1e-8f)
			a_diag_ib = sign_ls * ((d_xg_norm.y * abs(Ay[idx_ay + 1] - Ay[idx_ay])) / xg_ls / dy[idx_y] * 2.f +
			(d_xg_norm.x * abs(Ax[idx_ax + dim.y] - Ax[idx_ax])) / xg_ls / dx[idx_x] * 2.f +
			(d_xg_norm.z * abs(Az[idx_az + slice_xy] - Az[idx_az])) / xg_ls / dz[idx_z] * 2.f);
		//if (xg_ls > 1e-8f)
		//	a_diag_ib = (sign_ls * (d_xg_norm.y * (Ay[idx_ay + 1] - Ay[idx_ay])) / xg_ls / dy[idx_y] * 2.f +
		//		sign_ls * (d_xg_norm.x * (Ax[idx_ax + dim.y] - Ax[idx_ax])) / xg_ls / dx[idx_x] * 2.f +
		//		sign_ls * (d_xg_norm.z * (Az[idx_az + slice_xy] - Az[idx_az])) / xg_ls / dz[idx_z] * 2.f);
		a_diag += a_diag_ib;

		DTYPE val = 0.f;
		if (a_diag > eps<DTYPE> && xg_ls > eps<DTYPE>)
		{
			val = f[idx] - (a_diag * src[idx] + off_d);
		}
		dst[idx] = val;
	}
}

void CuWeightedJacobi3D::SolveFracIntLs(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
	int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			break;
		case 'z':
			break;
		default:
			cuWjFrac3D_int_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, ls,
				dim, dim_ls, dx, dy, dz, dx_ls, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

DTYPE CuWeightedJacobi3D::GetFracRhsIntLs(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
	int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, ls, dim, dim_ls, dx, dy, dz, dx_ls, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);

	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi3D::GetFracRhsIntLs(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* ls,
	int3 dim, int3 dim_ls, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE dx_ls, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	switch (bc)
	{
	case 'd':
		break;
	case 'z':
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, ls, dim, dim_ls, dx, dy, dz, dx_ls, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}

/******************************************************************************************/
__global__ void cuWjFrac3D_int_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
			DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y > 0)
			{
				off_d += -cf * src[idx - 1] * is_interbound[idx - 1];
			}


			// top
			dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
			cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y < dim.y - 1)
			{
				off_d += -cf * src[idx + 1] * is_interbound[idx + 1];
			}

			// left
			DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
			cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x > 0)
				off_d += -src[idx - dim.y] * cf * is_interbound[idx - dim.y];

			// right
			dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
			cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x < dim.x - 1)
				off_d += -src[idx + dim.y] * cf * is_interbound[idx + dim.y];

			// front
			DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
			cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z > 0)
			{
				off_d += -src[idx - slice_xy] * cf * is_interbound[idx - slice_xy];
			}

			// back
			dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
			cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z < dim.z - 1)
			{
				off_d += -src[idx + slice_xy] * cf * is_interbound[idx + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjFrac3D_int_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE weight, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;
			DTYPE d_cf = 2.f;
			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay] + d_cf * (Ay_df[idx_ay]));
				off_d += -cf_base * Ay[idx_ay] * src[idx - 1];
			}

			// top
			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay + 1] + d_cf * (Ay_df[idx_ay + 1]));
				off_d += -cf_base * Ay[idx_ay + 1] * src[idx + 1];
			}

			// left
			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax] + d_cf * (Ax_df[idx_ax]));
				off_d += -src[idx - dim.y] * Ax[idx_ax] * cf_base;
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax + dim.y] + d_cf * (Ax_df[idx_ax + dim.y]));
				off_d += -src[idx + dim.y] * cf_base * Ax[idx_ax + dim.y];
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az] + d_cf * (Az_df[idx_az]));
				off_d += -src[idx - slice_xy] * cf_base * Az[idx_az];
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az + slice_xy] + d_cf * (Az_df[idx_az + slice_xy]));
				off_d += -src[idx + slice_xy] * cf_base * Az[idx_az + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = (1 - weight) * src[idx] + weight * (f[idx] - off_d) / a_diag;
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
	}
}

__global__ void cuWjGetFracRes_3d_z(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;

			// bottom
			DTYPE dy_nei = idx_y > 0 ? dy[idx_y - 1] : dx_e.y;
			DTYPE cf = Ay[idx_ay] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y > 0)
			{
				off_d += -cf * src[idx - 1] * is_interbound[idx - 1];
			}


			// top
			dy_nei = idx_y < dim.y - 1 ? dy[idx_y + 1] : dx_e.y;
			cf = Ay[idx_ay + 1] / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
			a_diag += cf;
			if (idx_y < dim.y - 1)
			{
				off_d += -cf * src[idx + 1] * is_interbound[idx + 1];
			}

			// left
			DTYPE dx_nei = idx_x > 0 ? dx[idx_x - 1] : dx_e.x;
			cf = Ax[idx_ax] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x > 0)
				off_d += -src[idx - dim.y] * cf * is_interbound[idx - dim.y];

			// right
			dx_nei = idx_x < dim.x - 1 ? dx[idx_x + 1] : dx_e.x;
			cf = Ax[idx_ax + dim.y] / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
			a_diag += cf;
			if (idx_x < dim.x - 1)
				off_d += -src[idx + dim.y] * cf * is_interbound[idx + dim.y];

			// front
			DTYPE dz_nei = idx_z > 0 ? dz[idx_z - 1] : dx_e.z;
			cf = Az[idx_az] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z > 0)
			{
				off_d += -src[idx - slice_xy] * cf * is_interbound[idx - slice_xy];
			}

			// back
			dz_nei = idx_z < dim.z - 1 ? dz[idx_z + 1] : dx_e.z;
			cf = Az[idx_az + slice_xy] / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
			a_diag += cf;
			if (idx_z < dim.z - 1)
			{
				off_d += -src[idx + slice_xy] * cf * is_interbound[idx + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = f[idx] - (a_diag * src[idx] + off_d);
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

__global__ void cuWjGetFracRes_3d_n(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_interbound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, int max_size)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < max_size)
	{
		if (is_interbound[idx] > 0.5f)
		{
			int idx_xz = idx / dim.y;
			int idx_y = idx - idx_xz * dim.y;
			int idx_z = idx_xz / dim.x;
			int idx_x = idx_xz - dim.x * idx_z;
			int slice_xy = dim.x * dim.y;

			int idx_ax = idx_y + idx_x * dim.y + idx_z * (dim.x + 1) * dim.y;
			int idx_ay = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
			int idx_az = idx;

			DTYPE a_diag = 0.f;
			DTYPE off_d = 0.f;
			DTYPE d_cf = 2.f;

			// bottom
			if (idx_y > 0)
			{
				DTYPE dy_nei = dy[idx_y - 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay] + d_cf * (Ay_df[idx_ay]));
				off_d += -cf_base * Ay[idx_ay] * src[idx - 1];
			}

			// top
			if (idx_y < dim.y - 1)
			{
				DTYPE dy_nei = dy[idx_y + 1];
				DTYPE cf_base = 1.f / (dy_nei + dy[idx_y]) / dy[idx_y] * 2.f;
				a_diag += cf_base * (Ay[idx_ay + 1] + d_cf * (Ay_df[idx_ay + 1]));
				off_d += -cf_base * Ay[idx_ay + 1] * src[idx + 1];
			}

			// left
			if (idx_x > 0)
			{
				DTYPE dx_nei = dx[idx_x - 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax] + d_cf * (Ax_df[idx_ax]));
				off_d += -src[idx - dim.y] * Ax[idx_ax] * cf_base;
			}

			// right
			if (idx_x < dim.x - 1)
			{
				DTYPE dx_nei = dx[idx_x + 1];
				DTYPE cf_base = 1.f / (dx_nei + dx[idx_x]) / dx[idx_x] * 2.f;
				a_diag += cf_base * (Ax[idx_ax + dim.y] + d_cf * (Ax_df[idx_ax + dim.y]));
				off_d += -src[idx + dim.y] * cf_base * Ax[idx_ax + dim.y];
			}

			// front
			if (idx_z > 0)
			{
				DTYPE dz_nei = dz[idx_z - 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az] + d_cf * (Az_df[idx_az]));
				off_d += -src[idx - slice_xy] * cf_base * Az[idx_az];
			}

			// back
			if (idx_z < dim.z - 1)
			{
				DTYPE dz_nei = dz[idx_z + 1];
				DTYPE cf_base = 1.f / (dz_nei + dz[idx_z]) / dz[idx_z] * 2.f;
				a_diag += cf_base * (Az[idx_az + slice_xy] + d_cf * (Az_df[idx_az + slice_xy]));
				off_d += -src[idx + slice_xy] * cf_base * Az[idx_az + slice_xy];
			}

			DTYPE val = 0.f;
			if (a_diag > eps<DTYPE>)
			{
				val = f[idx] - (a_diag * src[idx] + off_d);
			}
			dst[idx] = val;
		}
		else
		{
			dst[idx] = 0.f;
		}
		//printf("idx: %d, idx_x, idx_y: %d, %d, dst: %f, src: %f, f: %f, a_diag: %f\n", idx, idx_x, idx_y, dst[idx], src[idx], f[idx], a_diag);
	}
}

void CuWeightedJacobi3D::SolveFracInt(DTYPE* dst, DTYPE* src, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, 
	DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e,
	int n_iters, DTYPE weight, char bc)
{
	int max_size = dim.x * dim.y * dim.z;
	if (src == nullptr || dst == src)
	{
		src = CuMemoryManager::GetInstance()->GetData("wj3d", max_size);
		cudaMemcpy(src, dst, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
		//CudaPrintfMat(is_bound, dim);
	}

	DTYPE* result = dst;
	DTYPE* sss = src;
	for (int i = 0; i < n_iters; i++)
	{
		switch (bc)
		{
		case 'd':
			//CudaPrintfMat(sss, dim);
			cuWjFrac3D_n_dy << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, dim, dx, dy, dz, weight, max_size);
			break;
		case 'z':
			cuWjFrac3D_int_z << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, is_bound,
				dim, dx, dy, dz, dx_e, weight, max_size);
			break;
		default:
			cuWjFrac3D_int_n << <BLOCKS(max_size), THREADS(max_size) >> > (result, sss, f, Ax, Ay, Az, Ax_df, Ay_df, Az_df, is_bound,
				dim, dx, dy, dz, weight, max_size);
		}
		SwapPointer((void**)&result, (void**)&sss);
	}
	if ((n_iters & 1) == 0)  // n_iters is even
	{
		cudaMemcpy(dst, src, sizeof(DTYPE) * max_size, cudaMemcpyDeviceToDevice);
	}
}

DTYPE CuWeightedJacobi3D::GetFracRhsInt(DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
	}

	DTYPE* dev_res = CuMemoryManager::GetInstance()->GetData("rhs", max_size);
	switch (bc)
	{
	case 'd':
		cuWjGetFracRes_3d_n_dy << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dev_res, v, f, Ax, Ay, Az, Ax_df, Ay_df, Az_df, is_bound, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
	//CudaPrintfMat(dev_res, dim);
	DTYPE sum_rhs = CudaArrayNormSum(dev_res, max_size);
	DTYPE sum_f = CudaArrayNormSum(f, max_size);
	return std::sqrt(sum_rhs) / std::sqrt(sum_f);

	cudaCheckError(cudaGetLastError());
}

void CuWeightedJacobi3D::GetFracRhsInt(DTYPE* dst, DTYPE* v, DTYPE* f, DTYPE* Ax, DTYPE* Ay, DTYPE* Az, DTYPE* Ax_df, DTYPE* Ay_df, DTYPE* Az_df, DTYPE* is_bound,
	int3 dim, DTYPE* dx, DTYPE* dy, DTYPE* dz, DTYPE3 dx_e, char bc)
{
	int max_size = dim.x * dim.y * dim.z;

	if (is_bound == nullptr)
	{
		is_bound = CuMemoryManager::GetInstance()->GetData("wj3d_int", max_size);
		cuWjGetInteriorBound << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, Ax, Ay, Az, dim, max_size);
	}

	switch (bc)
	{
	case 'd':
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, dim, dx, dy, dz, max_size);
		break;
	case 'z':
		cuWjGetFracRes_3d_z << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, is_bound, dim, dx, dy, dz, dx_e, max_size);
		break;
	default:
		cuWjGetFracRes_3d_n << <BLOCKS(max_size), NUM_THREADS >> > (dst, v, f, Ax, Ay, Az, Ax_df, Ay_df, Az_df, is_bound, dim, dx, dy, dz, max_size);
		break;
	}
	cudaCheckError(cudaGetLastError());
}