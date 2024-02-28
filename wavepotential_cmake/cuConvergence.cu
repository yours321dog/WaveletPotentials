#include "cuConvergence.cuh"
#include "Interpolation.cuh"

std::auto_ptr<CuConvergence> CuConvergence::instance_;

CuConvergence* CuConvergence::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<CuConvergence>(new CuConvergence); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

__global__ void cuGetB_vel_3d(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim.x * dim.y;
    int slice_xv = (dim.x + 1) * dim.y;
    int slice_yv = dim.x * (dim.y + 1);
    int slice_zv = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = i + j * dim.y + k * slice_qb;
        int idx_xv = i + j * dim.y + k * slice_xv;
        int idx_yv = i + j * (dim.y + 1) + k * slice_yv;
        int idx_zv = i + j * dim.y + k * slice_zv;

        DTYPE val = (xv[idx_xv + dim.y] - xv[idx_xv]) * dx_inv.x + (yv[idx_yv + 1] - yv[idx_yv]) * dx_inv.y +
            (zv[idx_zv + slice_zv] - zv[idx_zv]) * dx_inv.z;

        qb[idx_qb] = -val;
    }
}

__global__ void cuGetB_vel_frac_3d(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, 
    int3 dim, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim.x * dim.y;
    int slice_xv = (dim.x + 1) * dim.y;
    int slice_yv = dim.x * (dim.y + 1);
    int slice_zv = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = i + j * dim.y + k * slice_qb;
        int idx_xv = i + j * dim.y + k * slice_xv;
        int idx_yv = i + j * (dim.y + 1) + k * slice_yv;
        int idx_zv = i + j * dim.y + k * slice_zv;

        DTYPE val = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y]  - xv[idx_xv] * ax[idx_xv]) * dx_inv.x
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) * dx_inv.y +
            (zv[idx_zv + slice_zv] * az[idx_zv + slice_zv] - zv[idx_zv] * az[idx_zv]) * dx_inv.z;

        qb[idx_qb] = -val;
    }
}


__global__ void cuGetB_vel_frac_3d_n(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az,
    int3 dim, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim.x * dim.y;
    int slice_xv = (dim.x + 1) * dim.y;
    int slice_yv = dim.x * (dim.y + 1);
    int slice_zv = dim.x * dim.y;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim.y;
        int i = idx_xy - j * dim.y;

        int idx_qb = i + j * dim.y + k * slice_qb;
        int idx_xv = i + j * dim.y + k * slice_xv;
        int idx_yv = i + j * (dim.y + 1) + k * slice_yv;
        int idx_zv = i + j * dim.y + k * slice_zv;

        DTYPE val = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) * dx_inv.x
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) * dx_inv.y +
            (zv[idx_zv + slice_zv] * az[idx_zv + slice_zv] - zv[idx_zv] * az[idx_zv]) * dx_inv.z;

        DTYPE val_ext = 0.f;
        if (j == 0)
        {
            val_ext += xv[idx_xv] * dx_inv.x;
        }
        if (j == dim.x - 1)
        {
            val_ext -= xv[idx_xv + dim.y] * dx_inv.x;
        }
        if (i == 0)
        {
            val_ext += yv[idx_yv] * dx_inv.y;
        }
        if (i == dim.y - 1)
        {
            val_ext -= yv[idx_yv + 1] * dx_inv.y;
        }
        if (k == 0)
        {
            val_ext += zv[idx_zv] * dx_inv.z;
        }
        if (k == dim.z - 1)
        {
            val_ext -= zv[idx_zv + slice_zv] * dx_inv.z;
        }

        qb[idx_qb] = -val;

    }
}

__global__ void cuGetB_q_zero_3d(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_phi.x * dim_phi.y;
    int dim_qx_y = dim_phi.y + 2;
    int dim_qy_y = dim_phi.y + 1;
    int dim_qz_y = dim_phi.y + 2;
    int slice_qx = dim_qx_y * (dim_phi.x + 1);
    int slice_qy = dim_qy_y * (dim_phi.x + 2);
    int slice_qz = dim_qz_y * (dim_phi.x + 2);

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        int i_1 = i + 1;
        int j_1 = j + 1;
        int k_1 = k + 1;

        int idx_qb = idx;
        int idx_qx_l = i_1 + j * dim_qx_y + k_1 * slice_qx;
        int idx_qx_r = idx_qx_l + dim_qx_y;
        int idx_qy_b = i + j_1 * dim_qy_y + k_1 * slice_qy;
        int idx_qy_t = idx_qy_b + 1;
        int idx_qz_f = i_1 + j_1 * dim_qz_y + k * slice_qz;
        int idx_qz_b = idx_qz_f + slice_qz;

        qb[idx_qb] = -((qx[idx_qx_r] - qx[idx_qx_l]) * dx_inv.x + (qy[idx_qy_t] - qy[idx_qy_b]) * dx_inv.y
            + (qz[idx_qz_b] - qz[idx_qz_f]) * dx_inv.z);
    }
}

__global__ void cuGetB_q_d_3d(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int slice_qb = dim_phi.x * dim_phi.y;
        int dim_qx_y = dim_phi.y;
        int dim_qy_y = dim_phi.y - 1;
        int dim_qz_y = dim_phi.y;
        int slice_qx = dim_qx_y * (dim_phi.x - 1);
        int slice_qy = dim_qy_y * (dim_phi.x);
        int slice_qz = dim_qz_y * (dim_phi.x);

        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        int i_b = max(i - 1, 0);
        int j_l = max(j - 1, 0);
        int k_f = max(k - 1, 0);
        
        int i_t = min(i, dim_phi.y - 2);
        int j_r = min(j, dim_phi.x - 2);
        int k_b = min(k, dim_phi.z - 2);

        int idx_qb = idx;
        int idx_qx_l = i + j_l * dim_qx_y + k * slice_qx;
        int idx_qx_r = i + j_r * dim_qx_y + k * slice_qx;
        int idx_qy_b = i_b + j * dim_qy_y + k * slice_qy;
        int idx_qy_t = i_t + j * dim_qy_y + k * slice_qy;
        int idx_qz_f = i + j * dim_qz_y + k_f * slice_qz;
        int idx_qz_b = i + j * dim_qz_y + k_b * slice_qz;

        qb[idx_qb] = -((qx[idx_qx_r] - qx[idx_qx_l]) * dx_inv.x + (qy[idx_qy_t] - qy[idx_qy_b]) * dx_inv.y
            + (qz[idx_qz_b] - qz[idx_qz_f]) * dx_inv.z);
    }
}

__global__ void cuGetB_q_zero_3d_interp(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE dx, DTYPE* qp_o, 
    int3 dim_o, DTYPE dx_inv_o,  DTYPE3 off_o, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int slice_qb = dim_phi.x * dim_phi.y;
    int dim_qx_y = dim_phi.y + 2;
    int dim_qy_y = dim_phi.y + 1;
    int dim_qz_y = dim_phi.y + 2;
    int slice_qx = dim_qx_y * (dim_phi.x + 1);
    int slice_qy = dim_qy_y * (dim_phi.x + 2);
    int slice_qz = dim_qz_y * (dim_phi.x + 2);

    DTYPE dx_inv = 1.f / dx;

    if (idx < max_size)
    {
        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        int i_1 = i + 1;
        int j_1 = j + 1;
        int k_1 = k + 1;

        int idx_qb = idx;
        int idx_qx_l = i_1 + j * dim_qx_y + k_1 * slice_qx;
        int idx_qx_r = idx_qx_l + dim_qx_y;
        int idx_qy_b = i + j_1 * dim_qy_y + k_1 * slice_qy;
        int idx_qy_t = idx_qy_b + 1;
        int idx_qz_f = i_1 + j_1 * dim_qz_y + k * slice_qz;
        int idx_qz_b = idx_qz_f + slice_qz;

        DTYPE val = -((qx[idx_qx_r] - qx[idx_qx_l]) * dx_inv + (qy[idx_qy_t] - qy[idx_qy_b]) * dx_inv
            + (qz[idx_qz_b] - qz[idx_qz_f]) * dx_inv);

        DTYPE3 pos = make_DTYPE3((i + 1) * dx, (j + 1) * dx, (k + 1) * dx);
        // y direction
        DTYPE val_ext = 0.f;
        if (i == 0 && dim_o.y != dim_phi.y)
        {
            DTYPE3 xg = pos;
            xg.y -= dx;
            xg *= dx_inv_o;
            DTYPE qb_ip =  InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val_ext -= qb_ip * dx_inv * dx_inv;
        }
        if (i == dim_phi.y - 1 && dim_o.y != dim_phi.y)
        {
            DTYPE3 xg = pos;
            xg.y += dx;
            xg *= dx_inv_o;
            DTYPE qb_ip = InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val_ext += qb_ip * dx_inv * dx_inv;
        }

        // x direction
        if (j == 0 && dim_o.x != dim_phi.x)
        {
            DTYPE3 xg = pos;
            xg.x -= dx;
            xg *= dx_inv_o;
            DTYPE qb_ip = InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val_ext -= qb_ip * dx_inv * dx_inv;
        }
        if (j == dim_phi.x - 1 && dim_o.x != dim_phi.x)
        {
            DTYPE3 xg = pos;
            xg.x += dx;
            xg *= dx_inv_o;
            DTYPE qb_ip = InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val += qb_ip * dx_inv * dx_inv;
        }

        // z direction
        if (k == 0 && dim_o.z != dim_phi.z)
        {
            DTYPE3 xg = pos;
            xg.z -= dx;
            xg *= dx_inv_o;
            DTYPE qb_ip = InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val_ext -= qb_ip * dx_inv * dx_inv;
        }
        if (k == dim_phi.z - 1 && dim_o.z != dim_phi.z)
        {
            DTYPE3 xg = pos;
            xg.z += dx;
            xg *= dx_inv_o;
            DTYPE qb_ip = InterpolateLinear3D(qp_o, xg.x - off_o.x, xg.y - off_o.y, xg.z - off_o.z, dim_o);
            val_ext += qb_ip * dx_inv * dx_inv;
        }

        qb[idx_qb] = val - val_ext;
    }
}

__global__ void cuGetB_q_d_3d_interp(DTYPE* qb, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE dx, DTYPE* qx_o, DTYPE* qy_o, DTYPE* qz_o, 
    int3 dim_o, DTYPE dx_inv_o, DTYPE3 c_min, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int3 dim_qx_o = dim_o + make_int3(0, 1, 1);
        int3 dim_qy_o = dim_o + make_int3(1, 0, 1);
        int3 dim_qz_o = dim_o + make_int3(1, 1, 0);

        DTYPE dx_inv = 1.f / dx;
        int slice_qb = dim_phi.x * dim_phi.y;
        int dim_qx_y = dim_phi.y;
        int dim_qy_y = dim_phi.y - 1;
        int dim_qz_y = dim_phi.y;
        int slice_qx = dim_qx_y * (dim_phi.x - 1);
        int slice_qy = dim_qy_y * (dim_phi.x);
        int slice_qz = dim_qz_y * (dim_phi.x);

        int k = idx / slice_qb;
        int idx_xy = idx - k * slice_qb;
        int j = idx_xy / dim_phi.y;
        int i = idx_xy - j * dim_phi.y;

        int i_b = i - 1;
        int j_l = j - 1;
        int k_f = k - 1;

        int i_t = i;
        int j_r = j;
        int k_b = k;

        int idx_qb = idx;


        DTYPE3 pos = make_DTYPE3((i + 1) * dx, (j + 1) * dx, (k + 1) * dx) + c_min;
        DTYPE qx_l = 0.f;
        if (j > 0)
        {
            int idx_qx_l = i + j_l * dim_qx_y + k * slice_qx;
            qx_l = qx[idx_qx_l];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.x -= 0.5f * dx;
            xg *= dx_inv_o;
            qx_l = InterpolateLinear3D(qx_o, xg.x - 0.5f, xg.y, xg.z, dim_qx_o);
        }
        DTYPE qx_r = 0.f;
        if (j < dim_phi.x - 1)
        {
            int idx_qx_r = i + j_r * dim_qx_y + k * slice_qx;
            qx_r = qx[idx_qx_r];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.x += 0.5f * dx;
            xg *= dx_inv_o;
            qx_r = InterpolateLinear3D(qx_o, xg.x - 0.5f, xg.y, xg.z, dim_qx_o);
        }
        DTYPE qy_b = 0.f;
        if (i > 0)
        {
            int idx_qy_b = i_b + j * dim_qy_y + k * slice_qy;
            qy_b = qy[idx_qy_b];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.y -= 0.5f * dx;
            xg *= dx_inv_o;
            qy_b = InterpolateLinear3D(qy_o, xg.x, xg.y - 0.5f, xg.z, dim_qy_o);
        }
        DTYPE qy_t = 0.f;
        if (i < dim_phi.y - 1)
        {
            int idx_qy_t = i_t + j * dim_qy_y + k * slice_qy;
            qy_t = qy[idx_qy_t];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.y += 0.5f * dx;
            xg *= dx_inv_o;
            qy_t = InterpolateLinear3D(qy_o, xg.x, xg.y - 0.5f, xg.z, dim_qy_o);
        }
        DTYPE qz_f = 0.f;
        if (k > 0)
        {
            int idx_qz_f = i + j * dim_qz_y + k_f * slice_qz;
            qz_f = qz[idx_qz_f];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.z -= 0.5f * dx;
            xg *= dx_inv_o;
            qz_f = InterpolateLinear3D(qy_o, xg.x, xg.y, xg.z - 0.5f, dim_qz_o);
        }
        DTYPE qz_b = 0.f;
        if (k < dim_phi.z - 1)
        {
            int idx_qz_b = i + j * dim_qz_y + k_b * slice_qz;
            qz_b = qz[idx_qz_b];
        }
        else
        {
            DTYPE3 xg = pos;
            xg.z += 0.5f * dx;
            xg *= dx_inv_o;
            qz_b = InterpolateLinear3D(qy_o, xg.x, xg.y, xg.z - 0.5f, dim_qz_o);
        }

        qb[idx_qb] = -((qx_r - qx_l) * dx_inv + (qy_t - qy_b) * dx_inv
            + (qz_b - qz_f) * dx_inv);
        //qb[idx_qb] = val;
    }
}

void CuConvergence::GetB_3d_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, int3 dim, DTYPE3 dx)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };

    cuGetB_vel_3d << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, zv, dim, dx_inv, max_size);
}

void CuConvergence::GetB_3d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };

    cuGetB_vel_frac_3d << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, zv, ax, ay, az, dim, dx_inv, max_size);
}

void CuConvergence::GetB_3d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, DTYPE3 dx, char bc)
{
    int max_size = dim.x * dim.y * dim.z;
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };
    if (bc == 'n')
    {
        cuGetB_vel_frac_3d_n << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, zv, ax, ay, az, dim, dx_inv, max_size);
    }
    else
    {
        cuGetB_vel_frac_3d << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, zv, ax, ay, az, dim, dx_inv, max_size);
    }
}

void CuConvergence::GetB_3d_q_zeros(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx)
{
    int max_size = getsize(dim_phi);
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };

    cuGetB_q_zero_3d << < BLOCKS(max_size), THREADS(max_size) >> > (b, qx, qy, qz, dim_phi, dx_inv, max_size);
}

void CuConvergence::GetB_3d_q_zeros_interp(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx, DTYPE* qp_o,
    int3 dim_o, DTYPE dx_inv_o, DTYPE3 off_o, DTYPE3 c_min)
{
    int max_size = getsize(dim_phi);
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };
  
    cuGetB_q_zero_3d_interp << < BLOCKS(max_size), THREADS(max_size) >> > (b, qx, qy, qz, dim_phi, dx_inv.x, qp_o, dim_o, dx_inv_o, off_o, c_min, max_size);
}

void CuConvergence::GetB_3d_q_zeros_interp(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx, DTYPE* qx_o, DTYPE* qy_o, DTYPE* qz_o,
    int3 dim_o, DTYPE dx_inv_o, DTYPE3 c_min)
{
    int max_size = getsize(dim_phi);
    cuGetB_q_d_3d_interp << < BLOCKS(max_size), THREADS(max_size) >> > (b, qx, qy, qz, dim_phi, 1.f / dx.x, qx_o, qy_o, qz_o, dim_o, dx_inv_o, c_min, max_size);
}

void CuConvergence::GetB_3d_q_d(DTYPE* b, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim_phi, DTYPE3 dx)
{
    int max_size = getsize(dim_phi);
    DTYPE3 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y, DTYPE(1.f) / dx.z };

    cuGetB_q_d_3d << < BLOCKS(max_size), THREADS(max_size) >> > (b, qx, qy, qz, dim_phi, dx_inv, max_size);
}


//-------------------------------------------------------2D----------------------------------------------------------------------//
__global__ void cuGetB_vel_frac_2d(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay,
    int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int j = idx / dim.y;
        int i = idx - j * dim.y;

        int idx_qb = i + j * dim.y;
        int idx_xv = i + j * dim.y;
        int idx_yv = i + j * (dim.y + 1);

        DTYPE val = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) * dx_inv.x
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) * dx_inv.y;

        qb[idx_qb] = -val;
    }
}

__global__ void cuGetB_vel_frac_2d_n(DTYPE* qb, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay,
    int2 dim, DTYPE2 dx_inv, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int j = idx / dim.y;
        int i = idx - j * dim.y;

        int idx_qb = i + j * dim.y;
        int idx_xv = i + j * dim.y;
        int idx_yv = i + j * (dim.y + 1);

        DTYPE val = (xv[idx_xv + dim.y] * ax[idx_xv + dim.y] - xv[idx_xv] * ax[idx_xv]) * dx_inv.x
            + (yv[idx_yv + 1] * ay[idx_yv + 1] - yv[idx_yv] * ay[idx_yv]) * dx_inv.y;

        DTYPE val_ext = 0.f;
        if (j == 0)
        {
            val_ext += xv[idx_xv] * dx_inv.x;
        }
        if (j == dim.x - 1)
        {
            val_ext -= xv[idx_xv + dim.y] * dx_inv.x;
        }
        if (i == 0)
        {
            val_ext += yv[idx_yv] * dx_inv.y;
        }
        if (i == dim.y - 1)
        {
            val_ext -= yv[idx_yv + 1] * dx_inv.y;
        }

        qb[idx_qb] = -val;

    }
}

void CuConvergence::GetB_2d_frac_vel(DTYPE* b, DTYPE* xv, DTYPE* yv, DTYPE* ax, DTYPE* ay, int2 dim, DTYPE2 dx, char bc)
{
    int max_size = dim.x * dim.y;
    DTYPE2 dx_inv = { DTYPE(1.f) / dx.x, DTYPE(1.f) / dx.y };
    if (bc == 'n')
    {
        cuGetB_vel_frac_2d_n << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, ax, ay, dim, dx_inv, max_size);
    }
    else
    {
        cuGetB_vel_frac_2d << < BLOCKS(max_size), THREADS(max_size) >> > (b, xv, yv, ax, ay, dim, dx_inv, max_size);
    }
}