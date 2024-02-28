#include "cuBoundSelector.cuh"
#include "Interpolation.cuh"

__global__ void cuSelectByFrac(int* is_select, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim, int max_size)
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
			is_select[idx] = 0;
		}
		else
		{
			is_select[idx] = 1;
		}
	}
}


__global__ void cuSelectByLevelSet(DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, int3 dim, DTYPE select_width, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        int idx_xz = idx / dim.y;
        int idx_y = idx - idx_xz * dim.y;
        int idx_z = idx_xz / (dim.x);
        int idx_x = idx_xz - idx_z * (dim.x);

        DTYPE3 xg = make_DTYPE3(idx_x, idx_y, idx_z);
        DTYPE ls_p = InterpolateQuadratic3D(ls, xg.x, xg.y, xg.z, dim);
        //if (abs(ls_p) > 4.f * dx)
        if (ls_p > select_width)
        {
            int idx_xv = idx_y + idx_x * dim.y + idx_z * dim.y * (dim.x + 1);
            int idx_yv = idx_y + idx_x * (dim.y + 1) + idx_z * dim.x * (dim.y + 1);
            int idx_zv = idx_y + idx_x * dim.y + idx_z * dim.x * dim.y;
            ax[idx_xv] = 0.f;
            if (idx_x == dim.x - 1)
            {
                ax[idx_xv + dim.y] = 0.f;
            }
            ay[idx_yv] = 0.f;
            if (idx_y == dim.y - 1)
            {
                ay[idx_yv + 1] = 0.f;
            }
            az[idx_zv] = 0.f;
            if (idx_z == dim.z - 1)
            {
                az[idx_zv + dim.x * dim.y] = 0.f;
            }
        }

        //if (idx_z == 10)
        //    printf("idx: %d (%d, %d, %d), xg: (%f, %f, %f), ls_p: %f, 5 * dx: %f\n", idx, idx_x, idx_y, idx_z, xg.x, xg.y, xg.z, ls_p, 5 * dx);
    }
}

__global__ void cuSelectToIndex(int* index, int* select_sum, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size - 1)
    {
        if (select_sum[idx] < select_sum[idx + 1])
        {
            index[select_sum[idx]] = idx;
        }

        //if (idx_z == 10)
        //    printf("idx: %d (%d, %d, %d), xg: (%f, %f, %f), ls_p: %f, 5 * dx: %f\n", idx, idx_x, idx_y, idx_z, xg.x, xg.y, xg.z, ls_p, 5 * dx);
    }
}

__global__ void cuSelectFromValue(int* is_select, DTYPE* value, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        if (value[idx] > eps<DTYPE>)
        {
            is_select[idx] = 1;
        }
        else
        {
            is_select[idx] = 0;
        }

        //if (idx_z == 10)
        //    printf("idx: %d (%d, %d, %d), xg: (%f, %f, %f), ls_p: %f, 5 * dx: %f\n", idx, idx_x, idx_y, idx_z, xg.x, xg.y, xg.z, ls_p, 5 * dx);
    }
}

__global__ void cuValueSelect(DTYPE* value, int* is_select, int max_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < max_size)
    {
        if (is_select[idx] == 0)
        {
            value[idx] = 0.f;
        }
    }
}

void CuBoundSelector::SelectByFrac(int* is_bound, DTYPE* ax, DTYPE* ay, DTYPE* az, int3 dim)
{
	int max_size = getsize(dim);
	cuSelectByFrac << <BLOCKS(max_size), THREADS(max_size) >> > (is_bound, ax, ay, az, dim, max_size);
}

void CuBoundSelector::SelectAxyzByLevels(DTYPE* ax, DTYPE* ay, DTYPE* az, DTYPE* ls, int3 dim, DTYPE select_width)
{
    int max_size = getsize(dim);
    cuSelectByLevelSet << <BLOCKS(max_size), THREADS(max_size) >> > (ax, ay, az, ls, dim, select_width, max_size);
}

void CuBoundSelector::SelectToIndex(int* index, int* select_sum, int size)
{
    cuSelectToIndex << <BLOCKS(size), THREADS(size) >> > (index, select_sum, size);
}

void CuBoundSelector::SelectFromWholeDomain(int* is_select, DTYPE* value, int size)
{
    cuSelectFromValue << <BLOCKS(size), THREADS(size) >> > (is_select, value, size);
}

void CuBoundSelector::ValueSelect(DTYPE* value, int* is_select, int size)
{
    cuValueSelect << <BLOCKS(size), THREADS(size) >> > (value, is_select, size);
}