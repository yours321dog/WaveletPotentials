#include "StreamInitial.h"

#include <random>

void SetSliceZero(DTYPE* val, int3 dim, char ns)
{
	if (ns == 'x')
	{
		for (int k = 0; k < dim.z; k++)
		{
			for (int i = 0; i < dim.y; i++)
			{
				int idx = i + k * dim.x * dim.y;
				val[idx] = 0.f;
				val[idx + (dim.x - 1) * dim.y] = 0.f;
			}
		}
	}
	else if (ns == 'y')
	{
		for (int k = 0; k < dim.z; k++)
		{
			for (int j = 0; j < dim.x; j++)
			{
				int idx = j * dim.y + k * dim.x * dim.y;
				val[idx] = 0.f;
				val[idx + dim.y - 1] = 0.f;
			}
		}
	}
	else if (ns == 'z')
	{
		for (int j = 0; j < dim.x; j++)
		{
			for (int i = 0; i < dim.y; i++)
			{
				int idx = i + j * dim.y;
				val[idx] = 0.f;
				val[idx + (dim.z - 1) * dim.x * dim.y] = 0.f;
			}
		}
	}
}

void StreamInitial::InitialRandStream(DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, char bc)
{
	int3 dim_qx = { dim.x, dim.y + 1, dim.z + 1 };
	int3 dim_qy = { dim.x + 1, dim.y, dim.z + 1 };
	int3 dim_qz = { dim.x + 1, dim.y + 1, dim.z };

	int size_qx = dim_qx.x * dim_qx.y * dim_qx.z;
	int size_qy = dim_qy.x * dim_qy.y * dim_qy.z;
	int size_qz = dim_qz.x * dim_qz.y * dim_qz.z;


	std::default_random_engine dre;
	std::uniform_real_distribution<DTYPE> u(-1.f, 1.f);

	for (int i = 0; i < size_qx; i++)
	{
		qx[i] = u(dre);
	}

	for (int i = 0; i < size_qy; i++)
	{
		qy[i] = u(dre);
	}

	for (int i = 0; i < size_qz; i++)
	{
		qz[i] = u(dre);
	}

	if (bc == 'n')
	{
		SetSliceZero(qx, dim_qx, 'z');
		SetSliceZero(qx, dim_qx, 'y');

		SetSliceZero(qy, dim_qy, 'z');
		SetSliceZero(qy, dim_qy, 'x');

		SetSliceZero(qz, dim_qz, 'y');
		SetSliceZero(qz, dim_qz, 'x');
	}
}

void StreamInitial::GradientStream(DTYPE* xv, DTYPE* yv, DTYPE* zv, DTYPE* qx, DTYPE* qy, DTYPE* qz, int3 dim, DTYPE3 dx)
{
	int3 dim_qx = { dim.x, dim.y + 1, dim.z + 1 };
	int3 dim_qy = { dim.x + 1, dim.y, dim.z + 1 };
	int3 dim_qz = { dim.x + 1, dim.y + 1, dim.z };

	int3 dim_xv = { dim.x + 1, dim.y, dim.z };
	int3 dim_yv = { dim.x, dim.y + 1, dim.z };
	int3 dim_zv = { dim.x, dim.y, dim.z + 1 };

	DTYPE3 dx_inv = { DTYPE(1) / dx.x, DTYPE(1) / dx.y, DTYPE(1) / dx.z };;

	for (int k = 0; k < dim_xv.z; k++)
	{
		for (int j = 0; j < dim_xv.x; j++)
		{
			for (int i = 0; i < dim_xv.y; i++)
			{
				int idx = i + j * dim_xv.y + k * dim_xv.x * dim_xv.y;
				int idx_qz = i + j * dim_qz.y + k * dim_qz.x * dim_qz.y;
				int idx_qy = i + j * dim_qy.y + k * dim_qy.x * dim_qy.y;

				xv[idx] = (qz[idx_qz + 1] - qz[idx_qz]) * dx_inv.y - (qy[idx_qy + dim_qy.x * dim_qy.y] - qy[idx_qy]) * dx_inv.z;
			}
		}
	}

	for (int k = 0; k < dim_yv.z; k++)
	{
		for (int j = 0; j < dim_yv.x; j++)
		{
			for (int i = 0; i < dim_yv.y; i++)
			{
				int idx = i + j * dim_yv.y + k * dim_yv.x * dim_yv.y;
				int idx_qz = i + j * dim_qz.y + k * dim_qz.x * dim_qz.y;
				int idx_qx = i + j * dim_qx.y + k * dim_qx.x * dim_qx.y;

				yv[idx] = (qx[idx_qx + dim_qx.x * dim_qx.y] - qx[idx_qx]) * dx_inv.z - (qz[idx_qz + dim_qz.y] - qz[idx_qz]) * dx_inv.x;
			}
		}
	}

	for (int k = 0; k < dim_zv.z; k++)
	{
		for (int j = 0; j < dim_zv.x; j++)
		{
			for (int i = 0; i < dim_zv.y; i++)
			{
				int idx = i + j * dim_zv.y + k * dim_zv.x * dim_zv.y;
				int idx_qx = i + j * dim_qx.y + k * dim_qx.x * dim_qx.y;
				int idx_qy = i + j * dim_qy.y + k * dim_qy.x * dim_qy.y;

				zv[idx] = (qy[idx_qy + dim_qy.y] - qy[idx_qy]) * dx_inv.x - (qx[idx_qx + 1] - qx[idx_qx]) * dx_inv.y;
			}
		}
	}
}

void StreamInitial::GradientStream(DTYPE* xv, DTYPE* yv,  DTYPE* qz, int2 dim, DTYPE2 dx)
{
	int2 dim_qz = { dim.x + 1, dim.y + 1 };

	int2 dim_xv = { dim.x + 1, dim.y };
	int2 dim_yv = { dim.x, dim.y + 1 };

	DTYPE2 dx_inv = { DTYPE(1) / dx.x, DTYPE(1) / dx.y };

	for (int j = 0; j < dim_xv.x; j++)
	{
		for (int i = 0; i < dim_xv.y; i++)
		{
			int idx = i + j * dim_xv.y;
			int idx_qz = i + j * dim_qz.y;

			xv[idx] = (qz[idx_qz + 1] - qz[idx_qz]) * dx_inv.y;
		}
	}

	for (int j = 0; j < dim_yv.x; j++)
	{
		for (int i = 0; i < dim_yv.y; i++)
		{
			int idx = i + j * dim_yv.y;
			int idx_qz = i + j * dim_qz.y;

			yv[idx] = -(qz[idx_qz + dim_qz.y] - qz[idx_qz]) * dx_inv.x;
		}
	}
}

StreamInitial* StreamInitial::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<StreamInitial>(new StreamInitial); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

std::auto_ptr<StreamInitial> StreamInitial::instance_;