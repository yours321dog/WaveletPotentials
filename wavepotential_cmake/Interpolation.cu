#include "Interpolation.cuh"

std::auto_ptr<Interpolation> Interpolation::instance_;

Interpolation* Interpolation::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<Interpolation>(new Interpolation); // 智能指针可以释放改资源
	return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}