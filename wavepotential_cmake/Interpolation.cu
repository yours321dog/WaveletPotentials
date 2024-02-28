#include "Interpolation.cuh"

std::auto_ptr<Interpolation> Interpolation::instance_;

Interpolation* Interpolation::GetInstance()
{
	if (!instance_.get())
		instance_ = std::auto_ptr<Interpolation>(new Interpolation); // ����ָ������ͷŸ���Դ
	return instance_.get(); // ����instance_.get();��û�з���instance��ָ�������Ȩ
}