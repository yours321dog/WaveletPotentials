#include "TimeCounter.h"

TimeCounter::TimeCounter()
{}

std::auto_ptr<TimeCounter> TimeCounter::instance_;

TimeCounter* TimeCounter::GetInstance()
{
    if (!instance_.get())
        instance_ = std::auto_ptr<TimeCounter>(new TimeCounter); // 智能指针可以释放改资源
    return instance_.get(); // 返回instance_.get();并没有返回instance的指针的所有权
}

void TimeCounter::tic(std::string str)
{
    times_map_[str] = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
}

double TimeCounter::toc(std::string str)
{
    double end_time = std::clock() / static_cast<double>(CLOCKS_PER_SEC);
    double res = 0.;
    if (times_map_.find(str) != times_map_.end())
    {
        res = end_time - times_map_[str];
    }
    return res;
}

double TimeCounter::record_toc(std::string str)
{
    if (times_total_.find(str) != times_total_.end())
    {
        times_total_[str] += toc(str);
    }
    else
    {
        reset_total_time(str);
    }
    return times_total_[str];
}

double TimeCounter::get_total_time(std::string str)
{
    if (times_total_.find(str) != times_total_.end())
    {
        return times_total_[str];
    }
    else
    {
        return 0.;
    }
}

void TimeCounter::reset_total_time(std::string str)
{
    times_total_[str] = 0.;
}