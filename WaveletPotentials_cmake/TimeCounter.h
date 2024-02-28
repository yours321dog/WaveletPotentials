#ifndef __TIMECOUNTER_H__
#define __TIMECOUNTER_H__

#include <map>
#include "Utils.h"

class TimeCounter
{
public:
    TimeCounter();
    static TimeCounter* GetInstance();

    void tic(std::string str);
    double toc(std::string str);

    double record_toc(std::string str);
    double get_total_time(std::string str);
    void reset_total_time(std::string str);

private:
    std::map<std::string, double> times_map_;
    std::map<std::string, double> times_total_;
    static std::auto_ptr<TimeCounter> instance_;
};

#endif