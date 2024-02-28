#ifndef _LOG_H
#define _LOG_H
#include <stdio.h>
#include <stdarg.h>
#include <Windows.h>
#include "cuWHHD_Export.h"
#ifdef USE_MEX
#include "mex.h"
#endif

CUWHHD_API int commonPrintf(const char* Format, ...);

static long long milliseconds_now() {
	static LARGE_INTEGER s_frequency;
	static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
	if (s_use_qpc) {
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		return (1000LL * now.QuadPart) / s_frequency.QuadPart;
	}
	else {
		return GetTickCount64();
	}
}
static long long tickTime(bool isPrint, const char* info = NULL) {
	static long long prevTime = milliseconds_now();
	long long curTime = milliseconds_now();
	long long res = curTime - prevTime;
	prevTime = curTime;
	if (isPrint) {
		if (info) {
			commonPrintf("%s:%f\n", info, res / 1000.f);
		}
		else {
			commonPrintf("%f\n", res / 1000.f);
		}

	}
	return res;
}
#define tic tickTime(false);
#define toc tickTime(true);
#define toc_info(info) tickTime(true,info);
#endif