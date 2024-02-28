#include "Log.h"
CUWHHD_API int commonPrintf(const char* format, ...) {
	char buff[128];
	va_list args;
	va_start(args, format);
	vsprintf_s(buff, format, args);
	va_end(args);
	int r = 0;
#ifdef USE_MEX
	r = mexPrintf(buff);
#else
	r = printf(buff);
#endif

	return r;
}