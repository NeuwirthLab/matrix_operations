#ifndef _PROFILER_HPP_
#define _PROFILER_HPP_



#ifdef LIKWID_PERF
#include "likwid-marker.h"
#define PROFILER_INIT LIKWID_MARKER_INIT
#define PROFILER_START(tag) LIKWID_MARKER_START(tag)
#define PROFILER_STOP(tag) LIKWID_MARKER_STOP(tag)
#define PROFILER_CLOSE LIKWID_MARKER_CLOSE
#elif PAPI_PERF
#include "papi.h"
#define PROFILER_INIT
#define PROFILER_CLOSE
#define PROFILER_START(tag) PAPI_hl_region_begin(tag)
#define PROFILER_STOP(tag) PAPI_hl_region_end(tag)
#else
#define PROFILER_INIT
#define PROFILER_CLOSE
#define PROFILER_START(tag)
#define PROFILER_STOP(tag)
#endif
#endif
