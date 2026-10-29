#ifndef __REMOVEGRAINT___H__
#define __REMOVEGRAINT___H__

#include "common.h" // INTEL_INTRINSICS for the MSVC solution
#include "rg_avs.h"

// plane processors: TemporalRepair (modes 0, 4) and SmoothTemporalRepair (modes 1, 2, 3)
typedef void (PlaneProcessor_t)(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height);
typedef void (PlaneProcessor_st)(BYTE* dp8, const BYTE* previous8, const BYTE* sp8, const BYTE* next8, intptr_t* pitches, int width, int height);

#ifdef INTEL_INTRINSICS
// RemoveGrainT_sse.cpp, indexed by mode
extern PlaneProcessor_t* t_sse2_functions[];
extern PlaneProcessor_t* t_sse41_functions[];
extern PlaneProcessor_t* t_sse41_functions_16[];
extern PlaneProcessor_t* t_sse41_functions_32[];
extern PlaneProcessor_t* t_sse41_functions_chroma_32[];
extern PlaneProcessor_st* st_sse2_functions[];
extern PlaneProcessor_st* st_sse41_functions_16[];
extern PlaneProcessor_st* st_sse2_functions_32[];
extern PlaneProcessor_st* st_sse2_functions_chroma_32[];
#endif

AVSValue __cdecl Create_TemporalRepair(AVSValue args, void* user_data, IScriptEnvironment* env);

#endif	// __REMOVEGRAINT___H__
