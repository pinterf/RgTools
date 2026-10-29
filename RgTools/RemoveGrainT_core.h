#ifndef __REMOVEGRAINT_CORE_H__
#define __REMOVEGRAINT_CORE_H__

// TemporalRepair core: no host API

#include "rg_core.h"

// plane processors: TemporalRepair (modes 0, 4) and SmoothTemporalRepair (modes 1, 2, 3); width in pixels
typedef void (PlaneProcessor_t)(uint8_t* dp8, int dpitch, const uint8_t* sp1_8, int spitch1, const uint8_t* sp2_8, int spitch2, const uint8_t* pp8, int ppitch, const uint8_t* np8, int npitch, int width, int height);
typedef void (PlaneProcessor_st)(uint8_t* dp8, const uint8_t* previous8, const uint8_t* sp8, const uint8_t* next8, intptr_t* pitches, int width, int height);

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

struct TemporalRepairParams {
  int mode; // 0..4; 1, 2, 3: SmoothTemporalRepair
  bool grey; // only the first plane is processed (the others: random content)
  int opt;  // -1: auto, 0: C, 1: SSE2, 2: SSE4.1
};

class TemporalRepairCore {
public:
  // checks the parameters and selects the kernels; returns nullptr or the error text
  const char* init(const TemporalRepairParams& params, const RgFormat& format, int cpu_flags);

  bool smooth() const { return smooth_; } // modes 1, 2, 3

  // plane_index: 0..2 (Y, U, V or G, B, R); plane.src: the clip, frame n;
  // prev, cur, next: the same plane of the original clip, frames n-1, n, n+1
  void process_plane(int plane_index, const RgPlane& plane, const uint8_t* prev, int prev_pitch,
    const uint8_t* cur, int cur_pitch, const uint8_t* next, int next_pitch) const;

private:
  RgFormat format_ = {};
  bool grey_ = false;
  bool smooth_ = false;
  // modes 0, 4
  PlaneProcessor_t* t_ = nullptr;
  PlaneProcessor_t* t_chroma_ = nullptr; // float: chroma range -0.5..0.5
  PlaneProcessor_t* t_c_ = nullptr;      // planes too narrow for SIMD
  PlaneProcessor_t* t_chroma_c_ = nullptr;
  // modes 1, 2, 3
  PlaneProcessor_st* st_ = nullptr;
  PlaneProcessor_st* st_chroma_ = nullptr;
  PlaneProcessor_st* st_c_ = nullptr;
  PlaneProcessor_st* st_chroma_c_ = nullptr;
  char error_[64] = {};
};

#endif
