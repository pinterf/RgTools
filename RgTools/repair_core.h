#ifndef __REPAIR_CORE_H__
#define __REPAIR_CORE_H__

// Repair core: no host API

#include "rg_core.h"

// plane processor; rowsize in bytes, bits_per_pixel: the max pixel value of the 10-16 bit kernels
typedef void (RepairPlaneProcessor)(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel);

// mode -1 and 0 entries of the function tables, shared by the C and SIMD tables
RepairPlaneProcessor repair_do_nothing;
RepairPlaneProcessor repair_copy_plane;

struct RepairParams {
  int mode;  // -1..28, except 25
  int modeU; // UNDEFINED_MODE (or below): same as mode
  int modeV; // UNDEFINED_MODE (or below): same as modeU
  int opt;   // 0: auto, 1: C, 2: SSE2, 3: SSE4.1
};

class RepairCore {
public:
  static constexpr int UNDEFINED_MODE = -2;

  // checks the parameters and the formats (clip, reference clip) and selects the kernels; returns nullptr or the error text
  const char* init(const RepairParams& params, const RgFormat& format, const RgFormat& ref_format, int cpu_flags);

  // plane_index: 0..2 (Y, U, V or G, B, R); ref: the same plane of the reference clip
  void process_plane(int plane_index, const RgPlane& plane, const uint8_t* ref, int ref_pitch) const;

private:
  int mode_[3] = {};
  RgFormat format_ = {};
  RepairPlaneProcessor** functions_ = nullptr;
  RepairPlaneProcessor** functions_chroma_ = nullptr; // float: chroma range -0.5..0.5
};

#endif
