#ifndef __REMOVEGRAIN_CORE_H__
#define __REMOVEGRAIN_CORE_H__

// RemoveGrain core: no host API

#include "rg_core.h"

// plane processor; rowsize in bytes, bits_per_pixel: the max pixel value of the 10-16 bit kernels
typedef void (PlaneProcessor)(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel);

// mode -1 and 0 entries of the C and SSE function tables (the AVX2 tables have their own)
PlaneProcessor rg_do_nothing;
PlaneProcessor rg_copy_plane;

struct RemoveGrainParams {
  int mode;  // -1..28
  int modeU; // UNDEFINED_MODE (or below): same as mode
  int modeV; // UNDEFINED_MODE (or below): same as modeU
  int opt;   // 0: auto, 1: C, 2: SSE2, 3: SSE4.1, 4: AVX2
};

class RemoveGrainCore {
public:
  static constexpr int UNDEFINED_MODE = -2;

  // checks the parameters and selects the kernels; returns nullptr or the error text
  const char* init(const RemoveGrainParams& params, const RgFormat& format, int cpu_flags);

  // plane_index: 0..2 (Y, U, V or G, B, R)
  void process_plane(int plane_index, const RgPlane& plane) const;

private:
  int mode_[3] = {};
  RgFormat format_ = {};
  PlaneProcessor** functions_ = nullptr;
  PlaneProcessor** functions_chroma_ = nullptr; // float: chroma range -0.5..0.5
};

#endif
