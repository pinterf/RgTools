#ifndef __VERTICAL_CLEANER_CORE_H__
#define __VERTICAL_CLEANER_CORE_H__

// VerticalCleaner core: no host API

#include "rg_core.h"

// plane processor; rowsize in bytes, bits_per_pixel: the max pixel value of the 10-16 bit kernels
typedef void (VCleanerProcessor)(uint8_t* pDst, const uint8_t *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel);

// mode -1 and 0 entries of the function tables, shared by the C and SIMD tables
VCleanerProcessor vcleaner_do_nothing;
VCleanerProcessor vcleaner_copy_plane;

struct VerticalCleanerParams {
  int mode;  // -1..2
  int modeU; // UNDEFINED_MODE (or below): same as mode
  int modeV; // UNDEFINED_MODE (or below): same as modeU
};

class VerticalCleanerCore {
public:
  static constexpr int UNDEFINED_MODE = -2;

  // checks the parameters and sets up the plane modes; returns nullptr or the error text
  const char* init(const VerticalCleanerParams& params, const RgFormat& format, int cpu_flags);

  // plane_index: 0..2 (Y, U, V or G, B, R)
  void process_plane(int plane_index, const RgPlane& plane) const;

private:
  int mode_[3] = {};
  RgFormat format_ = {};
  int cpu_flags_ = 0;
};

#endif
