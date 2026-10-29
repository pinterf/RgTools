#ifndef __CLENSE_CORE_H__
#define __CLENSE_CORE_H__

// Clense, ForwardClense, BackwardClense core: no host API

#include "rg_core.h"

template<typename pixel_t>
using CModeProcessor = pixel_t (*)(pixel_t, pixel_t, pixel_t, int); // src, ref1, ref2, max_pixel_value (10-16 bit)

enum class ClenseMode {
    FORWARD,  // ForwardClense: ref1 = n+1, ref2 = n+2
    BACKWARD, // BackwardClense: ref1 = n-1, ref2 = n-2
    BOTH      // Clense: ref1 = n-1 (previous), ref2 = n+1 (next)
};

// plane processor; width: rowsize in bytes
typedef void (ClenseProcessor)(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int width, int height, int bits_per_pixel);

#ifdef INTEL_INTRINSICS
// clense_sse.cpp
ClenseProcessor* clense_sse_processor(ClenseMode mode, int pixelsize, bool chroma);
#endif

struct ClenseParams {
  ClenseMode mode;
  bool grey; // YUV: only Y is processed, U and V are left unprocessed (random content)
};

class ClenseCore {
public:
  // checks the parameters and selects the kernels; returns nullptr or the error text
  const char* init(const ClenseParams& params, const RgFormat& format, int cpu_flags);

  // plane_index: 0..2 (Y, U, V or G, B, R); ref1, ref2: the same plane of the reference frames (see ClenseMode)
  void process_plane(int plane_index, const RgPlane& plane, const uint8_t* ref1, int ref1_pitch, const uint8_t* ref2, int ref2_pitch) const;

private:
  RgFormat format_ = {};
  bool grey_ = false;
  bool simd_ = false;
  ClenseProcessor* processor_ = nullptr;
  ClenseProcessor* processor_chroma_ = nullptr; // float: chroma range -0.5..0.5
  ClenseProcessor* processor_c_ = nullptr;      // planes not 16 byte aligned
  ClenseProcessor* processor_chroma_c_ = nullptr;
};

#endif
