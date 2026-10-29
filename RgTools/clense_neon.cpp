// Clense, ForwardClense, BackwardClense: aarch64 NEON kernels

#include "clense_core.h"
#include "common.h"

#ifdef NEON_INTRINSICS

#include "common_neon.h"

namespace {

// The vector loop runs to the end of the last vector of the row: past rowsize, within the pitch (the hosts give
// pitches of a multiple of 16 bytes at least), as the SSE code.

// Clense: the pixel clipped to the range of the previous and the next frame
template<class P>
static void clense_plane_neon(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int /*bits_per_pixel*/) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < rowsize; x += 16) {
      auto ref1 = P::load(pRef1 + x);
      auto ref2 = P::load(pRef2 + x);
      P::store(pDst + x, neon_clip<P>(P::load(pSrc + x), P::min(ref1, ref2), P::max(ref1, ref2)));
    }
    pDst += dstPitch;
    pSrc += srcPitch;
    pRef1 += ref1Pitch;
    pRef2 += ref2Pitch;
  }
}

// ForwardClense, BackwardClense: the range extrapolated from ref1 (the nearer frame) and ref2
template<class P>
static void sclense_plane_neon(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int bits_per_pixel) {
  const auto ctx = P::make_ctx(bits_per_pixel);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < rowsize; x += 16) {
      auto ref1 = P::load(pRef1 + x);
      auto ref2 = P::load(pRef2 + x);
      auto minref = P::min(ref1, ref2);
      auto maxref = P::max(ref1, ref2);
      auto ma = P::subs(maxref, ref2);
      auto mi = P::subs(ref2, minref);
      if constexpr (P::is_float) {
        ma = P::adds_px(ma, maxref);
        mi = P::subs_px(minref, mi);
      }
      else {
        ma = P::clamp_max(P::adds(ma, maxref), ctx); // saturation is not enough below 16 bit
        mi = P::subs(minref, mi);
      }
      P::store(pDst + x, neon_clip<P>(P::load(pSrc + x), mi, ma));
    }
    pDst += dstPitch;
    pSrc += srcPitch;
    pRef1 += ref1Pitch;
    pRef2 += ref2Pitch;
  }
}

} // namespace

// The NEON processor of the mode / format, the counterpart of the C selection in ClenseCore::init
ClenseProcessor* clense_neon_processor(ClenseMode mode, int pixelsize, bool chroma) {
  const bool both = mode == ClenseMode::BOTH;
  if (pixelsize == 1)
    return both ? clense_plane_neon<NeonU8> : sclense_plane_neon<NeonU8>;
  if (pixelsize == 2)
    return both ? clense_plane_neon<NeonU16> : sclense_plane_neon<NeonU16>;
  if (both)
    return clense_plane_neon<NeonF32<false>>;
  return chroma ? sclense_plane_neon<NeonF32<true>> : sclense_plane_neon<NeonF32<false>>;
}

#endif // NEON_INTRINSICS
