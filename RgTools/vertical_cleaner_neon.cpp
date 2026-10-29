// VerticalCleaner: aarch64 NEON kernels and function tables

#include "vertical_cleaner_core.h"
#include "common.h"

#ifdef NEON_INTRINSICS

#include "common_neon.h"

namespace {

// The vector loops run to the end of the last vector of the row: past rowsize, within the pitch (the hosts give
// pitches of a multiple of 16 bytes at least), as the SSE code.

// mode 1: the median of the pixel and the pixels above and below
template<class P>
static void vcleaner_median_neon(uint8_t* pDst, const uint8_t* pSrc, int dstPitch, int srcPitch, int rowsize, int height, int /*bits_per_pixel*/) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
  pSrc += srcPitch;
  pDst += dstPitch;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 0; x < rowsize; x += 16) {
      auto up = P::load(pSrc + x - srcPitch);
      auto center = P::load(pSrc + x);
      auto down = P::load(pSrc + x + srcPitch);
      P::store(pDst + x, P::min(P::max(P::min(up, down), center), P::max(up, down)));
    }
    pSrc += srcPitch;
    pDst += dstPitch;
  }
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

// mode 2: relaxed median (two rows above and below)
template<class P>
static void vcleaner_relaxed_median_neon(uint8_t* pDst, const uint8_t* pSrc, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
  const auto ctx = P::make_ctx(bits_per_pixel);
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);
  pSrc += srcPitch * 2;
  pDst += dstPitch * 2;
  for (int y = 2; y < height - 2; ++y) {
    for (int x = 0; x < rowsize; x += 16) {
      auto p2 = P::load(pSrc + x - srcPitch * 2);
      auto p1 = P::load(pSrc + x - srcPitch);
      auto c = P::load(pSrc + x);
      auto n1 = P::load(pSrc + x + srcPitch);
      auto n2 = P::load(pSrc + x + srcPitch * 2);

      typename P::V pt, nt;
      if constexpr (P::is_float) {
        pt = P::adds_px(P::subs(p1, p2), p1);
        nt = P::adds_px(P::subs(n1, n2), n1);
      }
      else {
        pt = P::clamp_max(P::adds(P::subs(p1, p2), p1), ctx);
        nt = P::clamp_max(P::adds(P::subs(n1, n2), n1), ctx);
      }
      auto upper = P::max(P::max(P::min(pt, nt), p1), n1);

      if constexpr (P::is_float) {
        pt = P::subs_px(p1, P::subs(p2, p1));
        nt = P::subs_px(n1, P::subs(n2, n1));
      }
      else {
        pt = P::subs(p1, P::subs(p2, p1));
        nt = P::subs(n1, P::subs(n2, n1));
      }
      auto lower = P::min(P::max(pt, nt), P::min(p1, n1));

      P::store(pDst + x, neon_clip<P>(c, lower, upper));
    }
    pSrc += srcPitch;
    pDst += dstPitch;
  }
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);
}

} // namespace

VCleanerProcessor* vcleaner_neon_functions[] = { vcleaner_do_nothing, vcleaner_copy_plane, vcleaner_median_neon<NeonU8>, vcleaner_relaxed_median_neon<NeonU8> };
// 9-16 bit (the max pixel value comes from bits_per_pixel)
VCleanerProcessor* vcleaner_neon_functions_16[] = { vcleaner_do_nothing, vcleaner_copy_plane, vcleaner_median_neon<NeonU16>, vcleaner_relaxed_median_neon<NeonU16> };
VCleanerProcessor* vcleaner_neon_functions_32[] = { vcleaner_do_nothing, vcleaner_copy_plane, vcleaner_median_neon<NeonF32<false>>, vcleaner_relaxed_median_neon<NeonF32<false>> };
VCleanerProcessor* vcleaner_neon_functions_32_chroma[] = { vcleaner_do_nothing, vcleaner_copy_plane, vcleaner_median_neon<NeonF32<true>>, vcleaner_relaxed_median_neon<NeonF32<true>> };

#endif // NEON_INTRINSICS
