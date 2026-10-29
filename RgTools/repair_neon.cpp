// Repair: aarch64 NEON plane loop and function tables

#include "repair_core.h"
#include "common.h"

#ifdef NEON_INTRINSICS

#include "repair_functions_neon.h" // kernels, in an anonymous namespace

namespace {

template<class P> using RepairNeonKernel = RV<P>(*)(const uint8_t*, RV<P>, int, const RCtx<P>&);

// rows 1..height-2: the vector kernel from column 1 to width-2, the border columns and rows are copied from the clip
template<class P, RepairNeonKernel<P> kernel>
static void process_plane_neon(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel) {
  using T = typename P::T;
  constexpr int ps = (int)sizeof(T);
  constexpr int N = P::N;
  const auto ctx = P::make_ctx(bits_per_pixel);
  const int width = rowsize / ps;

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;

  for (int y = 1; y < height - 1; ++y) {
    reinterpret_cast<T*>(pDst)[0] = reinterpret_cast<const T*>(pSrc)[0];
    P::store(pDst + ps, kernel(pRef + ps, P::load(pSrc + ps), refPitch, ctx));
    // the vector and its right neighbours stay inside the row
    for (int x = N; x < width - N; x += N)
      P::store(pDst + x * ps, kernel(pRef + x * ps, P::load(pSrc + x * ps), refPitch, ctx));
    if (width - 1 - N >= 1) { // the last pixels before the right border; width == vector + 1: covered by the first vector
      const int x = width - 1 - N;
      P::store(pDst + x * ps, kernel(pRef + x * ps, P::load(pSrc + x * ps), refPitch, ctx));
    }
    reinterpret_cast<T*>(pDst)[width - 1] = reinterpret_cast<const T*>(pSrc)[width - 1];
    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

} // namespace

// mode -1..28 (11: as 1, 25: does not exist); one table per pixel type (float: luma and chroma range)
#define REPAIR_NEON_TABLE(P) { \
  repair_do_nothing, \
  repair_copy_plane, \
  process_plane_neon<P, repair_mode1_neon<P>>, \
  process_plane_neon<P, repair_mode2_neon<P>>, \
  process_plane_neon<P, repair_mode3_neon<P>>, \
  process_plane_neon<P, repair_mode4_neon<P>>, \
  process_plane_neon<P, repair_mode5_neon<P>>, \
  process_plane_neon<P, repair_mode6_neon<P>>, \
  process_plane_neon<P, repair_mode7_neon<P>>, \
  process_plane_neon<P, repair_mode8_neon<P>>, \
  process_plane_neon<P, repair_mode9_neon<P>>, \
  process_plane_neon<P, repair_mode10_neon<P>>, \
  process_plane_neon<P, repair_mode1_neon<P>>, \
  process_plane_neon<P, repair_mode12_neon<P>>, \
  process_plane_neon<P, repair_mode13_neon<P>>, \
  process_plane_neon<P, repair_mode14_neon<P>>, \
  process_plane_neon<P, repair_mode15_neon<P>>, \
  process_plane_neon<P, repair_mode16_neon<P>>, \
  process_plane_neon<P, repair_mode17_neon<P>>, \
  process_plane_neon<P, repair_mode18_neon<P>>, \
  process_plane_neon<P, repair_mode19_neon<P>>, \
  process_plane_neon<P, repair_mode20_neon<P>>, \
  process_plane_neon<P, repair_mode21_neon<P>>, \
  process_plane_neon<P, repair_mode22_neon<P>>, \
  process_plane_neon<P, repair_mode23_neon<P>>, \
  process_plane_neon<P, repair_mode24_neon<P>>, \
  repair_do_nothing, \
  process_plane_neon<P, repair_mode26_neon<P>>, \
  process_plane_neon<P, repair_mode27_neon<P>>, \
  process_plane_neon<P, repair_mode28_neon<P>>, \
}

RepairPlaneProcessor* repair_neon_functions[] = REPAIR_NEON_TABLE(NeonU8);
// 9-16 bit (the max pixel value comes from bits_per_pixel)
RepairPlaneProcessor* repair_neon_functions_16[] = REPAIR_NEON_TABLE(NeonU16);
RepairPlaneProcessor* repair_neon_functions_32[] = REPAIR_NEON_TABLE(NeonF32<false>);
RepairPlaneProcessor* repair_neon_functions_32_chroma[] = REPAIR_NEON_TABLE(NeonF32<true>);

#undef REPAIR_NEON_TABLE

#endif // NEON_INTRINSICS
