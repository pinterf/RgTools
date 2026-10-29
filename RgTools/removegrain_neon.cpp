// RemoveGrain: aarch64 NEON plane loops and function tables

#include "removegrain_core.h"
#include "common.h"

#ifdef NEON_INTRINSICS

#include "rg_functions_neon.h" // kernels, in an anonymous namespace

namespace {

template<class P> using NeonKernel = NV<P>(*)(const uint8_t*, int, const NCtx<P>&);

// rows 1..height-2: the vector kernel from column 1 to width-2, the border columns and rows are copied
template<class P, NeonKernel<P> kernel>
static void process_plane_neon(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
  using T = typename P::T;
  constexpr int ps = (int)sizeof(T);
  constexpr int N = P::N;
  const auto ctx = P::make_ctx(bits_per_pixel);
  const int width = rowsize / ps;

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
  pSrc += srcPitch;
  pDst += dstPitch;

  for (int y = 1; y < height - 1; ++y) {
    reinterpret_cast<T*>(pDst)[0] = reinterpret_cast<const T*>(pSrc)[0];
    P::store(pDst + ps, kernel(pSrc + ps, srcPitch, ctx));
    // the vector and its right neighbours stay inside the row
    for (int x = N; x < width - N; x += N)
      P::store(pDst + x * ps, kernel(pSrc + x * ps, srcPitch, ctx));
    if (width - 1 - N >= 1) { // the last pixels before the right border; width == vector + 1: covered by the first vector
      const int x = width - 1 - N;
      P::store(pDst + x * ps, kernel(pSrc + x * ps, srcPitch, ctx));
    }
    reinterpret_cast<T*>(pDst)[width - 1] = reinterpret_cast<const T*>(pSrc)[width - 1];
    pSrc += srcPitch;
    pDst += dstPitch;
  }

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

// field modes: every second row from pSrc (one row below the first processed row), the rows of the other field copied
template<class P, NeonKernel<P> kernel>
static void process_halfplane_neon(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
  using T = typename P::T;
  constexpr int ps = (int)sizeof(T);
  constexpr int N = P::N;
  const auto ctx = P::make_ctx(bits_per_pixel);
  const int width = rowsize / ps;
  const int pitch = srcPitch / ps; // in pixels, for the border pixels

  pSrc += srcPitch;
  pDst += dstPitch;

  for (int y = 1; y < height / 2; ++y) {
    const T* s = reinterpret_cast<const T*>(pSrc);
    T* d = reinterpret_cast<T*>(pDst);
    d[0] = (s[pitch] + s[-pitch] + (P::is_float ? 0 : 1)) / 2; // float: no rounding
    P::store(pDst + ps, kernel(pSrc + ps, srcPitch, ctx));
    for (int x = N; x < width - N; x += N)
      P::store(pDst + x * ps, kernel(pSrc + x * ps, srcPitch, ctx));
    if (width - 1 - N >= 1) {
      const int x = width - 1 - N;
      P::store(pDst + x * ps, kernel(pSrc + x * ps, srcPitch, ctx));
    }
    d[width - 1] = (s[width - 1 + pitch] + s[width - 1 - pitch] + (P::is_float ? 0 : 1)) / 2;
    pSrc += srcPitch;
    pDst += dstPitch;

    rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); // the other field

    pSrc += srcPitch;
    pDst += dstPitch;
  }
}

template<class P, NeonKernel<P> kernel>
static void process_even_rows_neon(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); // the first two rows
  process_halfplane_neon<P, kernel>(pSrc + srcPitch, pDst + dstPitch, rowsize, height, srcPitch, dstPitch, bits_per_pixel);
  if (height & 1) // odd height: the last row is an even row without a row below it: border, copied
    rg_copy_rows(pDst + dstPitch * (height - 1), dstPitch, pSrc + srcPitch * (height - 1), srcPitch, rowsize, 1);
}

template<class P, NeonKernel<P> kernel>
static void process_odd_rows_neon(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); // top border
  // height + 1: with an odd height the last odd row (height-2) has both neighbours and is processed too
  process_halfplane_neon<P, kernel>(pSrc, pDst, rowsize, height + 1, srcPitch, dstPitch, bits_per_pixel);
  rg_copy_rows(pDst + dstPitch * (height - 1), dstPitch, pSrc + srcPitch * (height - 1), srcPitch, rowsize, 1); // bottom border
}

} // namespace

// mode -1..28; one table per pixel type (float: luma and chroma range)
#define RG_NEON_TABLE(P) { \
  rg_do_nothing, \
  rg_copy_plane, \
  process_plane_neon<P, rg_mode1_neon<P>>, \
  process_plane_neon<P, rg_mode2_neon<P>>, \
  process_plane_neon<P, rg_mode3_neon<P>>, \
  process_plane_neon<P, rg_mode4_neon<P>>, \
  process_plane_neon<P, rg_mode5_neon<P>>, \
  process_plane_neon<P, rg_mode6_neon<P>>, \
  process_plane_neon<P, rg_mode7_neon<P>>, \
  process_plane_neon<P, rg_mode8_neon<P>>, \
  process_plane_neon<P, rg_mode9_neon<P>>, \
  process_plane_neon<P, rg_mode10_neon<P>>, \
  process_plane_neon<P, rg_mode11_neon<P>>, \
  process_plane_neon<P, rg_mode12_neon<P>>, \
  process_even_rows_neon<P, rg_mode13_and14_neon<P>>, \
  process_odd_rows_neon<P, rg_mode13_and14_neon<P>>, \
  process_even_rows_neon<P, rg_mode15_and16_neon<P>>, \
  process_odd_rows_neon<P, rg_mode15_and16_neon<P>>, \
  process_plane_neon<P, rg_mode17_neon<P>>, \
  process_plane_neon<P, rg_mode18_neon<P>>, \
  process_plane_neon<P, rg_mode19_neon<P>>, \
  process_plane_neon<P, rg_mode20_neon<P>>, \
  process_plane_neon<P, rg_mode21_neon<P>>, \
  process_plane_neon<P, rg_mode22_neon<P>>, \
  process_plane_neon<P, rg_mode23_neon<P>>, \
  process_plane_neon<P, rg_mode24_neon<P>>, \
  process_plane_neon<P, rg_mode25_neon<P>>, \
  process_plane_neon<P, rg_mode26_neon<P>>, \
  process_plane_neon<P, rg_mode27_neon<P>>, \
  process_plane_neon<P, rg_mode28_neon<P>>, \
}

PlaneProcessor* rg_neon_functions[] = RG_NEON_TABLE(NeonU8);
// 9-16 bit (the max pixel value comes from bits_per_pixel)
PlaneProcessor* rg_neon_functions_16[] = RG_NEON_TABLE(NeonU16);
PlaneProcessor* rg_neon_functions_32_luma[] = RG_NEON_TABLE(NeonF32<false>);
PlaneProcessor* rg_neon_functions_32_chroma[] = RG_NEON_TABLE(NeonF32<true>);

#undef RG_NEON_TABLE

#endif // NEON_INTRINSICS
