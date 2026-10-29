// RemoveGrain: x86 SSE2 / SSE4.1 plane loops and function tables

#include "removegrain.h"

#ifdef INTEL_INTRINSICS

#include "rg_functions_sse.h" // kernels, in an anonymous namespace

namespace {

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_plane_sse(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);  // FIXME: change single row copy to memcpy (1st and last rows)

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);
    const int pixels_at_at_time = 16 / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height - 1; ++y) {
      pDst[0] = pSrc[0];

      // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
      __m128i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

      // aligned
      for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
        __m128i result = processor_a((uint8_t *)(pSrc + x), srcPitchOrig);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
      }
      
      if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
        __m128i result = processor((uint8_t *)(pSrc + width - 1 - pixels_at_at_time), srcPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + width - 1 - pixels_at_at_time), result);
      }
      
      pDst[width - 1] = pSrc[width - 1];

      pSrc += srcPitch;
      pDst += dstPitch;
    }

    env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_plane_sse41(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

  pixel_t* pDst = reinterpret_cast<pixel_t*>(pDst8);
  const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(pSrc8);

  dstPitch /= sizeof(pixel_t);
  const int srcPitchOrig = srcPitch;
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  int mod_width = width / pixels_at_at_time * pixels_at_at_time;

  for (int y = 1; y < height - 1; ++y) {
    pDst[0] = pSrc[0];

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i result = processor((uint8_t*)(pSrc + 1), srcPitchOrig);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

    // aligned
    for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
      __m128i result = processor_a((uint8_t*)(pSrc + x), srcPitchOrig);
      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
    }

    if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
      __m128i result = processor((uint8_t*)(pSrc + width - 1 - pixels_at_at_time), srcPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + width - 1 - pixels_at_at_time), result);
    }

    pDst[width - 1] = pSrc[width - 1];

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt((uint8_t*)(pDst), dstPitch * sizeof(pixel_t), (uint8_t*)(pSrc), srcPitch * sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_halfplane_sse(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
  pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
  const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

  dstPitch /= sizeof(pixel_t);
  const int srcPitchOrig = srcPitch;
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding

        // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
        __m128i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

        // aligned
        for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
          __m128i result = processor_a((uint8_t *)(pSrc + x), srcPitchOrig);
          _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
        }

        if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
          __m128i result = processor((uint8_t *)(pSrc+width-1-pixels_at_at_time), srcPitchOrig);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+width-1-pixels_at_at_time), result);
        }

        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_halfplane_sse41(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
  pixel_t* pDst = reinterpret_cast<pixel_t*>(pDst8);
  const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(pSrc8);

  dstPitch /= sizeof(pixel_t);
  const int srcPitchOrig = srcPitch;
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  int mod_width = width / pixels_at_at_time * pixels_at_at_time;

  for (int y = 1; y < height / 2; ++y) {
    pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i result = processor((uint8_t*)(pSrc + 1), srcPitchOrig);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

    // aligned
    for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
      __m128i result = processor_a((uint8_t*)(pSrc + x), srcPitchOrig);
      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
    }

    if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
      __m128i result = processor((uint8_t*)(pSrc + width - 1 - pixels_at_at_time), srcPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + width - 1 - pixels_at_at_time), result);
    }

    pDst[width - 1] = (pSrc[width - 1 + srcPitch] + pSrc[width - 1 - srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding
    pSrc += srcPitch;
    pDst += dstPitch;

    env->BitBlt((uint8_t*)(pDst), dstPitch * sizeof(pixel_t), (uint8_t*)(pSrc), srcPitch * sizeof(pixel_t), rowsize, 1); //other field

    pSrc += srcPitch;
    pDst += dstPitch;
  }
}


template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_even_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
    if (height & 1) // odd height: the last row is an even row without a row below it: border, copied
        env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_odd_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    // height + 1: with an odd height the last odd row (height-2) has both neighbours and is processed too
    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height + 1, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_even_rows_sse41(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

  process_halfplane_sse41<pixel_t, processor, processor_a>(env, pSrc + srcPitch, pDst + dstPitch, rowsize, height, srcPitch, dstPitch);
  if (height & 1) // odd height: the last row is an even row without a row below it: border, copied
    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_odd_rows_sse41(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

  // height + 1: with an odd height the last odd row (height-2) has both neighbours and is processed too
  process_halfplane_sse41<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height + 1, srcPitch, dstPitch);

  env->BitBlt(pDst + dstPitch * (height - 1), dstPitch, pSrc + srcPitch * (height - 1), srcPitch, rowsize, 1); //bottom border
}

} // namespace

PlaneProcessor* rg_sse2_functions[] = {
    rg_do_nothing,
    rg_copy_plane,
    process_plane_sse<uint8_t, rg_mode1_sse2<false>, rg_mode1_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode2_sse2<false>, rg_mode2_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode3_sse2<false>, rg_mode3_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode4_sse2<false>, rg_mode4_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode5_sse2<false>, rg_mode5_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode6_sse2<false>, rg_mode6_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode7_sse2<false>, rg_mode7_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode8_sse2<false>, rg_mode8_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode9_sse2<false>, rg_mode9_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode10_sse2<false>, rg_mode10_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode11_sse2<false>, rg_mode11_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode12_sse2<false>, rg_mode12_sse2<true>>,
    process_even_rows_sse<uint8_t, rg_mode13_and14_sse2<false>, rg_mode13_and14_sse2<true>>,
    process_odd_rows_sse<uint8_t, rg_mode13_and14_sse2<false>, rg_mode13_and14_sse2<true>>,
    process_even_rows_sse<uint8_t, rg_mode15_and16_sse2<false>, rg_mode15_and16_sse2<true>>,
    process_odd_rows_sse<uint8_t, rg_mode15_and16_sse2<false>, rg_mode15_and16_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode17_sse2<false>, rg_mode17_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode18_sse2<false>, rg_mode18_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode19_sse2<false>, rg_mode19_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode20_sse2<false>, rg_mode20_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode21_sse2<false>, rg_mode21_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode22_sse2<false>, rg_mode22_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode23_sse2<false>, rg_mode23_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode24_sse2<false>, rg_mode24_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode25_sse2<false>, rg_mode25_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode26_sse2<false>, rg_mode26_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode27_sse2<false>, rg_mode27_sse2<true>>,
    process_plane_sse<uint8_t, rg_mode28_sse2<false>, rg_mode28_sse2<true>>,
};

PlaneProcessor* rg_sse4_functions[] = {
    rg_do_nothing,
    rg_copy_plane,
    process_plane_sse41<uint8_t, rg_mode1_sse<false>, rg_mode1_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode2_sse<false>, rg_mode2_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode3_sse<false>, rg_mode3_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode4_sse<false>, rg_mode4_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode5_sse<false>, rg_mode5_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode6_sse<false>, rg_mode6_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode7_sse<false>, rg_mode7_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode8_sse<false>, rg_mode8_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode9_sse<false>, rg_mode9_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode10_sse<false>, rg_mode10_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode11_sse<false>, rg_mode11_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode12_sse<false>, rg_mode12_sse<true>>,
    process_even_rows_sse41<uint8_t, rg_mode13_and14_sse<false>, rg_mode13_and14_sse<true>>,
    process_odd_rows_sse41<uint8_t, rg_mode13_and14_sse<false>, rg_mode13_and14_sse<true>>,
    process_even_rows_sse41<uint8_t, rg_mode15_and16_sse<false>, rg_mode15_and16_sse<true>>,
    process_odd_rows_sse41<uint8_t, rg_mode15_and16_sse<false>, rg_mode15_and16_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode17_sse<false>, rg_mode17_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode18_sse<false>, rg_mode18_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode19_sse<false>, rg_mode19_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode20_sse<false>, rg_mode20_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode21_sse<false>, rg_mode21_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode22_sse<false>, rg_mode22_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode23_sse<false>, rg_mode23_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode24_sse<false>, rg_mode24_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode25_sse<false>, rg_mode25_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode26_sse<false>, rg_mode26_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode27_sse<false>, rg_mode27_sse<true>>,
    process_plane_sse41<uint8_t, rg_mode28_sse<false>, rg_mode28_sse<true>>,
};

PlaneProcessor* rg_sse4_functions_16_10[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode6_sse_16<10, false>, rg_mode6_sse_16<10, false>>,
  process_plane_sse41<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode8_sse_16<10, false>, rg_mode8_sse_16<10, true>>,
  process_plane_sse41<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode25_sse_16<10, false>, rg_mode25_sse_16<10, true>>,
  process_plane_sse41<uint16_t, rg_mode26_sse_16<false>, rg_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode27_sse_16<false>, rg_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode28_sse_16<false>, rg_mode28_sse_16<true>>,
};

PlaneProcessor* rg_sse4_functions_16_12[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode6_sse_16<12, false>, rg_mode6_sse_16<12, false>>,
  process_plane_sse41<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode8_sse_16<12, false>, rg_mode8_sse_16<12, true>>,
  process_plane_sse41<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode25_sse_16<12, false>, rg_mode25_sse_16<12, true>>,
  process_plane_sse41<uint16_t, rg_mode26_sse_16<false>, rg_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode27_sse_16<false>, rg_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode28_sse_16<false>, rg_mode28_sse_16<true>>,
};

PlaneProcessor* rg_sse4_functions_16_14[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode6_sse_16<14, false>, rg_mode6_sse_16<14, true>>,
  process_plane_sse41<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode8_sse_16<14, false>, rg_mode8_sse_16<14, true>>,
  process_plane_sse41<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode25_sse_16<14, false>, rg_mode25_sse_16<14, true>>,
  process_plane_sse41<uint16_t, rg_mode26_sse_16<false>, rg_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode27_sse_16<false>, rg_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode28_sse_16<false>, rg_mode28_sse_16<true>>,
};

PlaneProcessor* rg_sse4_functions_16_16[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode6_sse_16<16, false>, rg_mode6_sse_16<16, true>>,
  process_plane_sse41<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode8_sse_16<16, false>, rg_mode8_sse_16<16, true>>,
  process_plane_sse41<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse41<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode25_sse_16<16, false>, rg_mode25_sse_16<16, true>>,
  process_plane_sse41<uint16_t, rg_mode26_sse_16<false>, rg_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode27_sse_16<false>, rg_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, rg_mode28_sse_16<false>, rg_mode28_sse_16<true>>,
};



PlaneProcessor* rg_sse4_functions_32_luma[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<float, rg_mode1_sse_32<false>, rg_mode1_sse_32<true>>,
  process_plane_sse41<float, rg_mode2_sse_32<false>, rg_mode2_sse_32<true>>,
  process_plane_sse41<float, rg_mode3_sse_32<false>, rg_mode3_sse_32<true>>,
  process_plane_sse41<float, rg_mode4_sse_32<false>, rg_mode4_sse_32<true>>,
  process_plane_sse41<float, rg_mode5_sse_32<false>, rg_mode5_sse_32<true>>,
  process_plane_sse41<float, rg_mode6_sse_32<false>, rg_mode6_sse_32<true>>,
  process_plane_sse41<float, rg_mode7_sse_32<false>, rg_mode7_sse_32<true>>,
  process_plane_sse41<float, rg_mode8_sse_32<false>, rg_mode8_sse_32<true>>,
  process_plane_sse41<float, rg_mode9_sse_32<false>, rg_mode9_sse_32<true>>,
  process_plane_sse41<float, rg_mode10_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse41<float, rg_mode11_sse_32<false>, rg_mode11_sse_32<true>>,
  process_plane_sse41<float, rg_mode12_sse_32<false>, rg_mode12_sse_32<true>>,
  process_even_rows_sse41<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_odd_rows_sse41<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_even_rows_sse41<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_odd_rows_sse41<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_plane_sse41<float, rg_mode17_sse_32<false>, rg_mode17_sse_32<true>>,
  process_plane_sse41<float, rg_mode18_sse_32<false>, rg_mode18_sse_32<true>>,
  process_plane_sse41<float, rg_mode19_sse_32<false>, rg_mode19_sse_32<true>>,
  process_plane_sse41<float, rg_mode20_sse_32<false>, rg_mode20_sse_32<true>>,
  process_plane_sse41<float, rg_mode21_sse_32<false>, rg_mode21_sse_32<true>>,
  process_plane_sse41<float, rg_mode22_sse_32<false>, rg_mode22_sse_32<true>>,
  process_plane_sse41<float, rg_mode23_sse_32<false, false>, rg_mode23_sse_32<true, false>>,
  process_plane_sse41<float, rg_mode24_sse_32<false, false>, rg_mode24_sse_32<true, false>>,
  process_plane_sse41<float, rg_mode25_sse_32<false, false>, rg_mode25_sse_32<true, false>>, // 2nd: luma false, chroma true
  process_plane_sse41<float, rg_mode26_sse_32<false>, rg_mode26_sse_32<true>>,
  process_plane_sse41<float, rg_mode27_sse_32<false>, rg_mode27_sse_32<true>>,
  process_plane_sse41<float, rg_mode28_sse_32<false>, rg_mode28_sse_32<true>>,
};


PlaneProcessor* rg_sse4_functions_32_chroma[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_sse41<float, rg_mode1_sse_32<false>, rg_mode1_sse_32<true>>,
  process_plane_sse41<float, rg_mode2_sse_32<false>, rg_mode2_sse_32<true>>,
  process_plane_sse41<float, rg_mode3_sse_32<false>, rg_mode3_sse_32<true>>,
  process_plane_sse41<float, rg_mode4_sse_32<false>, rg_mode4_sse_32<true>>,
  process_plane_sse41<float, rg_mode5_sse_32<false>, rg_mode5_sse_32<true>>,
  process_plane_sse41<float, rg_mode6_sse_32<false>, rg_mode6_sse_32<true>>,
  process_plane_sse41<float, rg_mode7_sse_32<false>, rg_mode7_sse_32<true>>,
  process_plane_sse41<float, rg_mode8_sse_32<false>, rg_mode8_sse_32<true>>,
  process_plane_sse41<float, rg_mode9_sse_32<false>, rg_mode9_sse_32<true>>,
  process_plane_sse41<float, rg_mode10_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse41<float, rg_mode11_sse_32<false>, rg_mode11_sse_32<true>>,
  process_plane_sse41<float, rg_mode12_sse_32<false>, rg_mode12_sse_32<true>>,
  process_even_rows_sse41<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_odd_rows_sse41<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_even_rows_sse41<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_odd_rows_sse41<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_plane_sse41<float, rg_mode17_sse_32<false>, rg_mode17_sse_32<true>>,
  process_plane_sse41<float, rg_mode18_sse_32<false>, rg_mode18_sse_32<true>>,
  process_plane_sse41<float, rg_mode19_sse_32<false>, rg_mode19_sse_32<true>>,
  process_plane_sse41<float, rg_mode20_sse_32<false>, rg_mode20_sse_32<true>>,
  process_plane_sse41<float, rg_mode21_sse_32<false>, rg_mode21_sse_32<true>>,
  process_plane_sse41<float, rg_mode22_sse_32<false>, rg_mode22_sse_32<true>>,
  process_plane_sse41<float, rg_mode23_sse_32<false, true>, rg_mode23_sse_32<true, true>>,
  process_plane_sse41<float, rg_mode24_sse_32<false, true>, rg_mode24_sse_32<true, true>>,
  process_plane_sse41<float, rg_mode25_sse_32<false, true>, rg_mode25_sse_32<true, true>>, // 2nd: luma false, chroma true
  process_plane_sse41<float, rg_mode26_sse_32<false>, rg_mode26_sse_32<true>>,
  process_plane_sse41<float, rg_mode27_sse_32<false>, rg_mode27_sse_32<true>>,
  process_plane_sse41<float, rg_mode28_sse_32<false>, rg_mode28_sse_32<true>>,
};


#endif // INTEL_INTRINSICS
