#include "rg_functions_c.h"
#include "rg_functions_sse.h"
#include "removegrain.h"


template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_plane_sse(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

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

      if (mod_width != width) {
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

        if (mod_width != width) {
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
static void process_even_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_odd_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_plane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_halfplane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no round
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_even_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_c<pixel_t, processor>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_odd_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_c<pixel_t, processor>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

static void doNothing(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {

}

static void copyPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}


PlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, rg_mode1_sse<false, SSE2>, rg_mode1_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode2_sse<false, SSE2>, rg_mode2_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode3_sse<false, SSE2>, rg_mode3_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode4_sse<false, SSE2>, rg_mode4_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode5_sse<false, SSE2>, rg_mode5_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode6_sse<false, SSE2>, rg_mode6_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode7_sse<false, SSE2>, rg_mode7_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode8_sse<false, SSE2>, rg_mode8_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode9_sse<false, SSE2>, rg_mode9_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode10_sse<false, SSE2>, rg_mode10_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode11_sse<false, SSE2>, rg_mode11_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode12_sse<false, SSE2>, rg_mode12_sse<true, SSE2>>,
    process_even_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE2>, rg_mode13_and14_sse<true, SSE2>>,
    process_odd_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE2>, rg_mode13_and14_sse<true, SSE2>>,
    process_even_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE2>, rg_mode15_and16_sse<true, SSE2>>,
    process_odd_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE2>, rg_mode15_and16_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode17_sse<false, SSE2>, rg_mode17_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode18_sse<false, SSE2>, rg_mode18_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode19_sse<false, SSE2>, rg_mode19_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode20_sse<false, SSE2>, rg_mode20_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode21_sse<false, SSE2>, rg_mode21_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode22_sse<false, SSE2>, rg_mode22_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode23_sse<false, SSE2>, rg_mode23_sse<true, SSE2>>,
    process_plane_sse<uint8_t, rg_mode24_sse<false, SSE2>, rg_mode24_sse<true, SSE2>>,
};

PlaneProcessor* sse3_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, rg_mode1_sse<false, SSE3>, rg_mode1_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode2_sse<false, SSE3>, rg_mode2_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode3_sse<false, SSE3>, rg_mode3_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode4_sse<false, SSE3>, rg_mode4_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode5_sse<false, SSE3>, rg_mode5_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode6_sse<false, SSE3>, rg_mode6_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode7_sse<false, SSE3>, rg_mode7_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode8_sse<false, SSE3>, rg_mode8_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode9_sse<false, SSE3>, rg_mode9_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode10_sse<false, SSE3>, rg_mode10_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode11_sse<false, SSE3>, rg_mode11_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode12_sse<false, SSE3>, rg_mode12_sse<true, SSE3>>,
    process_even_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE3>, rg_mode13_and14_sse<true, SSE3>>,
    process_odd_rows_sse<uint8_t, rg_mode13_and14_sse<false, SSE3>, rg_mode13_and14_sse<true, SSE3>>,
    process_even_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE3>, rg_mode15_and16_sse<true, SSE3>>,
    process_odd_rows_sse<uint8_t, rg_mode15_and16_sse<false, SSE3>, rg_mode15_and16_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode17_sse<false, SSE3>, rg_mode17_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode18_sse<false, SSE3>, rg_mode18_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode19_sse<false, SSE3>, rg_mode19_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode20_sse<false, SSE3>, rg_mode20_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode21_sse<false, SSE3>, rg_mode21_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode22_sse<false, SSE3>, rg_mode22_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode23_sse<false, SSE3>, rg_mode23_sse<true, SSE3>>,
    process_plane_sse<uint8_t, rg_mode24_sse<false, SSE3>, rg_mode24_sse<true, SSE3>>,
};

PlaneProcessor* sse4_functions_16_10[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<10, false>, rg_mode6_sse_16<10, false>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<10, false>, rg_mode8_sse_16<10, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_12[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<12, false>, rg_mode6_sse_16<12, false>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<12, false>, rg_mode8_sse_16<12, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_14[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<14, false>, rg_mode6_sse_16<14, true>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<14, false>, rg_mode8_sse_16<14, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};

PlaneProcessor* sse4_functions_16_16[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, rg_mode1_sse_16<false>, rg_mode1_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode2_sse_16<false>, rg_mode2_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode3_sse_16<false>, rg_mode3_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode4_sse_16<false>, rg_mode4_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode5_sse_16<false>, rg_mode5_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode6_sse_16<16, false>, rg_mode6_sse_16<16, true>>,
  process_plane_sse<uint16_t, rg_mode7_sse_16<false>, rg_mode7_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode8_sse_16<16, false>, rg_mode8_sse_16<16, true>>,
  process_plane_sse<uint16_t, rg_mode9_sse_16<false>, rg_mode9_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode10_sse_16<false>, rg_mode10_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode11_sse_16<false>, rg_mode11_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode12_sse_16<false>, rg_mode12_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode13_and14_sse_16<false>, rg_mode13_and14_sse_16<true>>,
  process_even_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_odd_rows_sse<uint16_t, rg_mode15_and16_sse_16<false>, rg_mode15_and16_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode17_sse_16<false>, rg_mode17_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode18_sse_16<false>, rg_mode18_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode19_sse_16<false>, rg_mode19_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode20_sse_16<false>, rg_mode20_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode21_sse_16<false>, rg_mode21_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode22_sse_16<false>, rg_mode22_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode23_sse_16<false>, rg_mode23_sse_16<true>>,
  process_plane_sse<uint16_t, rg_mode24_sse_16<false>, rg_mode24_sse_16<true>>,
};



PlaneProcessor* sse4_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_sse<float, rg_mode1_sse_32<false>, rg_mode1_sse_32<true>>,
  process_plane_sse<float, rg_mode2_sse_32<false>, rg_mode2_sse_32<true>>,
  process_plane_sse<float, rg_mode3_sse_32<false>, rg_mode3_sse_32<true>>,
  process_plane_sse<float, rg_mode4_sse_32<false>, rg_mode4_sse_32<true>>,
  process_plane_sse<float, rg_mode5_sse_32<false>, rg_mode5_sse_32<true>>,
  process_plane_sse<float, rg_mode6_sse_32<false>, rg_mode6_sse_32<true>>,
  process_plane_sse<float, rg_mode7_sse_32<false>, rg_mode7_sse_32<true>>,
  process_plane_sse<float, rg_mode8_sse_32<false>, rg_mode8_sse_32<true>>,
  process_plane_sse<float, rg_mode9_sse_32<false>, rg_mode9_sse_32<true>>,
  process_plane_sse<float, rg_mode10_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse<float, rg_mode11_sse_32<false>, rg_mode10_sse_32<true>>,
  process_plane_sse<float, rg_mode12_sse_32<false>, rg_mode12_sse_32<true>>,
  process_even_rows_sse<float, rg_mode13_and14_sse_32<false>, rg_mode12_sse_32<true>>,
  process_odd_rows_sse<float, rg_mode13_and14_sse_32<false>, rg_mode13_and14_sse_32<true>>,
  process_even_rows_sse<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_odd_rows_sse<float, rg_mode15_and16_sse_32<false>, rg_mode15_and16_sse_32<true>>,
  process_plane_sse<float, rg_mode17_sse_32<false>, rg_mode17_sse_32<true>>,
  process_plane_sse<float, rg_mode18_sse_32<false>, rg_mode18_sse_32<true>>,
  process_plane_sse<float, rg_mode19_sse_32<false>, rg_mode19_sse_32<true>>,
  process_plane_sse<float, rg_mode20_sse_32<false>, rg_mode20_sse_32<true>>,
  process_plane_sse<float, rg_mode21_sse_32<false>, rg_mode21_sse_32<true>>,
  process_plane_sse<float, rg_mode22_sse_32<false>, rg_mode22_sse_32<true>>,
  process_plane_sse<float, rg_mode23_sse_32<false>, rg_mode23_sse_32<true>>,
  process_plane_sse<float, rg_mode24_sse_32<false>, rg_mode24_sse_32<true>>,
};


PlaneProcessor* c_functions[] = {
    doNothing,
    copyPlane,
    process_plane_c<uint8_t, rg_mode1_cpp>,
    process_plane_c<uint8_t, rg_mode2_cpp>,
    process_plane_c<uint8_t, rg_mode3_cpp>,
    process_plane_c<uint8_t, rg_mode4_cpp>,
    process_plane_c<uint8_t, rg_mode5_cpp>,
    process_plane_c<uint8_t, rg_mode6_cpp>,
    process_plane_c<uint8_t, rg_mode7_cpp>,
    process_plane_c<uint8_t, rg_mode8_cpp>,
    process_plane_c<uint8_t, rg_mode9_cpp>,
    process_plane_c<uint8_t, rg_mode10_cpp>,
    process_plane_c<uint8_t, rg_mode11_cpp>,
    process_plane_c<uint8_t, rg_mode12_cpp>,
    process_even_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_odd_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_even_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_odd_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_plane_c<uint8_t, rg_mode17_cpp>,
    process_plane_c<uint8_t, rg_mode18_cpp>,
    process_plane_c<uint8_t, rg_mode19_cpp>,
    process_plane_c<uint8_t, rg_mode20_cpp>,
    process_plane_c<uint8_t, rg_mode21_cpp>,
    process_plane_c<uint8_t, rg_mode22_cpp>,
    process_plane_c<uint8_t, rg_mode23_cpp>,
    process_plane_c<uint8_t, rg_mode24_cpp>
};

PlaneProcessor* c_functions_10[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_12[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_14[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};


PlaneProcessor* c_functions_16[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>
};

PlaneProcessor* c_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_c<float, rg_mode1_cpp_32>,
  process_plane_c<float, rg_mode2_cpp_32>,
  process_plane_c<float, rg_mode3_cpp_32>,
  process_plane_c<float, rg_mode4_cpp_32>,
  process_plane_c<float, rg_mode5_cpp_32>,
  process_plane_c<float, rg_mode6_cpp_32>,
  process_plane_c<float, rg_mode7_cpp_32>,
  process_plane_c<float, rg_mode8_cpp_32>,
  process_plane_c<float, rg_mode9_cpp_32>,
  process_plane_c<float, rg_mode10_cpp_32>,
  process_plane_c<float, rg_mode11_cpp_32>,
  process_plane_c<float, rg_mode12_cpp_32>,
  process_even_rows_c<float, rg_mode13_and14_cpp_32>,
  process_odd_rows_c<float, rg_mode13_and14_cpp_32>,
  process_even_rows_c<float, rg_mode15_and16_cpp_32>,
  process_odd_rows_c<float, rg_mode15_and16_cpp_32>,
  process_plane_c<float, rg_mode17_cpp_32>,
  process_plane_c<float, rg_mode18_cpp_32>,
  process_plane_c<float, rg_mode19_cpp_32>,
  process_plane_c<float, rg_mode20_cpp_32>,
  process_plane_c<float, rg_mode21_cpp_32>,
  process_plane_c<float, rg_mode22_cpp_32>,
  process_plane_c<float, rg_mode23_cpp_32>,
  process_plane_c<float, rg_mode24_cpp_32>
};


RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
    : GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("RemoveGrain works only with planar colorspaces");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 24 || modeU_ > 24 || modeV_ > 24) {
        env->ThrowError("RemoveGrain mode should be between -1 and 24!");
    }

    bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
      env->ThrowError("RemoveGrain: cannot specify U or V mode for planar RGB!");
    }

    //now change undefined mode value and EVERYTHING WILL BREAK
    if (modeU_ <= UNDEFINED_MODE) { 
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    if (pixelsize == 1) {
      functions = (env->GetCPUFlags() & CPUF_SSE3) ? sse3_functions
        : (env->GetCPUFlags() & CPUF_SSE2) ? sse2_functions
        : c_functions;

      if (vi.width < 17) { //not enough for XMM
        functions = c_functions;
      }
    }
    else if (pixelsize == 2) {
      if ((env->GetCPUFlags() & CPUF_SSE4) && vi.width >= (16/sizeof(uint16_t) + 1)) {
        // mode 6 and 8 bitdepth clamp specific
        switch (bits_per_pixel) {
        case 10: functions = sse4_functions_16_10; break;
        case 12: functions = sse4_functions_16_12; break;
        case 14: functions = sse4_functions_16_14; break;
        case 16: functions = sse4_functions_16_16; break;
        default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
        }
      }
      else {
        switch (bits_per_pixel) {
        case 10: functions = c_functions_10; break;
        case 12: functions = c_functions_12; break;
        case 14: functions = c_functions_14; break;
        case 16: functions = c_functions_16; break;
        default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
        }
      }
    }
    else {// if (pixelsize == 4) 
      if ((env->GetCPUFlags() & CPUF_SSE4) && vi.width >= (16/sizeof(float) + 1))
        functions = sse4_functions_32;
      else
        functions = c_functions_32;
    }
}


PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);
    
    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_G), dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetRowSize(PLANAR_G), 
        srcFrame->GetHeight(PLANAR_G), srcFrame->GetPitch(PLANAR_G), dstFrame->GetPitch(PLANAR_G));
      functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_B), dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetRowSize(PLANAR_B), 
        srcFrame->GetHeight(PLANAR_B), srcFrame->GetPitch(PLANAR_B), dstFrame->GetPitch(PLANAR_B));
      functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_R), dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetRowSize(PLANAR_R), 
        srcFrame->GetHeight(PLANAR_R), srcFrame->GetPitch(PLANAR_R), dstFrame->GetPitch(PLANAR_R));
    } else {
      functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y), 
        srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));

      if (vi.IsPlanar() && !vi.IsY()) {
        functions[modeU_ + 1](env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));

        functions[modeV_ + 1](env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    { // copy alpha
      env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
    }
    return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR };
    return new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrain::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrain::UNDEFINED_MODE), args[PLANAR].AsBool(false), env);
}
