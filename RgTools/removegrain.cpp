#include "removegrain.h"
#include "rg_functions_c.h"
#include "rg_functions_sse.h"


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

    if (mod_width != width) {
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

    if (mod_width != width) {
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
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_odd_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_sse<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_even_rows_sse41(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

  process_halfplane_sse41<pixel_t, processor, processor_a>(env, pSrc + srcPitch, pDst + dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_odd_rows_sse41(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

  process_halfplane_sse41<pixel_t, processor, processor_a>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

  env->BitBlt(pDst + dstPitch * (height - 1), dstPitch, pSrc + srcPitch * (height - 1), srcPitch, rowsize, 1); //bottom border
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


static PlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
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
};

static PlaneProcessor* sse4_functions[] = {
    doNothing,
    copyPlane,
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
};

static PlaneProcessor* sse4_functions_16_10[] = {
  doNothing,
  copyPlane,
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
};

static PlaneProcessor* sse4_functions_16_12[] = {
  doNothing,
  copyPlane,
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
};

static PlaneProcessor* sse4_functions_16_14[] = {
  doNothing,
  copyPlane,
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
};

static PlaneProcessor* sse4_functions_16_16[] = {
  doNothing,
  copyPlane,
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
};



static PlaneProcessor* sse4_functions_32_luma[] = {
  doNothing,
  copyPlane,
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
  process_plane_sse41<float, rg_mode25_sse_32<false, false>, rg_mode25_sse_32<true, false>> // 2nd: luma false, chroma true
};


static PlaneProcessor* sse4_functions_32_chroma[] = {
  doNothing,
  copyPlane,
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
  process_plane_sse41<float, rg_mode25_sse_32<false, true>, rg_mode25_sse_32<true, true>> // 2nd: luma false, chroma true
};


static PlaneProcessor* c_functions[] = {
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
    process_plane_c<uint8_t, rg_mode24_cpp>,
    process_plane_c<uint8_t, rg_mode25_cpp>
};

static PlaneProcessor* c_functions_10[] = {
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<10>>
};

static PlaneProcessor* c_functions_12[] = {
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<12>>
};

static PlaneProcessor* c_functions_14[] = {
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<14>>
};


static PlaneProcessor* c_functions_16[] = {
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<16>>
};

static PlaneProcessor* c_functions_32_luma[] = {
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
  process_plane_c<float, rg_mode23_cpp_32<false>>,
  process_plane_c<float, rg_mode24_cpp_32<false>>,
  process_plane_c<float, rg_mode25_cpp_32<false>> // false: luma, true: chroma
};

static PlaneProcessor* c_functions_32_chroma[] = {
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
  process_plane_c<float, rg_mode23_cpp_32<true>>,
  process_plane_c<float, rg_mode24_cpp_32<true>>,
  process_plane_c<float, rg_mode25_cpp_32<true>> // false: luma, true: chroma
};


extern PlaneProcessor* avx2_functions[];
extern PlaneProcessor* avx2_functions_16_10[];
extern PlaneProcessor* avx2_functions_16_12[];
extern PlaneProcessor* avx2_functions_16_14[];
extern PlaneProcessor* avx2_functions_16_16[];
extern PlaneProcessor* avx2_functions_32_luma[];
extern PlaneProcessor* avx2_functions_32_chroma[];

RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("RemoveGrain works only with planar colorspaces");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 25 || modeU_ > 25 || modeV_ > 25) {
        env->ThrowError("RemoveGrain mode should be between -1 and 25!");
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

    // 0: auto
    // 1: c
    // 2: sse2
    // 3: sse4.1
    // 4: avx2
    const bool use_avx2 = (opt == 0 || opt >= 4) && !!(env->GetCPUFlags() & CPUF_AVX2);
    const bool use_sse41 = (opt == 0 || opt >= 3) && !!(env->GetCPUFlags() & CPUF_SSE4_1);
    const bool use_sse2 = (opt == 0 || opt >= 2) && !!(env->GetCPUFlags() & CPUF_SSE2);

    functions_chroma = nullptr; // only for float

    if (pixelsize == 1) {
      if (use_avx2)
        functions = avx2_functions;
      else if (use_sse41)
        functions = sse4_functions;
      else if (use_sse2)
        functions = sse2_functions;
      else
        functions = c_functions;

      if (vi.width < 32 + 1 && use_avx2) { //not enough for YMM, try SSE4
        functions = sse4_functions;
      }
      if (vi.width < 16+1) { //not enough for XMM
        functions = c_functions;
      }
    }
    else if (pixelsize == 2) {
      if (use_avx2 && vi.width >= (32 / sizeof(uint16_t) + 1)) {
        // mode 6 and 8 bitdepth clamp specific
        switch (bits_per_pixel) {
        case 10: functions = avx2_functions_16_10; break;
        case 12: functions = avx2_functions_16_12; break;
        case 14: functions = avx2_functions_16_14; break;
        case 16: functions = avx2_functions_16_16; break;
        default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
        }
      }
      else if (use_sse41 && vi.width >= (16/sizeof(uint16_t) + 1)) {
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
      if (use_avx2 && vi.width >= (32 / sizeof(float) + 1)) {
        functions = avx2_functions_32_luma;
        functions_chroma = avx2_functions_32_chroma;
      }
      else if (use_sse41 && vi.width >= (16 / sizeof(float) + 1)) {
        functions = sse4_functions_32_luma;
        functions_chroma = sse4_functions_32_chroma;
      }
      else {
        functions = c_functions_32_luma;
        functions_chroma = c_functions_32_chroma;
      }
    }
}


PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto dstFrame = env->NewVideoFrame(vi);

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  // remark: no special alignment required for AVX2

  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
    for (int p = 0; p < 3; ++p) {
      const int plane = planes[p];

      functions[mode_ + 1](env, srcFrame->GetReadPtr(plane), dstFrame->GetWritePtr(plane), srcFrame->GetRowSize(plane),
        srcFrame->GetHeight(plane), srcFrame->GetPitch(plane), dstFrame->GetPitch(plane));
    }
  }
  else {
    if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_Y)))
      env->ThrowError("RemoveGrain: Invalid memory alignment. Unaligned crop?");

    functions[mode_ + 1](env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y),
      srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
      if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_U)))
        env->ThrowError("RemoveGrain: Invalid memory alignment. Unaligned crop?");

      if (functions_chroma != nullptr) {
        // for float
        functions_chroma[modeU_ + 1](env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        functions_chroma[modeV_ + 1](env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
      else {
        functions[modeU_ + 1](env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));

        functions[modeV_ + 1](env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
    }
  }
  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR, OPT };
    return new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrain::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrain::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), args[OPT].AsInt(0), env);
}
