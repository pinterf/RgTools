#include "repair.h"
#include "repair_functions_c.h"
#include "repair_functions_sse.h"

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
static void process_plane_sse(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1); // FIXME: change single row copy to memcpy (top and bottom rows)

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);
    const pixel_t *pRef = reinterpret_cast<const pixel_t *>(pRef8);

    dstPitch /= sizeof(pixel_t);
    const int refPitchOrig = refPitch;
    refPitch /= sizeof(pixel_t);
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);
    const int pixels_at_at_time = 16 / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];

        // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
        __m128i val = simd_loadu_si128((uint8_t *)(pSrc+1));
        __m128i result = processor((uint8_t *)(pRef+1), val, refPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+1), result);

        //aligned
        for (int x = pixels_at_at_time; x < mod_width-1; x+= pixels_at_at_time) {
            __m128i val = simd_loada_si128((uint8_t *)(pSrc+x));
            __m128i result = processor_a((uint8_t *)(pRef+x), val, refPitchOrig);
            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod_width != width) {
            __m128i val = simd_loadu_si128((uint8_t *)(pSrc + width - 1 - pixels_at_at_time));
            __m128i result = processor((uint8_t *)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig);
            _mm_storeu_si128(reinterpret_cast<__m128i*>((uint8_t *)(pDst + width - 1 - pixels_at_at_time)), result);
        }


        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
        pRef += refPitch;
    }

    env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1);  // FIXME: change single row copy to memcpy
}

#if 0
template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse3")))
#endif
static void process_plane_sse3(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

  pixel_t* pDst = reinterpret_cast<pixel_t*>(pDst8);
  const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(pSrc8);
  const pixel_t* pRef = reinterpret_cast<const pixel_t*>(pRef8);

  dstPitch /= sizeof(pixel_t);
  const int refPitchOrig = refPitch;
  refPitch /= sizeof(pixel_t);
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  int mod_width = width / pixels_at_at_time * pixels_at_at_time;

  for (int y = 1; y < height - 1; ++y) {
    pDst[0] = pSrc[0];

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + 1));
    __m128i result = processor((uint8_t*)(pRef + 1), val, refPitchOrig);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

    //aligned
    for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + x));
      __m128i result = processor_a((uint8_t*)(pRef + x), val, refPitchOrig);
      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
    }

    if (mod_width != width) {
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + width - 1 - pixels_at_at_time));
      __m128i result = processor((uint8_t*)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>((uint8_t*)(pDst + width - 1 - pixels_at_at_time)), result);
    }


    pDst[width - 1] = pSrc[width - 1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt((uint8_t*)(pDst), dstPitch * sizeof(pixel_t), (uint8_t*)(pSrc), srcPitch * sizeof(pixel_t), rowsize, 1);
}
#endif

template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_plane_sse41(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

  pixel_t* pDst = reinterpret_cast<pixel_t*>(pDst8);
  const pixel_t* pSrc = reinterpret_cast<const pixel_t*>(pSrc8);
  const pixel_t* pRef = reinterpret_cast<const pixel_t*>(pRef8);

  dstPitch /= sizeof(pixel_t);
  const int refPitchOrig = refPitch;
  refPitch /= sizeof(pixel_t);
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 16 / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  int mod_width = width / pixels_at_at_time * pixels_at_at_time;

  for (int y = 1; y < height - 1; ++y) {
    pDst[0] = pSrc[0];

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + 1));
    __m128i result = processor((uint8_t*)(pRef + 1), val, refPitchOrig);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

    //aligned
    for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + x));
      __m128i result = processor_a((uint8_t*)(pRef + x), val, refPitchOrig);
      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
    }

    if (mod_width != width) {
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + width - 1 - pixels_at_at_time));
      __m128i result = processor((uint8_t*)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>((uint8_t*)(pDst + width - 1 - pixels_at_at_time)), result);
    }


    pDst[width - 1] = pSrc[width - 1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt((uint8_t*)(pDst), dstPitch * sizeof(pixel_t), (uint8_t*)(pSrc), srcPitch * sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_plane_c(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); 

  const int width = rowsize / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  for (int y = 1; y < height-1; ++y) {
    reinterpret_cast<pixel_t *>(pDst)[0] = reinterpret_cast<const pixel_t *>(pSrc)[0];
    for (int x = 1; x < width-1; x+=1) {
      pixel_t result = processor(pRef + x*sizeof(pixel_t), reinterpret_cast<const pixel_t *>(pSrc)[x], srcPitch);
      reinterpret_cast<pixel_t *>(pDst)[x] = result;
    }
    reinterpret_cast<pixel_t *>(pDst)[width-1] = reinterpret_cast<const pixel_t *>(pSrc)[width-1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}


static void doNothing(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {

}

static void copyPlane(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

static RepairPlaneProcessor* sse4_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse41<uint8_t, repair_mode1_sse<false>, repair_mode1_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode2_sse<false>, repair_mode2_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode3_sse<false>, repair_mode3_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode4_sse<false>, repair_mode4_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode5_sse<false>, repair_mode5_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode6_sse<false>, repair_mode6_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode7_sse<false>, repair_mode7_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode8_sse<false>, repair_mode8_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode9_sse<false>, repair_mode9_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode10_sse<false>, repair_mode10_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode1_sse<false>, repair_mode1_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode12_sse<false>, repair_mode12_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode13_sse<false>, repair_mode13_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode14_sse<false>, repair_mode14_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode15_sse<false>, repair_mode15_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode16_sse<false>, repair_mode16_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode17_sse<false>, repair_mode17_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode18_sse<false>, repair_mode18_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode19_sse<false>, repair_mode19_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode20_sse<false>, repair_mode20_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode21_sse<false>, repair_mode21_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode22_sse<false>, repair_mode22_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode23_sse<false>, repair_mode23_sse<true>>, 
    process_plane_sse41<uint8_t, repair_mode24_sse<false>, repair_mode24_sse<true>>,
    doNothing,
    process_plane_sse41<uint8_t, repair_mode26_sse<false>, repair_mode26_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode27_sse<false>, repair_mode27_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode28_sse<false>, repair_mode28_sse<true>>,
};

static RepairPlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, repair_mode1_sse2<false>, repair_mode1_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode2_sse2<false>, repair_mode2_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode3_sse2<false>, repair_mode3_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode4_sse2<false>, repair_mode4_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode5_sse2<false>, repair_mode5_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode6_sse2<false>, repair_mode6_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode7_sse2<false>, repair_mode7_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode8_sse2<false>, repair_mode8_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode9_sse2<false>, repair_mode9_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode10_sse2<false>, repair_mode10_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode1_sse2<false>, repair_mode1_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode12_sse2<false>, repair_mode12_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode13_sse2<false>, repair_mode13_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode14_sse2<false>, repair_mode14_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode15_sse2<false>, repair_mode15_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode16_sse2<false>, repair_mode16_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode17_sse2<false>, repair_mode17_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode18_sse2<false>, repair_mode18_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode19_sse2<false>, repair_mode19_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode20_sse2<false>, repair_mode20_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode21_sse2<false>, repair_mode21_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode22_sse2<false>, repair_mode22_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode23_sse2<false>, repair_mode23_sse2<true>>, 
    process_plane_sse<uint8_t, repair_mode24_sse2<false>, repair_mode24_sse2<true>>,
    doNothing,
    process_plane_sse<uint8_t, repair_mode26_sse2<false>, repair_mode26_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode27_sse2<false>, repair_mode27_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode28_sse2<false>, repair_mode28_sse2<true>>,
};

static RepairPlaneProcessor* sse4_functions_16_10[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>, repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode2_sse_16<false>, repair_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode3_sse_16<false>, repair_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode4_sse_16<false>, repair_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode5_sse_16<false>, repair_mode5_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode6_sse_16<10, false>, repair_mode6_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode7_sse_16<false>, repair_mode7_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode8_sse_16<10, false>, repair_mode8_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode9_sse_16<false>, repair_mode9_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode10_sse_16<false>, repair_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>, repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode12_sse_16<false>, repair_mode12_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode13_sse_16<false>, repair_mode13_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode14_sse_16<false>, repair_mode14_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode15_sse_16<false>, repair_mode15_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode16_sse_16<10, false>, repair_mode16_sse_16<10, true>>,
  process_plane_sse41<uint16_t, repair_mode17_sse_16<false>, repair_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode18_sse_16<false>, repair_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode19_sse_16<10, false>, repair_mode19_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode20_sse_16<10, false>, repair_mode20_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode21_sse_16<10, false>, repair_mode21_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode22_sse_16<10, false>, repair_mode22_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode23_sse_16<10, false>, repair_mode23_sse_16<10, true>>, 
  process_plane_sse41<uint16_t, repair_mode24_sse_16<10, false>, repair_mode24_sse_16<10, true>>,
  doNothing,
  process_plane_sse41<uint16_t, repair_mode26_sse_16<false>,  repair_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode27_sse_16<false>,  repair_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode28_sse_16<false>,  repair_mode28_sse_16<true>>,
};

static RepairPlaneProcessor* sse4_functions_16_12[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode6_sse_16<12, false>,  repair_mode6_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode8_sse_16<12, false>,  repair_mode8_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode16_sse_16<12, false>,  repair_mode16_sse_16<12, true>>,
  process_plane_sse41<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode19_sse_16<12, false>,  repair_mode19_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode20_sse_16<12, false>,  repair_mode20_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode21_sse_16<12, false>,  repair_mode21_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode22_sse_16<12, false>,  repair_mode22_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode23_sse_16<12, false>,  repair_mode23_sse_16<12, true>>, 
  process_plane_sse41<uint16_t, repair_mode24_sse_16<12, false>,  repair_mode24_sse_16<12, true>>,
  doNothing,
  process_plane_sse41<uint16_t, repair_mode26_sse_16<false>,  repair_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode27_sse_16<false>,  repair_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode28_sse_16<false>,  repair_mode28_sse_16<true>>,
};

static RepairPlaneProcessor* sse4_functions_16_14[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode6_sse_16<14, false>,  repair_mode6_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode8_sse_16<14, false>,  repair_mode8_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode16_sse_16<14, false>,  repair_mode16_sse_16<14, true>>,
  process_plane_sse41<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode19_sse_16<14, false>,  repair_mode19_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode20_sse_16<14, false>,  repair_mode20_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode21_sse_16<14, false>,  repair_mode21_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode22_sse_16<14, false>,  repair_mode22_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode23_sse_16<14, false>,  repair_mode23_sse_16<14, true>>, 
  process_plane_sse41<uint16_t, repair_mode24_sse_16<14, false>,  repair_mode24_sse_16<14, true>>,
  doNothing,
  process_plane_sse41<uint16_t, repair_mode26_sse_16<false>,  repair_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode27_sse_16<false>,  repair_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode28_sse_16<false>,  repair_mode28_sse_16<true>>,
};

static RepairPlaneProcessor* sse4_functions_16_16[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode6_sse_16<16, false>,  repair_mode6_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode8_sse_16<16, false>,  repair_mode8_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode16_sse_16<16, false>,  repair_mode16_sse_16<16, true>>,
  process_plane_sse41<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode19_sse_16<16, false>,  repair_mode19_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode20_sse_16<16, false>,  repair_mode20_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode21_sse_16<16, false>,  repair_mode21_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode22_sse_16<16, false>,  repair_mode22_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode23_sse_16<16, false>,  repair_mode23_sse_16<16, true>>, 
  process_plane_sse41<uint16_t, repair_mode24_sse_16<16, false>,  repair_mode24_sse_16<16, true>>,
  doNothing,
  process_plane_sse41<uint16_t, repair_mode26_sse_16<false>,  repair_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode27_sse_16<false>,  repair_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode28_sse_16<false>,  repair_mode28_sse_16<true>>,
};


static RepairPlaneProcessor* sse4_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>>,
  process_plane_sse41<float, repair_mode2_sse_32<false>, repair_mode2_sse_32<true>>,
  process_plane_sse41<float, repair_mode3_sse_32<false>, repair_mode3_sse_32<true>>,
  process_plane_sse41<float, repair_mode4_sse_32<false>, repair_mode4_sse_32<true>>,
  process_plane_sse41<float, repair_mode5_sse_32<false>, repair_mode5_sse_32<true>>, 
  process_plane_sse41<float, repair_mode6_sse_32<false>, repair_mode6_sse_32<true>>, 
  process_plane_sse41<float, repair_mode7_sse_32<false>, repair_mode7_sse_32<true>>, 
  process_plane_sse41<float, repair_mode8_sse_32<false>, repair_mode8_sse_32<true>>, 
  process_plane_sse41<float, repair_mode9_sse_32<false>, repair_mode9_sse_32<true>>, 
  process_plane_sse41<float, repair_mode10_sse_32<false>, repair_mode10_sse_32<true>>,
  process_plane_sse41<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>>,
  process_plane_sse41<float, repair_mode12_sse_32<false>, repair_mode12_sse_32<true>>,
  process_plane_sse41<float, repair_mode13_sse_32<false>, repair_mode13_sse_32<true>>,
  process_plane_sse41<float, repair_mode14_sse_32<false>, repair_mode14_sse_32<true>>,
  process_plane_sse41<float, repair_mode15_sse_32<false>, repair_mode15_sse_32<true>>,
  process_plane_sse41<float, repair_mode16_sse_32<false>, repair_mode16_sse_32<true>>,
  process_plane_sse41<float, repair_mode17_sse_32<false>, repair_mode17_sse_32<true>>,
  process_plane_sse41<float, repair_mode18_sse_32<false>, repair_mode18_sse_32<true>>,
  process_plane_sse41<float, repair_mode19_sse_32<false, false>, repair_mode19_sse_32<true, false>>, // 2nd template param: luma false chroma true
  process_plane_sse41<float, repair_mode20_sse_32<false, false>, repair_mode20_sse_32<true, false>>, 
  process_plane_sse41<float, repair_mode21_sse_32<false, false>, repair_mode21_sse_32<true, false>>, 
  process_plane_sse41<float, repair_mode22_sse_32<false, false>, repair_mode22_sse_32<true, false>>, 
  process_plane_sse41<float, repair_mode23_sse_32<false, false>, repair_mode23_sse_32<true, false>>, 
  process_plane_sse41<float, repair_mode24_sse_32<false, false>, repair_mode24_sse_32<true, false>>,
  doNothing,
  process_plane_sse41<float, repair_mode26_sse_32<false>, repair_mode26_sse_32<true>>,
  process_plane_sse41<float, repair_mode27_sse_32<false>, repair_mode27_sse_32<true>>,
  process_plane_sse41<float, repair_mode28_sse_32<false>, repair_mode28_sse_32<true>>
};

static RepairPlaneProcessor* sse4_functions_32_chroma[] = {
  doNothing,
  copyPlane,
  process_plane_sse41<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>>,
  process_plane_sse41<float, repair_mode2_sse_32<false>, repair_mode2_sse_32<true>>,
  process_plane_sse41<float, repair_mode3_sse_32<false>, repair_mode3_sse_32<true>>,
  process_plane_sse41<float, repair_mode4_sse_32<false>, repair_mode4_sse_32<true>>,
  process_plane_sse41<float, repair_mode5_sse_32<false>, repair_mode5_sse_32<true>>,
  process_plane_sse41<float, repair_mode6_sse_32<false>, repair_mode6_sse_32<true>>,
  process_plane_sse41<float, repair_mode7_sse_32<false>, repair_mode7_sse_32<true>>,
  process_plane_sse41<float, repair_mode8_sse_32<false>, repair_mode8_sse_32<true>>,
  process_plane_sse41<float, repair_mode9_sse_32<false>, repair_mode9_sse_32<true>>,
  process_plane_sse41<float, repair_mode10_sse_32<false>, repair_mode10_sse_32<true>>,
  process_plane_sse41<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>>,
  process_plane_sse41<float, repair_mode12_sse_32<false>, repair_mode12_sse_32<true>>,
  process_plane_sse41<float, repair_mode13_sse_32<false>, repair_mode13_sse_32<true>>,
  process_plane_sse41<float, repair_mode14_sse_32<false>, repair_mode14_sse_32<true>>,
  process_plane_sse41<float, repair_mode15_sse_32<false>, repair_mode15_sse_32<true>>,
  process_plane_sse41<float, repair_mode16_sse_32<false>, repair_mode16_sse_32<true>>,
  process_plane_sse41<float, repair_mode17_sse_32<false>, repair_mode17_sse_32<true>>,
  process_plane_sse41<float, repair_mode18_sse_32<false>, repair_mode18_sse_32<true>>,
  process_plane_sse41<float, repair_mode19_sse_32<false, true>, repair_mode19_sse_32<true, true>>,  // 2nd template param: luma false chroma true
  process_plane_sse41<float, repair_mode20_sse_32<false, true>, repair_mode20_sse_32<true, true>>,
  process_plane_sse41<float, repair_mode21_sse_32<false, true>, repair_mode21_sse_32<true, true>>,
  process_plane_sse41<float, repair_mode22_sse_32<false, true>, repair_mode22_sse_32<true, true>>,
  process_plane_sse41<float, repair_mode23_sse_32<false, true>, repair_mode23_sse_32<true, true>>,
  process_plane_sse41<float, repair_mode24_sse_32<false, true>, repair_mode24_sse_32<true, true>>,
  doNothing,
  process_plane_sse41<float, repair_mode26_sse_32<false>, repair_mode26_sse_32<true>>,
  process_plane_sse41<float, repair_mode27_sse_32<false>, repair_mode27_sse_32<true>>,
  process_plane_sse41<float, repair_mode28_sse_32<false>, repair_mode28_sse_32<true>>
};

static RepairPlaneProcessor* c_functions[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint8_t,repair_mode1_cpp>,
  process_plane_c<uint8_t,repair_mode2_cpp>,
  process_plane_c<uint8_t,repair_mode3_cpp>,
  process_plane_c<uint8_t,repair_mode4_cpp>,
  process_plane_c<uint8_t,repair_mode5_cpp>, 
  process_plane_c<uint8_t,repair_mode6_cpp>, 
  process_plane_c<uint8_t,repair_mode7_cpp>, 
  process_plane_c<uint8_t,repair_mode8_cpp>, 
  process_plane_c<uint8_t,repair_mode9_cpp>, 
  process_plane_c<uint8_t,repair_mode10_cpp>,
  process_plane_c<uint8_t,repair_mode1_cpp>,
  process_plane_c<uint8_t,repair_mode12_cpp>,
  process_plane_c<uint8_t,repair_mode13_cpp>,
  process_plane_c<uint8_t,repair_mode14_cpp>,
  process_plane_c<uint8_t,repair_mode15_cpp>,
  process_plane_c<uint8_t,repair_mode16_cpp>,
  process_plane_c<uint8_t,repair_mode17_cpp>,
  process_plane_c<uint8_t,repair_mode18_cpp>,
  process_plane_c<uint8_t,repair_mode19_cpp>, 
  process_plane_c<uint8_t,repair_mode20_cpp>, 
  process_plane_c<uint8_t,repair_mode21_cpp>, 
  process_plane_c<uint8_t,repair_mode22_cpp>, 
  process_plane_c<uint8_t,repair_mode23_cpp>, 
  process_plane_c<uint8_t,repair_mode24_cpp>,
  doNothing,
  process_plane_c<uint8_t,repair_mode26_cpp>,
  process_plane_c<uint8_t,repair_mode27_cpp>,
  process_plane_c<uint8_t,repair_mode28_cpp>
};

static RepairPlaneProcessor* c_functions_10[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<10>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<10>>,
  doNothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_12[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<12>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<12>>,
  doNothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_14[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<14>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<14>>,
  doNothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_16[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<16>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<16>>,
  doNothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode2_cpp_32>,
  process_plane_c<float,repair_mode3_cpp_32>,
  process_plane_c<float,repair_mode4_cpp_32>,
  process_plane_c<float,repair_mode5_cpp_32>, 
  process_plane_c<float,repair_mode6_cpp_32>, 
  process_plane_c<float,repair_mode7_cpp_32>, 
  process_plane_c<float,repair_mode8_cpp_32>, 
  process_plane_c<float,repair_mode9_cpp_32>, 
  process_plane_c<float,repair_mode10_cpp_32>,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode12_cpp_32>,
  process_plane_c<float,repair_mode13_cpp_32>,
  process_plane_c<float,repair_mode14_cpp_32>,
  process_plane_c<float,repair_mode15_cpp_32>,
  process_plane_c<float,repair_mode16_cpp_32>,
  process_plane_c<float,repair_mode17_cpp_32>,
  process_plane_c<float,repair_mode18_cpp_32>,
  process_plane_c<float,repair_mode19_cpp_32<false>>,
  process_plane_c<float,repair_mode20_cpp_32<false>>,
  process_plane_c<float,repair_mode21_cpp_32<false>>,
  process_plane_c<float,repair_mode22_cpp_32<false>>,
  process_plane_c<float,repair_mode23_cpp_32<false>>,
  process_plane_c<float,repair_mode24_cpp_32<false>>,
  doNothing,
  process_plane_c<float,repair_mode26_cpp_32>,
  process_plane_c<float,repair_mode27_cpp_32>,
  process_plane_c<float,repair_mode28_cpp_32>
};

static RepairPlaneProcessor* c_functions_32_chroma[] = {
  doNothing,
  copyPlane,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode2_cpp_32>,
  process_plane_c<float,repair_mode3_cpp_32>,
  process_plane_c<float,repair_mode4_cpp_32>,
  process_plane_c<float,repair_mode5_cpp_32>,
  process_plane_c<float,repair_mode6_cpp_32>,
  process_plane_c<float,repair_mode7_cpp_32>,
  process_plane_c<float,repair_mode8_cpp_32>,
  process_plane_c<float,repair_mode9_cpp_32>,
  process_plane_c<float,repair_mode10_cpp_32>,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode12_cpp_32>,
  process_plane_c<float,repair_mode13_cpp_32>,
  process_plane_c<float,repair_mode14_cpp_32>,
  process_plane_c<float,repair_mode15_cpp_32>,
  process_plane_c<float,repair_mode16_cpp_32>,
  process_plane_c<float,repair_mode17_cpp_32>,
  process_plane_c<float,repair_mode18_cpp_32>,
  process_plane_c<float,repair_mode19_cpp_32<true>>,
  process_plane_c<float,repair_mode20_cpp_32<true>>,
  process_plane_c<float,repair_mode21_cpp_32<true>>,
  process_plane_c<float,repair_mode22_cpp_32<true>>,
  process_plane_c<float,repair_mode23_cpp_32<true>>,
  process_plane_c<float,repair_mode24_cpp_32<true>>,
  doNothing,
  process_plane_c<float,repair_mode26_cpp_32>,
  process_plane_c<float,repair_mode27_cpp_32>,
  process_plane_c<float,repair_mode28_cpp_32>
};

Repair::Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
  : GenericVideoFilter(child), ref_(ref), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr), functions_chroma(nullptr) {

  auto refVi = ref_->GetVideoInfo();

  if (!(vi.IsPlanar() || skip_cs_check)) {
    env->ThrowError("Repair works only with planar colorspaces");
  }

  if (vi.width != refVi.width || vi.height != refVi.height) {
    env->ThrowError("Clips should be of the same size!");
  }

  if (mode <= UNDEFINED_MODE || mode_ > 28 || modeU_ > 28 || modeV_ > 28 ||
    mode_ == 25 || modeU_ == 25 || modeV_ == 25 // mode 25 exists only in RemoveGrain
    ) {
    env->ThrowError("Repair mode should be between -1 and 28, except 25!");
  }

  bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
    env->ThrowError("Repair: cannot specify U or V mode for planar RGB!");
  }

  //now change undefined mode value and EVERYTHING WILL BREAK
  if (modeU_ <= UNDEFINED_MODE) {
    modeU_ = mode_;
  }
  if (modeV_ <= UNDEFINED_MODE) {
    modeV_ = modeU_;
  }

  if (vi.IsPlanar() && !vi.IsY() && (modeU_ != -1 || modeV_ != -1)) {
    if (!vi.IsSameColorspace(refVi)) {
      env->ThrowError("Both clips should have the same colorspace!");
    }
  }

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  int worst_case_width = vi.width;
  if (vi.IsYUV() && vi.NumComponents() >= 3)
    worst_case_width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);

  // 0: auto
  // 1: c
  // 2: sse2
  // 3: sse4.1
  // 4: avx2
  //const bool use_avx2 = (opt == 0 || opt >= 4) && !!(env->GetCPUFlags() & CPUF_AVX2);
  const bool use_sse41 = (opt == 0 || opt >= 3) && !!(env->GetCPUFlags() & CPUF_SSE4_1);
  const bool use_sse2 = (opt == 0 || opt >= 2) && !!(env->GetCPUFlags() & CPUF_SSE2);

  // no avx2 here

  if (pixelsize == 1) {

    functions = 
      use_sse41 ? sse4_functions :
      use_sse2 ? sse2_functions : c_functions;

    if (worst_case_width < 17) { //not enough for XMM
      functions = c_functions;
    }
  }
  else if (pixelsize == 2) {
    if (use_sse41 && worst_case_width >= (16 / sizeof(uint16_t) + 1)) {
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

    if (use_sse41 && worst_case_width >= (16 / sizeof(float) + 1)) {
      functions = sse4_functions_32;
      functions_chroma = sse4_functions_32_chroma;
    }
    else {
      functions = c_functions_32;
      functions_chroma = c_functions_32_chroma;
    }
  }
}


PVideoFrame Repair::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto refFrame = ref_->GetFrame(n, env);
  auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {

    for (int p = 0; p < 3; ++p) {
      const int plane = planes[p];

      functions[mode_ + 1](env, dstFrame->GetWritePtr(plane), srcFrame->GetReadPtr(plane), refFrame->GetReadPtr(plane),
        dstFrame->GetPitch(plane), srcFrame->GetPitch(plane), refFrame->GetPitch(plane),
        srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane));
    }
  }
  else {
    if(!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_Y)) || !is_16byte_aligned(refFrame->GetReadPtr(PLANAR_Y)))
      env->ThrowError("Repair: Invalid memory alignment. Unaligned crop?");

    functions[mode_ + 1](env, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), refFrame->GetReadPtr(PLANAR_Y),
      dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), refFrame->GetPitch(PLANAR_Y),
      srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
      if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_U)) || !is_16byte_aligned(refFrame->GetReadPtr(PLANAR_U)))
        env->ThrowError("Repair: Invalid memory alignment. Unaligned crop?");

      if (functions_chroma != nullptr) {
        // for float
        functions_chroma[modeU_ + 1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

        functions_chroma[modeV_ + 1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
      }
      else {
        functions[modeU_ + 1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

        functions[modeV_ + 1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
      }
    }
  }
  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, REF, MODE, MODEU, MODEV, PLANAR, OPT };
    return new Repair(args[CLIP].AsClip(), args[REF].AsClip(), 
      args[MODE].AsInt(1), 
      args[MODEU].AsInt(Repair::UNDEFINED_MODE), 
      args[MODEV].AsInt(Repair::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), // not used, compatibility parameter
      args[OPT].AsInt(0),
      env);
}
