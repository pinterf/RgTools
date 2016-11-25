#include "repair_functions_c.h"
#include "repair_functions_sse.h"
#include "repair.h"


template<typename pixel_t, SseModeProcessor processor, SseModeProcessor processor_a, InstructionSet optLevel>
static void process_plane_sse(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

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
        __m128i val = simd_loadu_si128<optLevel>((uint8_t *)(pSrc+1));
        __m128i result = processor((uint8_t *)(pRef+1), val, refPitchOrig);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+1), result);

        //aligned
        for (int x = pixels_at_at_time; x < mod_width-1; x+= pixels_at_at_time) {
            __m128i val = simd_loada_si128<optLevel>((uint8_t *)(pSrc+x));
            __m128i result = processor_a((uint8_t *)(pRef+x), val, refPitchOrig);
            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod_width != width) {
            __m128i val = simd_loadu_si128<optLevel>((uint8_t *)(pSrc + width - 1 - pixels_at_at_time));
            __m128i result = processor((uint8_t *)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig);
            _mm_storeu_si128(reinterpret_cast<__m128i*>((uint8_t *)(pDst + width - 1 - pixels_at_at_time)), result);
        }


        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
        pRef += refPitch;
    }

    env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1);
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


RepairPlaneProcessor* sse3_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, repair_mode1_sse<false, SSE3>, repair_mode1_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode2_sse<false, SSE3>, repair_mode2_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode3_sse<false, SSE3>, repair_mode3_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode4_sse<false, SSE3>, repair_mode4_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode5_sse<false, SSE3>, repair_mode5_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode6_sse<false, SSE3>, repair_mode6_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode7_sse<false, SSE3>, repair_mode7_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode8_sse<false, SSE3>, repair_mode8_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode9_sse<false, SSE3>, repair_mode9_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode10_sse<false, SSE3>, repair_mode10_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode1_sse<false, SSE3>, repair_mode1_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode12_sse<false, SSE3>, repair_mode12_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode13_sse<false, SSE3>, repair_mode13_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode14_sse<false, SSE3>, repair_mode14_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode15_sse<false, SSE3>, repair_mode15_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode16_sse<false, SSE3>, repair_mode16_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode17_sse<false, SSE3>, repair_mode17_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode18_sse<false, SSE3>, repair_mode18_sse<true, SSE3>, SSE3>,
    process_plane_sse<uint8_t, repair_mode19_sse<false, SSE3>, repair_mode19_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode20_sse<false, SSE3>, repair_mode20_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode21_sse<false, SSE3>, repair_mode21_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode22_sse<false, SSE3>, repair_mode22_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode23_sse<false, SSE3>, repair_mode23_sse<true, SSE3>, SSE3>, 
    process_plane_sse<uint8_t, repair_mode24_sse<false, SSE3>, repair_mode24_sse<true, SSE3>, SSE3> 
};

RepairPlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<uint8_t, repair_mode1_sse<false, SSE2>, repair_mode1_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode2_sse<false, SSE2>, repair_mode2_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode3_sse<false, SSE2>, repair_mode3_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode4_sse<false, SSE2>, repair_mode4_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode5_sse<false, SSE2>, repair_mode5_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode6_sse<false, SSE2>, repair_mode6_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode7_sse<false, SSE2>, repair_mode7_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode8_sse<false, SSE2>, repair_mode8_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode9_sse<false, SSE2>, repair_mode9_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode10_sse<false, SSE2>, repair_mode10_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode1_sse<false, SSE2>, repair_mode1_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode12_sse<false, SSE2>, repair_mode12_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode13_sse<false, SSE2>, repair_mode13_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode14_sse<false, SSE2>, repair_mode14_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode15_sse<false, SSE2>, repair_mode15_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode16_sse<false, SSE2>, repair_mode16_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode17_sse<false, SSE2>, repair_mode17_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode18_sse<false, SSE2>, repair_mode18_sse<true, SSE2>, SSE2>,
    process_plane_sse<uint8_t, repair_mode19_sse<false, SSE2>, repair_mode19_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode20_sse<false, SSE2>, repair_mode20_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode21_sse<false, SSE2>, repair_mode21_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode22_sse<false, SSE2>, repair_mode22_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode23_sse<false, SSE2>, repair_mode23_sse<true, SSE2>, SSE2>, 
    process_plane_sse<uint8_t, repair_mode24_sse<false, SSE2>, repair_mode24_sse<true, SSE2>, SSE2> 
};

RepairPlaneProcessor* sse4_functions_16_10[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode6_sse_16<10, false>,  repair_mode6_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode8_sse_16<10, false>,  repair_mode8_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode16_sse_16<10, false>,  repair_mode16_sse_16<10, true>,  SSE3>,
  process_plane_sse<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode19_sse_16<10, false>,  repair_mode19_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode20_sse_16<10, false>,  repair_mode20_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode21_sse_16<10, false>,  repair_mode21_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode22_sse_16<10, false>,  repair_mode22_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode23_sse_16<10, false>,  repair_mode23_sse_16<10, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode24_sse_16<10, false>,  repair_mode24_sse_16<10, true>,  SSE3> 
};

RepairPlaneProcessor* sse4_functions_16_12[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode6_sse_16<12, false>,  repair_mode6_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode8_sse_16<12, false>,  repair_mode8_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode16_sse_16<12, false>,  repair_mode16_sse_16<12, true>,  SSE3>,
  process_plane_sse<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode19_sse_16<12, false>,  repair_mode19_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode20_sse_16<12, false>,  repair_mode20_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode21_sse_16<12, false>,  repair_mode21_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode22_sse_16<12, false>,  repair_mode22_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode23_sse_16<12, false>,  repair_mode23_sse_16<12, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode24_sse_16<12, false>,  repair_mode24_sse_16<12, true>,  SSE3> 
};

RepairPlaneProcessor* sse4_functions_16_14[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode6_sse_16<14, false>,  repair_mode6_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode8_sse_16<14, false>,  repair_mode8_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode16_sse_16<14, false>,  repair_mode16_sse_16<14, true>,  SSE3>,
  process_plane_sse<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode19_sse_16<14, false>,  repair_mode19_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode20_sse_16<14, false>,  repair_mode20_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode21_sse_16<14, false>,  repair_mode21_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode22_sse_16<14, false>,  repair_mode22_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode23_sse_16<14, false>,  repair_mode23_sse_16<14, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode24_sse_16<14, false>,  repair_mode24_sse_16<14, true>,  SSE3> 
};

RepairPlaneProcessor* sse4_functions_16_16[] = {
  doNothing,
  copyPlane,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode6_sse_16<16, false>,  repair_mode6_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode8_sse_16<16, false>,  repair_mode8_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>, SSE3>, 
  process_plane_sse<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode16_sse_16<16, false>,  repair_mode16_sse_16<16, true>,  SSE3>,
  process_plane_sse<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>, SSE3>,
  process_plane_sse<uint16_t, repair_mode19_sse_16<16, false>,  repair_mode19_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode20_sse_16<16, false>,  repair_mode20_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode21_sse_16<16, false>,  repair_mode21_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode22_sse_16<16, false>,  repair_mode22_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode23_sse_16<16, false>,  repair_mode23_sse_16<16, true>,  SSE3>, 
  process_plane_sse<uint16_t, repair_mode24_sse_16<16, false>,  repair_mode24_sse_16<16, true>,  SSE3> 
};


RepairPlaneProcessor* sse4_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_sse<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode2_sse_32<false>, repair_mode2_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode3_sse_32<false>, repair_mode3_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode4_sse_32<false>, repair_mode4_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode5_sse_32<false>, repair_mode5_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode6_sse_32<false>, repair_mode6_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode7_sse_32<false>, repair_mode7_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode8_sse_32<false>, repair_mode8_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode9_sse_32<false>, repair_mode9_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode10_sse_32<false>, repair_mode10_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode1_sse_32<false>, repair_mode1_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode12_sse_32<false>, repair_mode12_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode13_sse_32<false>, repair_mode13_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode14_sse_32<false>, repair_mode14_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode15_sse_32<false>, repair_mode15_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode16_sse_32<false>, repair_mode16_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode17_sse_32<false>, repair_mode17_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode18_sse_32<false>, repair_mode18_sse_32<true>, SSE3>,
  process_plane_sse<float, repair_mode19_sse_32<false>, repair_mode19_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode20_sse_32<false>, repair_mode20_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode21_sse_32<false>, repair_mode21_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode22_sse_32<false>, repair_mode22_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode23_sse_32<false>, repair_mode23_sse_32<true>, SSE3>, 
  process_plane_sse<float, repair_mode24_sse_32<false>, repair_mode24_sse_32<true>, SSE3> 
};


RepairPlaneProcessor* c_functions[] = {
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
  process_plane_c<uint8_t,repair_mode24_cpp> 
};

RepairPlaneProcessor* c_functions_10[] = {
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<10>> 
};

RepairPlaneProcessor* c_functions_12[] = {
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<12>> 
};

RepairPlaneProcessor* c_functions_14[] = {
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<14>> 
};

RepairPlaneProcessor* c_functions_16[] = {
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<16>> 
};

RepairPlaneProcessor* c_functions_32[] = {
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
  process_plane_c<float,repair_mode19_cpp_32>, 
  process_plane_c<float,repair_mode20_cpp_32>, 
  process_plane_c<float,repair_mode21_cpp_32>, 
  process_plane_c<float,repair_mode22_cpp_32>, 
  process_plane_c<float,repair_mode23_cpp_32>, 
  process_plane_c<float,repair_mode24_cpp_32> 
};

Repair::Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
  : GenericVideoFilter(child), ref_(ref), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {

  auto refVi = ref_->GetVideoInfo();

  if (!(vi.IsPlanar() || skip_cs_check)) {
    env->ThrowError("Repair works only with planar colorspaces");
  }

  if (vi.width != refVi.width || vi.height != refVi.height) {
    env->ThrowError("Clips should be of the same size!");
  }

  if (mode <= UNDEFINED_MODE || mode_ > 24 || modeU_ > 24 || modeV_ > 24) {
    env->ThrowError("Repair mode should be between -1 and 24!");
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


PVideoFrame Repair::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto refFrame = ref_->GetFrame(n, env);
  auto dstFrame = env->NewVideoFrame(vi);

  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
    functions[mode_+1](env, dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetReadPtr(PLANAR_G), refFrame->GetReadPtr(PLANAR_G),
      dstFrame->GetPitch(PLANAR_G), srcFrame->GetPitch(PLANAR_G), refFrame->GetPitch(PLANAR_G),
      srcFrame->GetRowSize(PLANAR_G), srcFrame->GetHeight(PLANAR_G));
    functions[mode_+1](env, dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetReadPtr(PLANAR_B), refFrame->GetReadPtr(PLANAR_B),
      dstFrame->GetPitch(PLANAR_B), srcFrame->GetPitch(PLANAR_B), refFrame->GetPitch(PLANAR_B),
      srcFrame->GetRowSize(PLANAR_B), srcFrame->GetHeight(PLANAR_B));
    functions[mode_+1](env, dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetReadPtr(PLANAR_R), refFrame->GetReadPtr(PLANAR_R),
      dstFrame->GetPitch(PLANAR_R), srcFrame->GetPitch(PLANAR_R), refFrame->GetPitch(PLANAR_R),
      srcFrame->GetRowSize(PLANAR_R), srcFrame->GetHeight(PLANAR_R));
  }
  else {

    functions[mode_ + 1](env, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), refFrame->GetReadPtr(PLANAR_Y),
      dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), refFrame->GetPitch(PLANAR_Y),
      srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
      functions[modeU_ + 1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
        dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
        srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

      functions[modeV_ + 1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
        dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
        srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
    }
  }
  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, REF, MODE, MODEU, MODEV, PLANAR };
    return new Repair(args[CLIP].AsClip(), args[REF].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(Repair::UNDEFINED_MODE), args[MODEV].AsInt(Repair::UNDEFINED_MODE), args[PLANAR].AsBool(false), env);
}
