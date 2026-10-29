// Repair: x86 SSE2 / SSE4.1 plane loops and function tables

#include "repair.h"

#ifdef INTEL_INTRINSICS

#include "repair_functions_sse.h" // kernels, in an anonymous namespace

namespace {

template<typename pixel_t, auto processor, auto processor_a>
static void process_plane_sse(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int /*bits_per_pixel*/) {
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

        if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
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
template<typename pixel_t, auto processor, auto processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse3")))
#endif
static void process_plane_sse3(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel) {
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

    if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
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

template<typename pixel_t, auto processor, auto processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void process_plane_sse41(IScriptEnvironment* env, BYTE* pDst8, const BYTE* pSrc8, const BYTE* pRef8, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel) {
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
  // 10-16 bit kernels that depend on the max pixel value get it as a 4th parameter
  constexpr bool with_max = std::is_invocable_v<decltype(processor), const Byte*, const __m128i&, int, const __m128i&>;
  const __m128i pixel_max = _mm_set1_epi16(sizeof(pixel_t) == 2 ? (short)((1u << bits_per_pixel) - 1) : 0);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  int mod_width = width / pixels_at_at_time * pixels_at_at_time;

  for (int y = 1; y < height - 1; ++y) {
    pDst[0] = pSrc[0];

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + 1));
    __m128i result;
    if constexpr (with_max) result = processor((uint8_t*)(pRef + 1), val, refPitchOrig, pixel_max);
    else result = processor((uint8_t*)(pRef + 1), val, refPitchOrig);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst + 1), result);

    //aligned
    for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + x));
      __m128i result;
      if constexpr (with_max) result = processor_a((uint8_t*)(pRef + x), val, refPitchOrig, pixel_max);
      else result = processor_a((uint8_t*)(pRef + x), val, refPitchOrig);
      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x), result);
    }

    if (mod_width != width && width - 1 - pixels_at_at_time >= 1) { // width == vector + 1: covered by the first vector
      __m128i val = simd_loadu_si128_sse3((uint8_t*)(pSrc + width - 1 - pixels_at_at_time));
      __m128i result;
      if constexpr (with_max) result = processor((uint8_t*)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig, pixel_max);
      else result = processor((uint8_t*)(pRef + width - 1 - pixels_at_at_time), val, refPitchOrig);
      _mm_storeu_si128(reinterpret_cast<__m128i*>((uint8_t*)(pDst + width - 1 - pixels_at_at_time)), result);
    }


    pDst[width - 1] = pSrc[width - 1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt((uint8_t*)(pDst), dstPitch * sizeof(pixel_t), (uint8_t*)(pSrc), srcPitch * sizeof(pixel_t), rowsize, 1);
}

} // namespace

RepairPlaneProcessor* repair_sse4_functions[] = {
    repair_do_nothing,
    repair_copy_plane,
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
    repair_do_nothing,
    process_plane_sse41<uint8_t, repair_mode26_sse<false>, repair_mode26_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode27_sse<false>, repair_mode27_sse<true>>,
    process_plane_sse41<uint8_t, repair_mode28_sse<false>, repair_mode28_sse<true>>,
};

RepairPlaneProcessor* repair_sse2_functions[] = {
    repair_do_nothing,
    repair_copy_plane,
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
    repair_do_nothing,
    process_plane_sse<uint8_t, repair_mode26_sse2<false>, repair_mode26_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode27_sse2<false>, repair_mode27_sse2<true>>,
    process_plane_sse<uint8_t, repair_mode28_sse2<false>, repair_mode28_sse2<true>>,
};

// 10-16 bit (the max pixel value comes from bits_per_pixel)
RepairPlaneProcessor* repair_sse4_functions_16[] = {
  repair_do_nothing,
  repair_copy_plane,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode2_sse_16<false>,  repair_mode2_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode3_sse_16<false>,  repair_mode3_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode4_sse_16<false>,  repair_mode4_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode5_sse_16<false>,  repair_mode5_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode6_sse_16<false>,  repair_mode6_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode7_sse_16<false>,  repair_mode7_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode8_sse_16<false>,  repair_mode8_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode9_sse_16<false>,  repair_mode9_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode10_sse_16<false>,  repair_mode10_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode1_sse_16<false>,  repair_mode1_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode12_sse_16<false>,  repair_mode12_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode13_sse_16<false>,  repair_mode13_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode14_sse_16<false>,  repair_mode14_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode15_sse_16<false>,  repair_mode15_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode16_sse_16<false>,  repair_mode16_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode17_sse_16<false>,  repair_mode17_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode18_sse_16<false>,  repair_mode18_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode19_sse_16<false>,  repair_mode19_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode20_sse_16<false>,  repair_mode20_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode21_sse_16<false>,  repair_mode21_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode22_sse_16<false>,  repair_mode22_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode23_sse_16<false>,  repair_mode23_sse_16<true>>, 
  process_plane_sse41<uint16_t, repair_mode24_sse_16<false>,  repair_mode24_sse_16<true>>,
  repair_do_nothing,
  process_plane_sse41<uint16_t, repair_mode26_sse_16<false>,  repair_mode26_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode27_sse_16<false>,  repair_mode27_sse_16<true>>,
  process_plane_sse41<uint16_t, repair_mode28_sse_16<false>,  repair_mode28_sse_16<true>>,
};


RepairPlaneProcessor* repair_sse4_functions_32[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  repair_do_nothing,
  process_plane_sse41<float, repair_mode26_sse_32<false>, repair_mode26_sse_32<true>>,
  process_plane_sse41<float, repair_mode27_sse_32<false>, repair_mode27_sse_32<true>>,
  process_plane_sse41<float, repair_mode28_sse_32<false>, repair_mode28_sse_32<true>>
};

RepairPlaneProcessor* repair_sse4_functions_32_chroma[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  repair_do_nothing,
  process_plane_sse41<float, repair_mode26_sse_32<false>, repair_mode26_sse_32<true>>,
  process_plane_sse41<float, repair_mode27_sse_32<false>, repair_mode27_sse_32<true>>,
  process_plane_sse41<float, repair_mode28_sse_32<false>, repair_mode28_sse_32<true>>
};

#endif // INTEL_INTRINSICS
