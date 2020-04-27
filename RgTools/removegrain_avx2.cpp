#include "rg_functions_avx2.h"
#include "removegrain.h"

// AVX2: not using special aligned templates, loadu is fast is aligned
template<typename pixel_t, SseModeProcessor processor>
static void process_plane_avx2(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    _mm256_zeroupper();
    
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);
    const int pixels_at_at_time = 32 / sizeof(pixel_t); // 32!

    pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height - 1; ++y) {
      pDst[0] = pSrc[0];

      // unaligned first 32 bytes, last pixel overlaps with the next aligned loop
      __m256i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst + 1), result);

      // possibly aligned
      for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
        __m256i result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst + x), result); // store as unaligned
      }
      
      if (mod_width != width) {
        __m256i result = processor((uint8_t *)(pSrc + width - 1 - pixels_at_at_time), srcPitchOrig);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst + width - 1 - pixels_at_at_time), result);
      }
      
      pDst[width - 1] = pSrc[width - 1];

      pSrc += srcPitch;
      pDst += dstPitch;
    }
    _mm256_zeroupper();

    env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1);
}


template<typename pixel_t, SseModeProcessor processor>
static void process_halfplane_avx2(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
  _mm256_zeroupper();

  pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
  const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

  dstPitch /= sizeof(pixel_t);
  const int srcPitchOrig = srcPitch;
  srcPitch /= sizeof(pixel_t);

  const int width = rowsize / sizeof(pixel_t);
  const int pixels_at_at_time = 32 / sizeof(pixel_t); // 32!

  pSrc += srcPitch;
    pDst += dstPitch;
    int mod_width = width / pixels_at_at_time * pixels_at_at_time;

    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding

        // unaligned first 32 bytes, last pixel overlaps with the next aligned loop
        __m256i result = processor((uint8_t *)(pSrc + 1), srcPitchOrig);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst + 1), result);

        // possibly aligned
        for (int x = pixels_at_at_time; x < mod_width - 1; x += pixels_at_at_time) {
          __m256i result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst + x), result);
        }

        if (mod_width != width) {
          __m256i result = processor((uint8_t *)(pSrc+width-1-pixels_at_at_time), srcPitchOrig);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(pDst+width-1-pixels_at_at_time), result);
        }

        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t) == 4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        _mm256_zeroupper();

        env->BitBlt((uint8_t *)(pDst), dstPitch*sizeof(pixel_t), (uint8_t *)(pSrc), srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, SseModeProcessor processor>
static void process_even_rows_avx2(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    _mm256_zeroupper();
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_avx2<pixel_t, processor>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
}

template<typename pixel_t, SseModeProcessor processor>
static void process_odd_rows_avx2(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    _mm256_zeroupper();
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    process_halfplane_avx2<pixel_t, processor>(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);

    _mm256_zeroupper();
    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

static void doNothing(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {

}

static void copyPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  _mm256_zeroupper();
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}


PlaneProcessor* avx2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_avx2<uint8_t, rg_mode1_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode2_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode3_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode4_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode5_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode6_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode7_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode8_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode9_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode10_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode11_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode12_avx2<false>>,
    process_even_rows_avx2<uint8_t, rg_mode13_and14_avx2<false>>,
    process_odd_rows_avx2<uint8_t, rg_mode13_and14_avx2<false>>,
    process_even_rows_avx2<uint8_t, rg_mode15_and16_avx2<false>>,
    process_odd_rows_avx2<uint8_t, rg_mode15_and16_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode17_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode18_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode19_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode20_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode21_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode22_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode23_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode24_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode25_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode26_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode27_avx2<false>>,
    process_plane_avx2<uint8_t, rg_mode28_avx2<false>>,
};


PlaneProcessor* avx2_functions_16_10[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<uint16_t, rg_mode1_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode2_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode3_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode4_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode5_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode6_avx2_16<10, false>>,
  process_plane_avx2<uint16_t, rg_mode7_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode8_avx2_16<10, false>>,
  process_plane_avx2<uint16_t, rg_mode9_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode10_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode11_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode12_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode17_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode18_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode19_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode20_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode21_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode22_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode23_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode24_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode25_avx2_16<10,false>>,
  process_plane_avx2<uint16_t, rg_mode26_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode27_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode28_avx2_16<false>>,
};

PlaneProcessor* avx2_functions_16_12[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<uint16_t, rg_mode1_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode2_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode3_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode4_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode5_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode6_avx2_16<12, false>>,
  process_plane_avx2<uint16_t, rg_mode7_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode8_avx2_16<12, false>>,
  process_plane_avx2<uint16_t, rg_mode9_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode10_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode11_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode12_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode17_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode18_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode19_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode20_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode21_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode22_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode23_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode24_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode25_avx2_16<12,false>>,
  process_plane_avx2<uint16_t, rg_mode26_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode27_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode28_avx2_16<false>>,
};

PlaneProcessor* avx2_functions_16_14[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<uint16_t, rg_mode1_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode2_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode3_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode4_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode5_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode6_avx2_16<14, false>>,
  process_plane_avx2<uint16_t, rg_mode7_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode8_avx2_16<14, false>>,
  process_plane_avx2<uint16_t, rg_mode9_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode10_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode11_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode12_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode17_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode18_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode19_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode20_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode21_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode22_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode23_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode24_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode25_avx2_16<14,false>>,
  process_plane_avx2<uint16_t, rg_mode26_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode27_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode28_avx2_16<false>>,
};

PlaneProcessor* avx2_functions_16_16[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<uint16_t, rg_mode1_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode2_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode3_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode4_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode5_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode6_avx2_16<16, false>>,
  process_plane_avx2<uint16_t, rg_mode7_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode8_avx2_16<16, false>>,
  process_plane_avx2<uint16_t, rg_mode9_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode10_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode11_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode12_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode13_and14_avx2_16<false>>,
  process_even_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_odd_rows_avx2<uint16_t, rg_mode15_and16_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode17_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode18_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode19_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode20_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode21_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode22_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode23_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode24_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode25_avx2_16<16,false>>,
  process_plane_avx2<uint16_t, rg_mode26_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode27_avx2_16<false>>,
  process_plane_avx2<uint16_t, rg_mode28_avx2_16<false>>,
};



PlaneProcessor* avx2_functions_32_luma[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<float, rg_mode1_avx2_32<false>>,
  process_plane_avx2<float, rg_mode2_avx2_32<false>>,
  process_plane_avx2<float, rg_mode3_avx2_32<false>>,
  process_plane_avx2<float, rg_mode4_avx2_32<false>>,
  process_plane_avx2<float, rg_mode5_avx2_32<false>>,
  process_plane_avx2<float, rg_mode6_avx2_32<false>>,
  process_plane_avx2<float, rg_mode7_avx2_32<false>>,
  process_plane_avx2<float, rg_mode8_avx2_32<false>>,
  process_plane_avx2<float, rg_mode9_avx2_32<false>>,
  process_plane_avx2<float, rg_mode10_avx2_32<false>>,
  process_plane_avx2<float, rg_mode11_avx2_32<false>>,
  process_plane_avx2<float, rg_mode12_avx2_32<false>>,
  process_even_rows_avx2<float, rg_mode13_and14_avx2_32<false>>,
  process_odd_rows_avx2<float, rg_mode13_and14_avx2_32<false>>,
  process_even_rows_avx2<float, rg_mode15_and16_avx2_32<false>>,
  process_odd_rows_avx2<float, rg_mode15_and16_avx2_32<false>>,
  process_plane_avx2<float, rg_mode17_avx2_32<false>>,
  process_plane_avx2<float, rg_mode18_avx2_32<false>>,
  process_plane_avx2<float, rg_mode19_avx2_32<false>>,
  process_plane_avx2<float, rg_mode20_avx2_32<false>>,
  process_plane_avx2<float, rg_mode21_avx2_32<false>>,
  process_plane_avx2<float, rg_mode22_avx2_32<false>>,
  process_plane_avx2<float, rg_mode23_avx2_32<false, false>>,
  process_plane_avx2<float, rg_mode24_avx2_32<false, false>>,
  process_plane_avx2<float, rg_mode25_avx2_32<false, false>>, // false: luma
  process_plane_avx2<float, rg_mode26_avx2_32<false>>,
  process_plane_avx2<float, rg_mode27_avx2_32<false>>,
  process_plane_avx2<float, rg_mode28_avx2_32<false>>,
};

PlaneProcessor* avx2_functions_32_chroma[] = {
  doNothing,
  copyPlane,
  process_plane_avx2<float, rg_mode1_avx2_32<false>>,
  process_plane_avx2<float, rg_mode2_avx2_32<false>>,
  process_plane_avx2<float, rg_mode3_avx2_32<false>>,
  process_plane_avx2<float, rg_mode4_avx2_32<false>>,
  process_plane_avx2<float, rg_mode5_avx2_32<false>>,
  process_plane_avx2<float, rg_mode6_avx2_32<false>>,
  process_plane_avx2<float, rg_mode7_avx2_32<false>>,
  process_plane_avx2<float, rg_mode8_avx2_32<false>>,
  process_plane_avx2<float, rg_mode9_avx2_32<false>>,
  process_plane_avx2<float, rg_mode10_avx2_32<false>>,
  process_plane_avx2<float, rg_mode11_avx2_32<false>>,
  process_plane_avx2<float, rg_mode12_avx2_32<false>>,
  process_even_rows_avx2<float, rg_mode13_and14_avx2_32<false>>,
  process_odd_rows_avx2<float, rg_mode13_and14_avx2_32<false>>,
  process_even_rows_avx2<float, rg_mode15_and16_avx2_32<false>>,
  process_odd_rows_avx2<float, rg_mode15_and16_avx2_32<false>>,
  process_plane_avx2<float, rg_mode17_avx2_32<false>>,
  process_plane_avx2<float, rg_mode18_avx2_32<false>>,
  process_plane_avx2<float, rg_mode19_avx2_32<false>>,
  process_plane_avx2<float, rg_mode20_avx2_32<false>>,
  process_plane_avx2<float, rg_mode21_avx2_32<false>>,
  process_plane_avx2<float, rg_mode22_avx2_32<false>>,
  process_plane_avx2<float, rg_mode23_avx2_32<false, true>>,
  process_plane_avx2<float, rg_mode24_avx2_32<false, true>>,
  process_plane_avx2<float, rg_mode25_avx2_32<false, true>>, // true: chroma
  process_plane_avx2<float, rg_mode26_avx2_32<false>>,
  process_plane_avx2<float, rg_mode27_avx2_32<false>>,
  process_plane_avx2<float, rg_mode28_avx2_32<false>>,
};
