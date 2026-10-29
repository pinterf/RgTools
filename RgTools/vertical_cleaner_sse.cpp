// VerticalCleaner: x86 SSE2 / SSE4.1 kernels

#include "vertical_cleaner.h"

#ifdef INTEL_INTRINSICS

#include "common_sse.h"

namespace {

static void vcleaner_median_sse2_uint8(Byte* pDst, const Byte* pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment* env) {
  typedef uint8_t pixel_t;
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);

  const int width = rowsize / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 0; x < width; x += 16 / sizeof(pixel_t)) {
      __m128i up = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) - srcPitch));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t)));
      __m128i down = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) + srcPitch));

      __m128i mi = _mm_min_epu8(up, down);
      __m128i ma = _mm_max_epu8(up, down);

      __m128i cma = _mm_max_epu8(mi, center);
      __m128i dst = _mm_min_epu8(cma, ma);

      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x * sizeof(pixel_t)), dst);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}


#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void vcleaner_median_sse4_uint16(Byte* pDst, const Byte* pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment* env) {
  typedef uint16_t pixel_t;

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);

  const int width = rowsize / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 0; x < width; x += 16 / sizeof(pixel_t)) {
      __m128i up = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) - srcPitch));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t)));
      __m128i down = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) + srcPitch));

      __m128i mi = _mm_min_epu16(up, down);
      __m128i ma = _mm_max_epu16(up, down);

      __m128i cma = _mm_max_epu16(mi, center);
      __m128i dst = _mm_min_epu16(cma, ma);

      _mm_store_si128(reinterpret_cast<__m128i*>(pDst + x * sizeof(pixel_t)), dst);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

static void vcleaner_median_sse2_float(Byte* pDst, const Byte* pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment* env) {
  typedef float pixel_t;
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);

  const int width = rowsize / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 0; x < width; x += 16 / sizeof(pixel_t)) {
      __m128i up = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) - srcPitch));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t)));
      __m128i down = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x * sizeof(pixel_t) + srcPitch));

      __m128 mi = _mm_min_ps(_mm_castsi128_ps(up), _mm_castsi128_ps(down));
      __m128 ma = _mm_max_ps(_mm_castsi128_ps(up), _mm_castsi128_ps(down));

      __m128 cma = _mm_max_ps(mi, _mm_castsi128_ps(center));
      __m128 dst = _mm_min_ps(cma, ma);

      _mm_store_ps(reinterpret_cast<float *>(pDst + x * sizeof(pixel_t)), dst);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

static void vcleaner_relaxed_median_sse2(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    for (int y = 2; y < height-2; ++y) {
        for (int x = 0; x < rowsize; x+=16) {
            __m128i p2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch*2));
            __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch));
            __m128i c  = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
            __m128i n1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch));
            __m128i n2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch*2));

            __m128i pdiff = _mm_subs_epu8(p1, p2);
            __m128i ndiff = _mm_subs_epu8(n1, n2);

            __m128i pt = _mm_adds_epu8(pdiff, p1);
            __m128i nt = _mm_adds_epu8(ndiff, n1);

            __m128i upper = _mm_min_epu8(pt, nt);
            upper = _mm_max_epu8(upper, p1);
            upper = _mm_max_epu8(upper, n1);

            pdiff = _mm_subs_epu8(p2, p1);
            ndiff = _mm_subs_epu8(n2, n1);

            pt = _mm_subs_epu8(p1, pdiff);
            nt = _mm_subs_epu8(n1, ndiff);

            __m128i minpn1 = _mm_min_epu8(p1, n1);

            __m128i lower = _mm_max_epu8(pt, nt);
            lower = _mm_min_epu8(lower, minpn1);

            __m128i dst = simd_clip(c, lower, upper);

            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void vcleaner_relaxed_median_sse4_uint16(Byte* pDst8, const Byte *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

  uint16_t *pDst = reinterpret_cast<uint16_t *>(pDst8);
  const uint16_t *pSrc = reinterpret_cast<const uint16_t *>(pSrc8);

  dstPitch /= sizeof(uint16_t);
  srcPitch /= sizeof(uint16_t);

  const int width = rowsize / sizeof(uint16_t);

  pSrc += srcPitch*2;
  pDst += dstPitch*2;

  for (int y = 2; y < height-2; ++y) {
    for (int x = 0; x < width; x+=16/sizeof(uint16_t)) {
      __m128i p2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch*2));
      __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch));
      __m128i c  = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
      __m128i n1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch));
      __m128i n2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch*2));

      __m128i pdiff = _mm_subs_epu16(p1, p2);
      __m128i ndiff = _mm_subs_epu16(n1, n2);

      __m128i pt = _mm_adds_epu16(pdiff, p1);
      __m128i nt = _mm_adds_epu16(ndiff, n1);
      if (bits_per_pixel < 16) // for 16 bit _mm_adds limits to FFFF
      {
        __m128i pixel_max = _mm_set1_epi16((short)((1u << bits_per_pixel) - 1));
        pt = _mm_min_epu16(pt, pixel_max);
        nt = _mm_min_epu16(nt, pixel_max);
      }

      __m128i upper = _mm_min_epu16(pt, nt);
      upper = _mm_max_epu16(upper, p1);
      upper = _mm_max_epu16(upper, n1);

      pdiff = _mm_subs_epu16(p2, p1);
      ndiff = _mm_subs_epu16(n2, n1);

      pt = _mm_subs_epu16(p1, pdiff);
      nt = _mm_subs_epu16(n1, ndiff);

      __m128i minpn1 = _mm_min_epu16(p1, n1);

      __m128i lower = _mm_max_epu16(pt, nt);
      lower = _mm_min_epu16(lower, minpn1);

      __m128i dst = simd_clip_16(c, lower, upper);

      _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(uint16_t), (uint8_t *)pSrc, srcPitch*sizeof(uint16_t), rowsize, 2);
}

template<bool chroma>
static void vcleaner_relaxed_median_sse_32(Byte* pDst8, const Byte *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

  float *pDst = reinterpret_cast<float *>(pDst8);
  const float *pSrc = reinterpret_cast<const float *>(pSrc8);

  dstPitch /= sizeof(float);
  srcPitch /= sizeof(float);

  const int width = rowsize / sizeof(float);

  pSrc += srcPitch*2;
  pDst += dstPitch*2;

  for (int y = 2; y < height-2; ++y) {
    for (int x = 0; x < width; x+=16/sizeof(float)) {
      __m128 p2 = _mm_load_ps(reinterpret_cast<const float*>(pSrc + x-srcPitch*2));
      __m128 p1 = _mm_load_ps(reinterpret_cast<const float*>(pSrc + x-srcPitch));
      __m128 c  = _mm_load_ps(reinterpret_cast<const float*>(pSrc + x));
      __m128 n1 = _mm_load_ps(reinterpret_cast<const float*>(pSrc + x+srcPitch));
      __m128 n2 = _mm_load_ps(reinterpret_cast<const float*>(pSrc + x+srcPitch*2));

      __m128 pdiff = _mm_subs_ps_for_diff(p1, p2);
      __m128 ndiff = _mm_subs_ps_for_diff(n1, n2);

      __m128 pt = _mm_adds_ps<chroma>(pdiff, p1);
      __m128 nt = _mm_adds_ps<chroma>(ndiff, n1);

      // no max_pixel_value clamp for float

      __m128 upper = _mm_min_ps(pt, nt);
      upper = _mm_max_ps(upper, p1);
      upper = _mm_max_ps(upper, n1);

      pdiff = _mm_subs_ps_for_diff(p2, p1);
      ndiff = _mm_subs_ps_for_diff(n2, n1);

      pt = _mm_subs_ps<chroma>(p1, pdiff);
      nt = _mm_subs_ps<chroma>(n1, ndiff);

      __m128 minpn1 = _mm_min_ps(p1, n1);

      __m128 lower = _mm_max_ps(pt, nt);
      lower = _mm_min_ps(lower, minpn1);

      __m128 dst = simd_clip_32(c, lower, upper);

      _mm_store_ps(reinterpret_cast<float*>(pDst+x), dst);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(float), (uint8_t *)pSrc, srcPitch*sizeof(float), rowsize, 2);
}

} // namespace

VCleanerProcessor* vcleaner_sse4_functions_uint16_10[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse4_uint16,
  vcleaner_relaxed_median_sse4_uint16<10>
};

VCleanerProcessor* vcleaner_sse4_functions_uint16_12[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse4_uint16,
  vcleaner_relaxed_median_sse4_uint16<12>
};

VCleanerProcessor* vcleaner_sse4_functions_uint16_14[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse4_uint16,
  vcleaner_relaxed_median_sse4_uint16<14>
};

VCleanerProcessor* vcleaner_sse4_functions_uint16_16[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse4_uint16,
  vcleaner_relaxed_median_sse4_uint16<16>
};

VCleanerProcessor* vcleaner_sse2_functions_32[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse2_float,
  vcleaner_relaxed_median_sse_32<false>
};

VCleanerProcessor* vcleaner_sse2_functions_32_chroma[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_sse2_float,
  vcleaner_relaxed_median_sse_32<true>
};

VCleanerProcessor* vcleaner_sse2_functions[] = {
    vcleaner_do_nothing,
    vcleaner_copy_plane,
    vcleaner_median_sse2_uint8,
    vcleaner_relaxed_median_sse2
};

#endif // INTEL_INTRINSICS
