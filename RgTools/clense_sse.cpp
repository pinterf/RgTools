// Clense, ForwardClense, BackwardClense: x86 SSE2 / SSE4.1 kernels

#include "clense.h"

#ifdef INTEL_INTRINSICS

#include "common_sse.h"

namespace {

RG_FORCEINLINE void clense_process_line_sse2(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize) {
    for (int x = 0; x < rowsize; x+=16) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
        auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
        auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

        auto minref = _mm_min_epu8(ref1, ref2);
        auto maxref = _mm_max_epu8(ref1, ref2);
        auto dst = simd_clip(src, minref, maxref);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
    }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void clense_process_line_sse4_16(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize, int /*bits_per_pixel*/)
{
  for (int x = 0; x < rowsize; x+=16) {
    auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
    auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
    auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

    auto minref = _mm_min_epu16(ref1, ref2);
    auto maxref = _mm_max_epu16(ref1, ref2);
    auto dst = simd_clip_16(src, minref, maxref);

    _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
  }
}

RG_FORCEINLINE void clense_process_line_sse2_32(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize) {
  for (int x = 0; x < rowsize; x+=16) {
    auto src = _mm_load_ps(reinterpret_cast<const float*>(pSrc+x));
    auto ref1 = _mm_load_ps(reinterpret_cast<const float*>(pRef1+x));
    auto ref2 = _mm_load_ps(reinterpret_cast<const float*>(pRef2+x));

    auto minref = _mm_min_ps(ref1, ref2);
    auto maxref = _mm_max_ps(ref1, ref2);
    auto dst = simd_clip_32(src, minref, maxref);

    _mm_store_ps(reinterpret_cast<float*>(pDst+x), dst);
  }
}

RG_FORCEINLINE void sclense_process_line_sse2(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize) {
    for (int x = 0; x < rowsize; x+=16) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
        auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
        auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

        auto minref = _mm_min_epu8(ref1, ref2);
        auto maxref = _mm_max_epu8(ref1, ref2);

        auto ma = _mm_subs_epu8(maxref, ref2);
        auto mi = _mm_subs_epu8(ref2, minref);

        ma = _mm_adds_epu8(ma, maxref);
        mi = _mm_subs_epu8(minref, mi);

        auto dst = simd_clip(src, mi, ma);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
    }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void sclense_process_line_sse4_16(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize, int bits_per_pixel) {
  // 16 bit: the min with 0xFFFF is a no-op (_mm_adds_epu16 saturates there)
  const __m128i pixel_max = _mm_set1_epi16((short)((1u << bits_per_pixel) - 1));

  for (int x = 0; x < rowsize; x+=16) {
    auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
    auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
    auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

    auto minref = _mm_min_epu16(ref1, ref2);
    auto maxref = _mm_max_epu16(ref1, ref2);

    auto ma = _mm_subs_epu16(maxref, ref2);
    auto mi = _mm_subs_epu16(ref2, minref);

    ma = _mm_adds_epu16(ma, maxref);
    mi = _mm_subs_epu16(minref, mi);

    ma = _mm_min_epu16(ma, pixel_max); // saturation is not enough below 16 bit

    auto dst = simd_clip_16(src, mi, ma);

    _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
  }
}

template<bool chroma>
RG_FORCEINLINE void sclense_process_line_sse2_32(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int rowsize) {
  for (int x = 0; x < rowsize; x+=16) {
    auto src = _mm_load_ps(reinterpret_cast<const float*>(pSrc+x));
    auto ref1 = _mm_load_ps(reinterpret_cast<const float*>(pRef1+x));
    auto ref2 = _mm_load_ps(reinterpret_cast<const float*>(pRef2+x));

    auto minref = _mm_min_ps(ref1, ref2);
    auto maxref = _mm_max_ps(ref1, ref2);

    auto ma = _mm_subs_ps_for_diff(maxref, ref2);
    auto mi = _mm_subs_ps_for_diff(ref2, minref);

    ma = _mm_adds_ps<chroma>(ma, maxref);
    mi = _mm_subs_ps<chroma>(minref, mi);

    auto dst = simd_clip_32(src, mi, ma);

    _mm_store_ps(reinterpret_cast<float*>(pDst+x), dst);
  }
}

template<decltype(clense_process_line_sse2) processor>
void process_plane_sse(uint8_t* pDst, const uint8_t *pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int bits_per_pixel, IScriptEnvironment *env) {
    if (!is_16byte_aligned(pSrc) || !is_16byte_aligned(pRef1) || !is_16byte_aligned(pRef2)) {
        env->ThrowError("Invalid memory alignment. Used unaligned crop?"); //omg I feel so dumb
    }
    auto mod16Width = (rowsize / 16) * 16;

    for (int y = 0; y < height; ++y) {
        processor(pDst, pSrc, pRef1, pRef2, mod16Width);

        if (mod16Width != rowsize) {
            processor(pDst + mod16Width, pSrc + mod16Width, pRef1 + mod16Width, pRef2 + mod16Width, 16);
        }
        pDst += dstPitch;
        pSrc += srcPitch;
        pRef1 += ref1Pitch;
        pRef2 += ref2Pitch;
    }
}

template<decltype(clense_process_line_sse4_16) processor>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void process_plane_sse41(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef1, const uint8_t* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int bits_per_pixel, IScriptEnvironment* env) {
  if (!is_16byte_aligned(pSrc) || !is_16byte_aligned(pRef1) || !is_16byte_aligned(pRef2)) {
    env->ThrowError("Invalid memory alignment. Used unaligned crop?"); //omg I feel so dumb
  }
  auto mod16Width = (rowsize / 16) * 16;

  for (int y = 0; y < height; ++y) {
    processor(pDst, pSrc, pRef1, pRef2, mod16Width, bits_per_pixel);

    if (mod16Width != rowsize) {
      processor(pDst + mod16Width, pSrc + mod16Width, pRef1 + mod16Width, pRef2 + mod16Width, 16, bits_per_pixel);
    }
    pDst += dstPitch;
    pSrc += srcPitch;
    pRef1 += ref1Pitch;
    pRef2 += ref2Pitch;
  }
}

} // namespace

// The SSE processor of the mode / format, the counterpart of the C selection in the Clense constructor.
// 8 bit and float: SSE2, 10-16 bit: SSE4.1 (one variant, the max pixel value comes from bits_per_pixel)
ClenseProcessor* clense_sse_processor(ClenseMode mode, int pixelsize, bool chroma) {
  if (pixelsize == 1) {
    return (mode == ClenseMode::BOTH)
      ? process_plane_sse<clense_process_line_sse2>
      : process_plane_sse<sclense_process_line_sse2>;
  }
  if (pixelsize == 2) {
    return (mode == ClenseMode::BOTH)
      ? process_plane_sse41<clense_process_line_sse4_16>
      : process_plane_sse41<sclense_process_line_sse4_16>;
  }
  // pixelsize == 4
  if (mode == ClenseMode::BOTH)
    return process_plane_sse<clense_process_line_sse2_32>;
  return chroma
    ? process_plane_sse<sclense_process_line_sse2_32<true>>
    : process_plane_sse<sclense_process_line_sse2_32<false>>;
}

#endif // INTEL_INTRINSICS
