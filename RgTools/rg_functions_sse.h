#ifndef __RG_FUNCTIONS_SSE_H__
#define __RG_FUNCTIONS_SSE_H__

// note: 8 bit functions are duplicated
// one for pure sse2 the other (_sse suffix) for sse4.1
// the two differs only in the load command, but
// had to be separated because in gcc and clang they
// cannot be templatized
// reason: platform is function attribute, cannot mix e.g. sse2 and sse4.1

#include "common.h"

typedef __m128i (SseModeProcessor)(const Byte*, int);

//(x&y)+((x^y)/2) for (a+b)/2
static RG_FORCEINLINE __m128i not_rounded_average(__m128i a, __m128i b) {
    auto andop = _mm_and_si128(a, b);
    auto xorop = _mm_xor_si128(a, b);
    //kinda psrlb, probably not optimal but works
    xorop = _mm_srli_epi16(xorop, 1); // no _mm_srli_epi8: shift + mask
    xorop = _mm_and_si128(xorop, _mm_set1_epi8(0x7F));
    return _mm_adds_epu8(xorop, andop);
}

// PF saturates to FFFF
static RG_FORCEINLINE __m128i not_rounded_average_16(__m128i a, __m128i b) {
  auto andop = _mm_and_si128(a, b);
  auto xorop = _mm_xor_si128(a, b);
  //kinda psrlb, probably not optimal but works
  xorop = _mm_srli_epi16(xorop, 1); // /2, no tricks like at 8 bit
  return _mm_adds_epu16(xorop, andop);
}

//-------------------

// todo: 
// - Float parts were blindly converted, should be optimized,
//   check averaging simplifications, clamping

//-------------------
// For each mode: 
// 8 bit SSE2 
// 8 bit SSE4.1
// 10-16 bit SSE4.1
// 32 bit SSE4.1
// (see AVX2 in another source file)
//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode1_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    __m128i mi = _mm_min_epu8 (
        _mm_min_epu8(_mm_min_epu8(a1, a2), _mm_min_epu8(a3, a4)),
        _mm_min_epu8(_mm_min_epu8(a5, a6), _mm_min_epu8(a7, a8))
        );
    __m128i ma = _mm_max_epu8 (
        _mm_max_epu8(_mm_max_epu8(a1, a2), _mm_max_epu8(a3, a4)),
        _mm_max_epu8(_mm_max_epu8(a5, a6), _mm_max_epu8(a7, a8))
        );

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode1_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  __m128i mi = _mm_min_epu8(
    _mm_min_epu8(_mm_min_epu8(a1, a2), _mm_min_epu8(a3, a4)),
    _mm_min_epu8(_mm_min_epu8(a5, a6), _mm_min_epu8(a7, a8))
  );
  __m128i ma = _mm_max_epu8(
    _mm_max_epu8(_mm_max_epu8(a1, a2), _mm_max_epu8(a3, a4)),
    _mm_max_epu8(_mm_max_epu8(a5, a6), _mm_max_epu8(a7, a8))
  );

  return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode1_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  __m128i mi = _mm_min_epu16 (
    _mm_min_epu16(_mm_min_epu16(a1, a2), _mm_min_epu16(a3, a4)),
    _mm_min_epu16(_mm_min_epu16(a5, a6), _mm_min_epu16(a7, a8))
  );
  __m128i ma = _mm_max_epu16 (
    _mm_max_epu16(_mm_max_epu16(a1, a2), _mm_max_epu16(a3, a4)),
    _mm_max_epu16(_mm_max_epu16(a5, a6), _mm_max_epu16(a7, a8))
  );

  return simd_clip_16(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode1_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  __m128 mi = _mm_min_ps (
    _mm_min_ps(_mm_min_ps(a1, a2), _mm_min_ps(a3, a4)),
    _mm_min_ps(_mm_min_ps(a5, a6), _mm_min_ps(a7, a8))
  );
  __m128 ma = _mm_max_ps (
    _mm_max_ps(_mm_max_ps(a1, a2), _mm_max_ps(a3, a4)),
    _mm_max_ps(_mm_max_ps(a5, a6), _mm_max_ps(a7, a8))
  );

  return _mm_castps_si128(simd_clip_32(c, mi, ma));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode2_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair(a1, a2);
    sort_pair(a3, a4);
    sort_pair(a5, a6);
    sort_pair(a7, a8);

    sort_pair(a1, a3);
    sort_pair(a2, a4);
    sort_pair(a5, a7);
    sort_pair(a6, a8);

    sort_pair(a2, a3);
    sort_pair(a6, a7);

    a5 = _mm_max_epu8 (a1, a5);	// sort_pair (a1, a5);
    sort_pair (a2, a6);
    sort_pair (a3, a7);
    a4 = _mm_min_epu8 (a4, a8);	// sort_pair (a4, a8);

    a3 = _mm_min_epu8 (a3, a5);	// sort_pair (a3, a5);
    a6 = _mm_max_epu8 (a4, a6);	// sort_pair (a4, a6);

    a2 = _mm_min_epu8 (a2, a3);	// sort_pair (a2, a3);
    a7 = _mm_max_epu8 (a6, a7);	// sort_pair (a6, a7);

    return simd_clip(c, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode2_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a2);
  sort_pair(a3, a4);
  sort_pair(a5, a6);
  sort_pair(a7, a8);

  sort_pair(a1, a3);
  sort_pair(a2, a4);
  sort_pair(a5, a7);
  sort_pair(a6, a8);

  sort_pair(a2, a3);
  sort_pair(a6, a7);

  a5 = _mm_max_epu8(a1, a5);	// sort_pair (a1, a5);
  sort_pair(a2, a6);
  sort_pair(a3, a7);
  a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu8(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);

  a2 = _mm_min_epu8(a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_max_epu8(a6, a7);	// sort_pair (a6, a7);

  return simd_clip(c, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode2_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16(a1, a2);
  sort_pair_16(a3, a4);
  sort_pair_16(a5, a6);
  sort_pair_16(a7, a8);

  sort_pair_16(a1, a3);
  sort_pair_16(a2, a4);
  sort_pair_16(a5, a7);
  sort_pair_16(a6, a8);

  sort_pair_16(a2, a3);
  sort_pair_16(a6, a7);

  a5 = _mm_max_epu16 (a1, a5);	// sort_pair (a1, a5);
  sort_pair_16 (a2, a6);
  sort_pair_16 (a3, a7);
  a4 = _mm_min_epu16 (a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu16 (a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu16 (a4, a6);	// sort_pair (a4, a6);

  a2 = _mm_min_epu16 (a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_max_epu16 (a6, a7);	// sort_pair (a6, a7);

  return simd_clip_16(c, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode2_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32(a1, a2);
  sort_pair_32(a3, a4);
  sort_pair_32(a5, a6);
  sort_pair_32(a7, a8);

  sort_pair_32(a1, a3);
  sort_pair_32(a2, a4);
  sort_pair_32(a5, a7);
  sort_pair_32(a6, a8);

  sort_pair_32(a2, a3);
  sort_pair_32(a6, a7);

  a5 = _mm_max_ps (a1, a5);	// sort_pair (a1, a5);
  sort_pair_32 (a2, a6);
  sort_pair_32 (a3, a7);
  a4 = _mm_min_ps (a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_ps (a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_ps (a4, a6);	// sort_pair (a4, a6);

  a2 = _mm_min_ps (a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_max_ps (a6, a7);	// sort_pair (a6, a7);

  return _mm_castps_si128(simd_clip_32(c, a2, a7));
}


//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode3_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair(a1, a2);
    sort_pair(a3, a4);
    sort_pair(a5, a6);
    sort_pair(a7, a8);

    sort_pair(a1, a3);
    sort_pair(a2, a4);
    sort_pair(a5, a7);
    sort_pair(a6, a8);

    sort_pair(a2, a3);
    sort_pair(a6, a7);

    a5 = _mm_max_epu8(a1, a5);	// sort_pair (a1, a5);
    sort_pair (a2, a6);
    sort_pair (a3, a7);
    a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

    a3 = _mm_min_epu8(a3, a5);	// sort_pair (a3, a5);
    a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);

    a3 = _mm_max_epu8(a2, a3);	// sort_pair (a2, a3);
    a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

    return simd_clip(c, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode3_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a2);
  sort_pair(a3, a4);
  sort_pair(a5, a6);
  sort_pair(a7, a8);

  sort_pair(a1, a3);
  sort_pair(a2, a4);
  sort_pair(a5, a7);
  sort_pair(a6, a8);

  sort_pair(a2, a3);
  sort_pair(a6, a7);

  a5 = _mm_max_epu8(a1, a5);	// sort_pair (a1, a5);
  sort_pair(a2, a6);
  sort_pair(a3, a7);
  a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu8(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_epu8(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

  return simd_clip(c, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode3_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16(a1, a2);
  sort_pair_16(a3, a4);
  sort_pair_16(a5, a6);
  sort_pair_16(a7, a8);

  sort_pair_16(a1, a3);
  sort_pair_16(a2, a4);
  sort_pair_16(a5, a7);
  sort_pair_16(a6, a8);

  sort_pair_16(a2, a3);
  sort_pair_16(a6, a7);

  a5 = _mm_max_epu16(a1, a5);	// sort_pair (a1, a5);
  sort_pair_16 (a2, a6);
  sort_pair_16 (a3, a7);
  a4 = _mm_min_epu16(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu16(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu16(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_epu16(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_epu16(a6, a7);	// sort_pair (a6, a7);

  return simd_clip_16(c, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode3_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32(a1, a2);
  sort_pair_32(a3, a4);
  sort_pair_32(a5, a6);
  sort_pair_32(a7, a8);

  sort_pair_32(a1, a3);
  sort_pair_32(a2, a4);
  sort_pair_32(a5, a7);
  sort_pair_32(a6, a8);

  sort_pair_32(a2, a3);
  sort_pair_32(a6, a7);

  a5 = _mm_max_ps(a1, a5);	// sort_pair (a1, a5);
  sort_pair_32 (a2, a6);
  sort_pair_32 (a3, a7);
  a4 = _mm_min_ps(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_ps(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_ps(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_ps(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_ps(a6, a7);	// sort_pair (a6, a7);

  return _mm_castps_si128(simd_clip_32(c, a3, a6));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode4_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair (a1, a2);
    sort_pair (a3, a4);
    sort_pair (a5, a6);
    sort_pair (a7, a8);

    sort_pair (a1, a3);
    sort_pair (a2, a4);
    sort_pair (a5, a7);
    sort_pair (a6, a8);

    sort_pair (a2, a3);
    sort_pair (a6, a7);

    a5 = _mm_max_epu8 (a1, a5);	// sort_pair (a1, a5);
    a6 = _mm_max_epu8 (a2, a6);	// sort_pair (a2, a6);
    a3 = _mm_min_epu8 (a3, a7);	// sort_pair (a3, a7);
    a4 = _mm_min_epu8 (a4, a8);	// sort_pair (a4, a8);

    a5 = _mm_max_epu8 (a3, a5);	// sort_pair (a3, a5);
    a4 = _mm_min_epu8 (a4, a6);	// sort_pair (a4, a6);

    // sort_pair (au82, a3);
    sort_pair (a4, a5);
    // sort_pair (a6, a7);

    return simd_clip(c, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode4_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a2);
  sort_pair(a3, a4);
  sort_pair(a5, a6);
  sort_pair(a7, a8);

  sort_pair(a1, a3);
  sort_pair(a2, a4);
  sort_pair(a5, a7);
  sort_pair(a6, a8);

  sort_pair(a2, a3);
  sort_pair(a6, a7);

  a5 = _mm_max_epu8(a1, a5);	// sort_pair (a1, a5);
  a6 = _mm_max_epu8(a2, a6);	// sort_pair (a2, a6);
  a3 = _mm_min_epu8(a3, a7);	// sort_pair (a3, a7);
  a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

  a5 = _mm_max_epu8(a3, a5);	// sort_pair (a3, a5);
  a4 = _mm_min_epu8(a4, a6);	// sort_pair (a4, a6);

  // sort_pair (au82, a3);
  sort_pair(a4, a5);
  // sort_pair (a6, a7);

  return simd_clip(c, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode4_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16 (a1, a2);
  sort_pair_16 (a3, a4);
  sort_pair_16 (a5, a6);
  sort_pair_16 (a7, a8);

  sort_pair_16 (a1, a3);
  sort_pair_16 (a2, a4);
  sort_pair_16 (a5, a7);
  sort_pair_16 (a6, a8);

  sort_pair_16 (a2, a3);
  sort_pair_16 (a6, a7);

  a5 = _mm_max_epu16 (a1, a5);	// sort_pair_16 (a1, a5);
  a6 = _mm_max_epu16 (a2, a6);	// sort_pair_16 (a2, a6);
  a3 = _mm_min_epu16 (a3, a7);	// sort_pair_16 (a3, a7);
  a4 = _mm_min_epu16 (a4, a8);	// sort_pair_16 (a4, a8);

  a5 = _mm_max_epu16 (a3, a5);	// sort_pair_16 (a3, a5);
  a4 = _mm_min_epu16 (a4, a6);	// sort_pair_16 (a4, a6);

                              // sort_pair_16 (au82, a3);
  sort_pair_16 (a4, a5);
  // sort_pair (a6, a7);

  return simd_clip_16(c, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode4_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32 (a1, a2);
  sort_pair_32 (a3, a4);
  sort_pair_32 (a5, a6);
  sort_pair_32 (a7, a8);

  sort_pair_32 (a1, a3);
  sort_pair_32 (a2, a4);
  sort_pair_32 (a5, a7);
  sort_pair_32 (a6, a8);

  sort_pair_32 (a2, a3);
  sort_pair_32 (a6, a7);

  a5 = _mm_max_ps (a1, a5);	// sort_pair_32 (a1, a5);
  a6 = _mm_max_ps (a2, a6);	// sort_pair_32 (a2, a6);
  a3 = _mm_min_ps (a3, a7);	// sort_pair_32 (a3, a7);
  a4 = _mm_min_ps (a4, a8);	// sort_pair_32 (a4, a8);

  a5 = _mm_max_ps (a3, a5);	// sort_pair_32 (a3, a5);
  a4 = _mm_min_ps (a4, a6);	// sort_pair_32 (a4, a6);

                                // sort_pair_32 (au82, a3);
  sort_pair_32 (a4, a5);
  // sort_pair (a6, a7);

  return _mm_castps_si128(simd_clip_32(c, a4, a5));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode5_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto clipped1 = simd_clip(c, mil1, mal1);
    auto clipped2 = simd_clip(c, mil2, mal2);
    auto clipped3 = simd_clip(c, mil3, mal3);
    auto clipped4 = simd_clip(c, mil4, mal4);

    auto c1 = abs_diff(c, clipped1);
    auto c2 = abs_diff(c, clipped2);
    auto c3 = abs_diff(c, clipped3);
    auto c4 = abs_diff(c, clipped4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, c, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode5_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto clipped1 = simd_clip(c, mil1, mal1);
  auto clipped2 = simd_clip(c, mil2, mal2);
  auto clipped3 = simd_clip(c, mil3, mal3);
  auto clipped4 = simd_clip(c, mil4, mal4);

  auto c1 = abs_diff(c, clipped1);
  auto c2 = abs_diff(c, clipped2);
  auto c3 = abs_diff(c, clipped3);
  auto c4 = abs_diff(c, clipped4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, c, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode5_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto clipped1 = simd_clip_16(c, mil1, mal1);
  auto clipped2 = simd_clip_16(c, mil2, mal2);
  auto clipped3 = simd_clip_16(c, mil3, mal3);
  auto clipped4 = simd_clip_16(c, mil4, mal4);

  auto c1 = abs_diff_16(c, clipped1);
  auto c2 = abs_diff_16(c, clipped2);
  auto c3 = abs_diff_16(c, clipped3);
  auto c4 = abs_diff_16(c, clipped4);

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, c, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode5_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto c1 = abs_diff_32(c, clipped1);
  auto c2 = abs_diff_32(c, clipped2);
  auto c3 = abs_diff_32(c, clipped3);
  auto c4 = abs_diff_32(c, clipped4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode6_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(c, mil1, mal1);
    auto clipped2 = simd_clip(c, mil2, mal2);
    auto clipped3 = simd_clip(c, mil3, mal3);
    auto clipped4 = simd_clip(c, mil4, mal4);

    auto absdiff1 = abs_diff(c, clipped1);
    auto absdiff2 = abs_diff(c, clipped2);
    auto absdiff3 = abs_diff(c, clipped3);
    auto absdiff4 = abs_diff(c, clipped4);
    
    auto c1 = _mm_adds_epu8(_mm_adds_epu8(absdiff1, absdiff1), d1);
    auto c2 = _mm_adds_epu8(_mm_adds_epu8(absdiff2, absdiff2), d2);
    auto c3 = _mm_adds_epu8(_mm_adds_epu8(absdiff3, absdiff3), d3);
    auto c4 = _mm_adds_epu8(_mm_adds_epu8(absdiff4, absdiff4), d4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, c, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode6_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(c, mil1, mal1);
  auto clipped2 = simd_clip(c, mil2, mal2);
  auto clipped3 = simd_clip(c, mil3, mal3);
  auto clipped4 = simd_clip(c, mil4, mal4);

  auto absdiff1 = abs_diff(c, clipped1);
  auto absdiff2 = abs_diff(c, clipped2);
  auto absdiff3 = abs_diff(c, clipped3);
  auto absdiff4 = abs_diff(c, clipped4);

  auto c1 = _mm_adds_epu8(_mm_adds_epu8(absdiff1, absdiff1), d1);
  auto c2 = _mm_adds_epu8(_mm_adds_epu8(absdiff2, absdiff2), d2);
  auto c3 = _mm_adds_epu8(_mm_adds_epu8(absdiff3, absdiff3), d3);
  auto c4 = _mm_adds_epu8(_mm_adds_epu8(absdiff4, absdiff4), d4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, c, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode6_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(c, mil1, mal1);
  auto clipped2 = simd_clip_16(c, mil2, mal2);
  auto clipped3 = simd_clip_16(c, mil3, mal3);
  auto clipped4 = simd_clip_16(c, mil4, mal4);

  auto absdiff1 = abs_diff_16(c, clipped1);
  auto absdiff2 = abs_diff_16(c, clipped2);
  auto absdiff3 = abs_diff_16(c, clipped3);
  auto absdiff4 = abs_diff_16(c, clipped4);

  auto c1 = _mm_adds_epu16(_mm_adds_epu16(absdiff1, absdiff1), d1);
  auto c2 = _mm_adds_epu16(_mm_adds_epu16(absdiff2, absdiff2), d2);
  auto c3 = _mm_adds_epu16(_mm_adds_epu16(absdiff3, absdiff3), d3);
  auto c4 = _mm_adds_epu16(_mm_adds_epu16(absdiff4, absdiff4), d4);

  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    c1 = _mm_min_epu16(c1, pixel_max);
    c2 = _mm_min_epu16(c2, pixel_max);
    c3 = _mm_min_epu16(c3, pixel_max);
    c4 = _mm_min_epu16(c4, pixel_max);
  }

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, c, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode6_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto absdiff1 = abs_diff_32(c, clipped1);
  auto absdiff2 = abs_diff_32(c, clipped2);
  auto absdiff3 = abs_diff_32(c, clipped3);
  auto absdiff4 = abs_diff_32(c, clipped4);

  auto c1 = _mm_add_ps(_mm_add_ps(absdiff1, absdiff1), d1); // saturated add not needed, plus: absdiff is simply doubled
  auto c2 = _mm_add_ps(_mm_add_ps(absdiff2, absdiff2), d2);
  auto c3 = _mm_add_ps(_mm_add_ps(absdiff3, absdiff3), d3);
  auto c4 = _mm_add_ps(_mm_add_ps(absdiff4, absdiff4), d4);

  // no max_pixel_value clamp for float

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1); // clipped1 when min(c1,c2,c3,c4) == c1
  result = select_on_equal_32(mindiff, c3, result, clipped3); // clipped3 when min(c1,c2,c3,c4) == c3
  result = select_on_equal_32(mindiff, c2, result, clipped2); // clipped2 when min(c1,c2,c3,c4) == c2
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4)); // clipped4 when min(c1,c2,c3,c4) == c4
}


//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode7_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(c, mil1, mal1);
    auto clipped2 = simd_clip(c, mil2, mal2);
    auto clipped3 = simd_clip(c, mil3, mal3);
    auto clipped4 = simd_clip(c, mil4, mal4);
    //todo: what happens when this overflows?
    auto c1 = _mm_adds_epu8(abs_diff(c, clipped1), d1);
    auto c2 = _mm_adds_epu8(abs_diff(c, clipped2), d2);
    auto c3 = _mm_adds_epu8(abs_diff(c, clipped3), d3);
    auto c4 = _mm_adds_epu8(abs_diff(c, clipped4), d4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, c, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode7_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(c, mil1, mal1);
  auto clipped2 = simd_clip(c, mil2, mal2);
  auto clipped3 = simd_clip(c, mil3, mal3);
  auto clipped4 = simd_clip(c, mil4, mal4);
  //todo: what happens when this overflows?
  auto c1 = _mm_adds_epu8(abs_diff(c, clipped1), d1);
  auto c2 = _mm_adds_epu8(abs_diff(c, clipped2), d2);
  auto c3 = _mm_adds_epu8(abs_diff(c, clipped3), d3);
  auto c4 = _mm_adds_epu8(abs_diff(c, clipped4), d4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, c, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode7_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(c, mil1, mal1);
  auto clipped2 = simd_clip_16(c, mil2, mal2);
  auto clipped3 = simd_clip_16(c, mil3, mal3);
  auto clipped4 = simd_clip_16(c, mil4, mal4);

  auto c1 = _mm_adds_epu16(abs_diff_16(c, clipped1), d1);
  auto c2 = _mm_adds_epu16(abs_diff_16(c, clipped2), d2);
  auto c3 = _mm_adds_epu16(abs_diff_16(c, clipped3), d3);
  auto c4 = _mm_adds_epu16(abs_diff_16(c, clipped4), d4);

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, c, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode7_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto c1 = _mm_add_ps(abs_diff_32(c, clipped1), d1); // only comparison, saturated add not needed
  auto c2 = _mm_add_ps(abs_diff_32(c, clipped2), d2);
  auto c3 = _mm_add_ps(abs_diff_32(c, clipped3), d3);
  auto c4 = _mm_add_ps(abs_diff_32(c, clipped4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1); // clipped1 when min(c1,c2,c3,c4) == c1
  result = select_on_equal_32(mindiff, c3, result, clipped3); // clipped3 when min(c1,c2,c3,c4) == c3
  result = select_on_equal_32(mindiff, c2, result, clipped2); // clipped2 when min(c1,c2,c3,c4) == c2
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4)); // clipped4 when min(c1,c2,c3,c4) == c4
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode8_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(c, mil1, mal1);
    auto clipped2 = simd_clip(c, mil2, mal2);
    auto clipped3 = simd_clip(c, mil3, mal3);
    auto clipped4 = simd_clip(c, mil4, mal4);

    auto c1 = _mm_adds_epu8(abs_diff(c, clipped1), _mm_adds_epu8(d1, d1));
    auto c2 = _mm_adds_epu8(abs_diff(c, clipped2), _mm_adds_epu8(d2, d2));
    auto c3 = _mm_adds_epu8(abs_diff(c, clipped3), _mm_adds_epu8(d3, d3));
    auto c4 = _mm_adds_epu8(abs_diff(c, clipped4), _mm_adds_epu8(d4, d4));

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, c, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode8_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(c, mil1, mal1);
  auto clipped2 = simd_clip(c, mil2, mal2);
  auto clipped3 = simd_clip(c, mil3, mal3);
  auto clipped4 = simd_clip(c, mil4, mal4);

  auto c1 = _mm_adds_epu8(abs_diff(c, clipped1), _mm_adds_epu8(d1, d1));
  auto c2 = _mm_adds_epu8(abs_diff(c, clipped2), _mm_adds_epu8(d2, d2));
  auto c3 = _mm_adds_epu8(abs_diff(c, clipped3), _mm_adds_epu8(d3, d3));
  auto c4 = _mm_adds_epu8(abs_diff(c, clipped4), _mm_adds_epu8(d4, d4));

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, c, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode8_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(c, mil1, mal1);
  auto clipped2 = simd_clip_16(c, mil2, mal2);
  auto clipped3 = simd_clip_16(c, mil3, mal3);
  auto clipped4 = simd_clip_16(c, mil4, mal4);

  auto c1 = _mm_adds_epu16(abs_diff_16(c, clipped1), _mm_adds_epu16(d1, d1));
  auto c2 = _mm_adds_epu16(abs_diff_16(c, clipped2), _mm_adds_epu16(d2, d2));
  auto c3 = _mm_adds_epu16(abs_diff_16(c, clipped3), _mm_adds_epu16(d3, d3));
  auto c4 = _mm_adds_epu16(abs_diff_16(c, clipped4), _mm_adds_epu16(d4, d4));

  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    c1 = _mm_min_epu16(c1, pixel_max);
    c2 = _mm_min_epu16(c2, pixel_max);
    c3 = _mm_min_epu16(c3, pixel_max);
    c4 = _mm_min_epu16(c4, pixel_max);
  }

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, c, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode8_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto c1 = _mm_add_ps(abs_diff_32(c, clipped1), _mm_add_ps(d1, d1)); // no adds needed, only comparison
  auto c2 = _mm_add_ps(abs_diff_32(c, clipped2), _mm_add_ps(d2, d2));
  auto c3 = _mm_add_ps(abs_diff_32(c, clipped3), _mm_add_ps(d3, d3));
  auto c4 = _mm_add_ps(abs_diff_32(c, clipped4), _mm_add_ps(d4, d4));

  // no max_pixel_value clamp for float

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode9_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);

    auto result = select_on_equal(mindiff, d1, c, simd_clip(c, mil1, mal1));
    result = select_on_equal(mindiff, d3, result, simd_clip(c, mil3, mal3));
    result = select_on_equal(mindiff, d2, result, simd_clip(c, mil2, mal2));
    return select_on_equal(mindiff, d4, result, simd_clip(c, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode9_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);

  auto result = select_on_equal_sse4(mindiff, d1, c, simd_clip(c, mil1, mal1));
  result = select_on_equal_sse4(mindiff, d3, result, simd_clip(c, mil3, mal3));
  result = select_on_equal_sse4(mindiff, d2, result, simd_clip(c, mil2, mal2));
  return select_on_equal_sse4(mindiff, d4, result, simd_clip(c, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode9_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);

  auto result = select_on_equal_16(mindiff, d1, c, simd_clip_16(c, mil1, mal1));
  result = select_on_equal_16(mindiff, d3, result, simd_clip_16(c, mil3, mal3));
  result = select_on_equal_16(mindiff, d2, result, simd_clip_16(c, mil2, mal2));
  return select_on_equal_16(mindiff, d4, result, simd_clip_16(c, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode9_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);

  auto result = select_on_equal_32(mindiff, d1, c, simd_clip_32(c, mil1, mal1));
  result = select_on_equal_32(mindiff, d3, result, simd_clip_32(c, mil3, mal3));
  result = select_on_equal_32(mindiff, d2, result, simd_clip_32(c, mil2, mal2));
  return _mm_castps_si128(select_on_equal_32(mindiff, d4, result, simd_clip_32(c, mil4, mal4)));
}


//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode10_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto d1 = abs_diff(c, a1);
    auto d2 = abs_diff(c, a2);
    auto d3 = abs_diff(c, a3);
    auto d4 = abs_diff(c, a4);
    auto d5 = abs_diff(c, a5);
    auto d6 = abs_diff(c, a6);
    auto d7 = abs_diff(c, a7);
    auto d8 = abs_diff(c, a8);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);
    mindiff = _mm_min_epu8(mindiff, d5);
    mindiff = _mm_min_epu8(mindiff, d6);
    mindiff = _mm_min_epu8(mindiff, d7);
    mindiff = _mm_min_epu8(mindiff, d8);

    auto result = select_on_equal(mindiff, d4, c, a4);
    result = select_on_equal(mindiff, d5, result, a5);
    result = select_on_equal(mindiff, d1, result, a1);
    result = select_on_equal(mindiff, d3, result, a3);
    result = select_on_equal(mindiff, d2, result, a2);
    result = select_on_equal(mindiff, d6, result, a6);
    result = select_on_equal(mindiff, d8, result, a8);
    return select_on_equal(mindiff, d7, result, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode10_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff(c, a1);
  auto d2 = abs_diff(c, a2);
  auto d3 = abs_diff(c, a3);
  auto d4 = abs_diff(c, a4);
  auto d5 = abs_diff(c, a5);
  auto d6 = abs_diff(c, a6);
  auto d7 = abs_diff(c, a7);
  auto d8 = abs_diff(c, a8);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);
  mindiff = _mm_min_epu8(mindiff, d5);
  mindiff = _mm_min_epu8(mindiff, d6);
  mindiff = _mm_min_epu8(mindiff, d7);
  mindiff = _mm_min_epu8(mindiff, d8);

  auto result = select_on_equal_sse4(mindiff, d4, c, a4);
  result = select_on_equal_sse4(mindiff, d5, result, a5);
  result = select_on_equal_sse4(mindiff, d1, result, a1);
  result = select_on_equal_sse4(mindiff, d3, result, a3);
  result = select_on_equal_sse4(mindiff, d2, result, a2);
  result = select_on_equal_sse4(mindiff, d6, result, a6);
  result = select_on_equal_sse4(mindiff, d8, result, a8);
  return select_on_equal(mindiff, d7, result, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode10_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_16(c, a1);
  auto d2 = abs_diff_16(c, a2);
  auto d3 = abs_diff_16(c, a3);
  auto d4 = abs_diff_16(c, a4);
  auto d5 = abs_diff_16(c, a5);
  auto d6 = abs_diff_16(c, a6);
  auto d7 = abs_diff_16(c, a7);
  auto d8 = abs_diff_16(c, a8);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);
  mindiff = _mm_min_epu16(mindiff, d5);
  mindiff = _mm_min_epu16(mindiff, d6);
  mindiff = _mm_min_epu16(mindiff, d7);
  mindiff = _mm_min_epu16(mindiff, d8);

  auto result = select_on_equal_16(mindiff, d4, c, a4);
  result = select_on_equal_16(mindiff, d5, result, a5);
  result = select_on_equal_16(mindiff, d1, result, a1);
  result = select_on_equal_16(mindiff, d3, result, a3);
  result = select_on_equal_16(mindiff, d2, result, a2);
  result = select_on_equal_16(mindiff, d6, result, a6);
  result = select_on_equal_16(mindiff, d8, result, a8);
  return select_on_equal_16(mindiff, d7, result, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode10_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_32(c, a1);
  auto d2 = abs_diff_32(c, a2);
  auto d3 = abs_diff_32(c, a3);
  auto d4 = abs_diff_32(c, a4);
  auto d5 = abs_diff_32(c, a5);
  auto d6 = abs_diff_32(c, a6);
  auto d7 = abs_diff_32(c, a7);
  auto d8 = abs_diff_32(c, a8);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);
  mindiff = _mm_min_ps(mindiff, d5);
  mindiff = _mm_min_ps(mindiff, d6);
  mindiff = _mm_min_ps(mindiff, d7);
  mindiff = _mm_min_ps(mindiff, d8);

  auto result = select_on_equal_32(mindiff, d4, c, a4);
  result = select_on_equal_32(mindiff, d5, result, a5);
  result = select_on_equal_32(mindiff, d1, result, a1);
  result = select_on_equal_32(mindiff, d3, result, a3);
  result = select_on_equal_32(mindiff, d2, result, a2);
  result = select_on_equal_32(mindiff, d6, result, a6);
  result = select_on_equal_32(mindiff, d8, result, a8);
  return _mm_castps_si128(select_on_equal_32(mindiff, d7, result, a7));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode12_sse2(const Byte* pSrc, int srcPitch);

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch);

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse_16(const Byte* pSrc, int srcPitch);

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse_32(const Byte* pSrc, int srcPitch);

//todo: actually implement is as mode 11
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode11_sse2(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse2<aligned>(pSrc, srcPitch);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode11_sse(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse<aligned>(pSrc, srcPitch);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode11_sse_16(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse_16<aligned>(pSrc, srcPitch);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode11_sse_32(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse_32<aligned>(pSrc, srcPitch);
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode12_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    // different from C!
    // int	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;


    auto a13  = _mm_avg_epu8 (a1, a3);
    auto a123 = _mm_avg_epu8 (a2, a13);

    auto a68  = _mm_avg_epu8 (a6, a8);
    auto a678 = _mm_avg_epu8 (a7, a68);

    auto a45  = _mm_avg_epu8 (a4, a5);
    auto a4c5 = _mm_avg_epu8 (c, a45);

    auto a123678  = _mm_avg_epu8 (a123, a678);
    auto a123678b = _mm_subs_epu8 (a123678, _mm_set1_epi8(1));

    return _mm_avg_epu8 (a4c5, a123678b);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto a13 = _mm_avg_epu8(a1, a3);
  auto a123 = _mm_avg_epu8(a2, a13);

  auto a68 = _mm_avg_epu8(a6, a8);
  auto a678 = _mm_avg_epu8(a7, a68);

  auto a45 = _mm_avg_epu8(a4, a5);
  auto a4c5 = _mm_avg_epu8(c, a45);

  auto a123678 = _mm_avg_epu8(a123, a678);
  auto a123678b = _mm_subs_epu8(a123678, _mm_set1_epi8(1));

  return _mm_avg_epu8(a4c5, a123678b);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto a13  = _mm_avg_epu16 (a1, a3);
  auto a123 = _mm_avg_epu16 (a2, a13);

  auto a68  = _mm_avg_epu16 (a6, a8);
  auto a678 = _mm_avg_epu16 (a7, a68);

  auto a45  = _mm_avg_epu16 (a4, a5);
  auto a4c5 = _mm_avg_epu16 (c, a45);

  auto a123678  = _mm_avg_epu16 (a123, a678);
  auto a123678b = _mm_subs_epu16 (a123678, _mm_set1_epi16(1));

  return _mm_avg_epu16 (a4c5, a123678b);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode12_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto a13  = _mm_avg_ps (a1, a3);
  auto a123 = _mm_avg_ps (a2, a13);

  auto a68  = _mm_avg_ps (a6, a8);
  auto a678 = _mm_avg_ps (a7, a68);

  auto a45  = _mm_avg_ps (a4, a5);
  auto a4c5 = _mm_avg_ps (c, a45);

  auto a123678  = _mm_avg_ps (a123, a678);
  // no rounding at float: auto a123678b = _mm_subs_ps (a123678, _mm_set1_epi16(1));

  return _mm_castps_si128(_mm_avg_ps (a4c5, a123678));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode13_and14_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto d1 = abs_diff(a1, a8);
    auto d2 = abs_diff(a2, a7);
    auto d3 = abs_diff(a3, a6);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);

    auto result = select_on_equal(mindiff, d1, c, _mm_avg_epu8(a1, a8));
    result = select_on_equal(mindiff, d3, result,  _mm_avg_epu8(a3, a6));
    return select_on_equal(mindiff, d2, result,  _mm_avg_epu8(a2, a7));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode13_and14_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff(a1, a8);
  auto d2 = abs_diff(a2, a7);
  auto d3 = abs_diff(a3, a6);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);

  auto result = select_on_equal_sse4(mindiff, d1, c, _mm_avg_epu8(a1, a8));
  result = select_on_equal_sse4(mindiff, d3, result, _mm_avg_epu8(a3, a6));
  return select_on_equal_sse4(mindiff, d2, result, _mm_avg_epu8(a2, a7));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode13_and14_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_16(a1, a8);
  auto d2 = abs_diff_16(a2, a7);
  auto d3 = abs_diff_16(a3, a6);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);

  auto result = select_on_equal_16(mindiff, d1, c, _mm_avg_epu16(a1, a8));
  result = select_on_equal_16(mindiff, d3, result,  _mm_avg_epu16(a3, a6));
  return select_on_equal_16(mindiff, d2, result,  _mm_avg_epu16(a2, a7));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode13_and14_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_32(a1, a8);
  auto d2 = abs_diff_32(a2, a7);
  auto d3 = abs_diff_32(a3, a6);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);

  auto result = select_on_equal_32(mindiff, d1, c, _mm_avg_ps(a1, a8));
  result = select_on_equal_32(mindiff, d3, result,  _mm_avg_ps(a3, a6));
  return _mm_castps_si128(select_on_equal_32(mindiff, d2, result,  _mm_avg_ps(a2, a7)));
}

//-------------------

//rounding does not match with decade-old original
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode15_and16_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto max18 = _mm_max_epu8(a1, a8);
    auto min18 = _mm_min_epu8(a1, a8);

    auto max27 = _mm_max_epu8(a2, a7);
    auto min27 = _mm_min_epu8(a2, a7);

    auto max36 = _mm_max_epu8(a3, a6);
    auto min36 = _mm_min_epu8(a3, a6);

    auto d1 = _mm_subs_epu8(max18, min18);
    auto d2 = _mm_subs_epu8(max27, min27);
    auto d3 = _mm_subs_epu8(max36, min36);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);

    auto avg12 = _mm_avg_epu8(a1, a2);
    auto avg23 = _mm_avg_epu8(a2, a3);
    auto avg67 = _mm_avg_epu8(a6, a7);
    auto avg78 = _mm_avg_epu8(a7, a8);

    auto avg1223 = _mm_avg_epu8(avg12, avg23);

    auto avg6778 = _mm_avg_epu8(avg67, avg78);
    auto avg6778b = _mm_subs_epu8(avg6778, _mm_set1_epi8(1));
    auto avg = _mm_avg_epu8(avg1223, avg6778b);
    

    auto result = select_on_equal(mindiff, d1, c, simd_clip(avg, min18, max18));
    result = select_on_equal(mindiff, d3, result, simd_clip(avg, min36, max36));
    return select_on_equal(mindiff, d2, result, simd_clip(avg, min27, max27));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode15_and16_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto max18 = _mm_max_epu8(a1, a8);
  auto min18 = _mm_min_epu8(a1, a8);

  auto max27 = _mm_max_epu8(a2, a7);
  auto min27 = _mm_min_epu8(a2, a7);

  auto max36 = _mm_max_epu8(a3, a6);
  auto min36 = _mm_min_epu8(a3, a6);

  auto d1 = _mm_subs_epu8(max18, min18);
  auto d2 = _mm_subs_epu8(max27, min27);
  auto d3 = _mm_subs_epu8(max36, min36);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);

  auto avg12 = _mm_avg_epu8(a1, a2);
  auto avg23 = _mm_avg_epu8(a2, a3);
  auto avg67 = _mm_avg_epu8(a6, a7);
  auto avg78 = _mm_avg_epu8(a7, a8);

  auto avg1223 = _mm_avg_epu8(avg12, avg23);

  auto avg6778 = _mm_avg_epu8(avg67, avg78);
  auto avg6778b = _mm_subs_epu8(avg6778, _mm_set1_epi8(1));
  auto avg = _mm_avg_epu8(avg1223, avg6778b);


  auto result = select_on_equal_sse4(mindiff, d1, c, simd_clip(avg, min18, max18));
  result = select_on_equal_sse4(mindiff, d3, result, simd_clip(avg, min36, max36));
  return select_on_equal_sse4(mindiff, d2, result, simd_clip(avg, min27, max27));
}


template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode15_and16_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto max18 = _mm_max_epu16(a1, a8);
  auto min18 = _mm_min_epu16(a1, a8);

  auto max27 = _mm_max_epu16(a2, a7);
  auto min27 = _mm_min_epu16(a2, a7);

  auto max36 = _mm_max_epu16(a3, a6);
  auto min36 = _mm_min_epu16(a3, a6);

  auto d1 = _mm_subs_epu16(max18, min18);
  auto d2 = _mm_subs_epu16(max27, min27);
  auto d3 = _mm_subs_epu16(max36, min36);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);

  auto avg12 = _mm_avg_epu16(a1, a2);
  auto avg23 = _mm_avg_epu16(a2, a3);
  auto avg67 = _mm_avg_epu16(a6, a7);
  auto avg78 = _mm_avg_epu16(a7, a8);

  auto avg1223 = _mm_avg_epu16(avg12, avg23);

  auto avg6778 = _mm_avg_epu16(avg67, avg78);
  auto avg6778b = _mm_subs_epu16(avg6778, _mm_set1_epi16(1));
  auto avg = _mm_avg_epu16(avg1223, avg6778b);


  auto result = select_on_equal_16(mindiff, d1, c, simd_clip_16(avg, min18, max18));
  result = select_on_equal_16(mindiff, d3, result, simd_clip_16(avg, min36, max36));
  return select_on_equal_16(mindiff, d2, result, simd_clip_16(avg, min27, max27));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode15_and16_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto max18 = _mm_max_ps(a1, a8);
  auto min18 = _mm_min_ps(a1, a8);

  auto max27 = _mm_max_ps(a2, a7);
  auto min27 = _mm_min_ps(a2, a7);

  auto max36 = _mm_max_ps(a3, a6);
  auto min36 = _mm_min_ps(a3, a6);

  auto d1 = _mm_subs_ps_for_diff(max18, min18);
  auto d2 = _mm_subs_ps_for_diff(max27, min27);
  auto d3 = _mm_subs_ps_for_diff(max36, min36);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);

  auto avg12 = _mm_avg_ps(a1, a2);
  auto avg23 = _mm_avg_ps(a2, a3);
  auto avg67 = _mm_avg_ps(a6, a7);
  auto avg78 = _mm_avg_ps(a7, a8);

  auto avg1223 = _mm_avg_ps(avg12, avg23);

  auto avg6778 = _mm_avg_ps(avg67, avg78);
  // no rounding here at float: auto avg6778b = _mm_subs_ps(avg6778, _mm_set1_epi16(1));
  auto avg = _mm_avg_ps(avg1223, avg6778);


  auto result = select_on_equal_32(mindiff, d1, c, simd_clip_32(avg, min18, max18));
  result = select_on_equal_32(mindiff, d3, result, simd_clip_32(avg, min36, max36));
  return _mm_castps_si128(select_on_equal_32(mindiff, d2, result, simd_clip_32(avg, min27, max27)));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode17_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto lower = _mm_max_epu8(mil1, mil2);
    lower = _mm_max_epu8(lower, mil3);
    lower = _mm_max_epu8(lower, mil4);

    auto upper = _mm_min_epu8(mal1, mal2);
    upper = _mm_min_epu8(upper, mal3);
    upper = _mm_min_epu8(upper, mal4);

    auto real_upper = _mm_max_epu8(upper, lower);
    auto real_lower = _mm_min_epu8(upper, lower);

    return simd_clip(c, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode17_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto lower = _mm_max_epu8(mil1, mil2);
  lower = _mm_max_epu8(lower, mil3);
  lower = _mm_max_epu8(lower, mil4);

  auto upper = _mm_min_epu8(mal1, mal2);
  upper = _mm_min_epu8(upper, mal3);
  upper = _mm_min_epu8(upper, mal4);

  auto real_upper = _mm_max_epu8(upper, lower);
  auto real_lower = _mm_min_epu8(upper, lower);

  return simd_clip(c, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode17_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto lower = _mm_max_epu16(mil1, mil2);
  lower = _mm_max_epu16(lower, mil3);
  lower = _mm_max_epu16(lower, mil4);

  auto upper = _mm_min_epu16(mal1, mal2);
  upper = _mm_min_epu16(upper, mal3);
  upper = _mm_min_epu16(upper, mal4);

  auto real_upper = _mm_max_epu16(upper, lower);
  auto real_lower = _mm_min_epu16(upper, lower);

  return simd_clip_16(c, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode17_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto lower = _mm_max_ps(mil1, mil2);
  lower = _mm_max_ps(lower, mil3);
  lower = _mm_max_ps(lower, mil4);

  auto upper = _mm_min_ps(mal1, mal2);
  upper = _mm_min_ps(upper, mal3);
  upper = _mm_min_ps(upper, mal4);

  auto real_upper = _mm_max_ps(upper, lower);
  auto real_lower = _mm_min_ps(upper, lower);

  return _mm_castps_si128(simd_clip_32(c, real_lower, real_upper));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode18_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto absdiff1 = abs_diff(c, a1);
    auto absdiff2 = abs_diff(c, a2);
    auto absdiff3 = abs_diff(c, a3);
    auto absdiff4 = abs_diff(c, a4);
    auto absdiff5 = abs_diff(c, a5);
    auto absdiff6 = abs_diff(c, a6);
    auto absdiff7 = abs_diff(c, a7);
    auto absdiff8 = abs_diff(c, a8);

    auto d1 = _mm_max_epu8(absdiff1, absdiff8);
    auto d2 = _mm_max_epu8(absdiff2, absdiff7);
    auto d3 = _mm_max_epu8(absdiff3, absdiff6);
    auto d4 = _mm_max_epu8(absdiff4, absdiff5);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);
    
    __m128i c1 = simd_clip(c, _mm_min_epu8(a1, a8), _mm_max_epu8(a1, a8));
    __m128i c2 = simd_clip(c, _mm_min_epu8(a2, a7), _mm_max_epu8(a2, a7));
    __m128i c3 = simd_clip(c, _mm_min_epu8(a3, a6), _mm_max_epu8(a3, a6));
    __m128i c4 = simd_clip(c, _mm_min_epu8(a4, a5), _mm_max_epu8(a4, a5));
    
    auto result = select_on_equal(mindiff, d1, c, c1);
    result = select_on_equal(mindiff, d3, result, c3);
    result = select_on_equal(mindiff, d2, result, c2);
    return select_on_equal(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode18_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto absdiff1 = abs_diff(c, a1);
  auto absdiff2 = abs_diff(c, a2);
  auto absdiff3 = abs_diff(c, a3);
  auto absdiff4 = abs_diff(c, a4);
  auto absdiff5 = abs_diff(c, a5);
  auto absdiff6 = abs_diff(c, a6);
  auto absdiff7 = abs_diff(c, a7);
  auto absdiff8 = abs_diff(c, a8);

  auto d1 = _mm_max_epu8(absdiff1, absdiff8);
  auto d2 = _mm_max_epu8(absdiff2, absdiff7);
  auto d3 = _mm_max_epu8(absdiff3, absdiff6);
  auto d4 = _mm_max_epu8(absdiff4, absdiff5);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);

  __m128i c1 = simd_clip(c, _mm_min_epu8(a1, a8), _mm_max_epu8(a1, a8));
  __m128i c2 = simd_clip(c, _mm_min_epu8(a2, a7), _mm_max_epu8(a2, a7));
  __m128i c3 = simd_clip(c, _mm_min_epu8(a3, a6), _mm_max_epu8(a3, a6));
  __m128i c4 = simd_clip(c, _mm_min_epu8(a4, a5), _mm_max_epu8(a4, a5));

  auto result = select_on_equal_sse4(mindiff, d1, c, c1);
  result = select_on_equal_sse4(mindiff, d3, result, c3);
  result = select_on_equal_sse4(mindiff, d2, result, c2);
  return select_on_equal_sse4(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode18_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto absdiff1 = abs_diff_16(c, a1);
  auto absdiff2 = abs_diff_16(c, a2);
  auto absdiff3 = abs_diff_16(c, a3);
  auto absdiff4 = abs_diff_16(c, a4);
  auto absdiff5 = abs_diff_16(c, a5);
  auto absdiff6 = abs_diff_16(c, a6);
  auto absdiff7 = abs_diff_16(c, a7);
  auto absdiff8 = abs_diff_16(c, a8);

  auto d1 = _mm_max_epu16(absdiff1, absdiff8);
  auto d2 = _mm_max_epu16(absdiff2, absdiff7);
  auto d3 = _mm_max_epu16(absdiff3, absdiff6);
  auto d4 = _mm_max_epu16(absdiff4, absdiff5);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);

  __m128i c1 = simd_clip_16(c, _mm_min_epu16(a1, a8), _mm_max_epu16(a1, a8));
  __m128i c2 = simd_clip_16(c, _mm_min_epu16(a2, a7), _mm_max_epu16(a2, a7));
  __m128i c3 = simd_clip_16(c, _mm_min_epu16(a3, a6), _mm_max_epu16(a3, a6));
  __m128i c4 = simd_clip_16(c, _mm_min_epu16(a4, a5), _mm_max_epu16(a4, a5));

  auto result = select_on_equal_16(mindiff, d1, c, c1);
  result = select_on_equal_16(mindiff, d3, result, c3);
  result = select_on_equal_16(mindiff, d2, result, c2);
  return select_on_equal_16(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode18_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto absdiff1 = abs_diff_32(c, a1);
  auto absdiff2 = abs_diff_32(c, a2);
  auto absdiff3 = abs_diff_32(c, a3);
  auto absdiff4 = abs_diff_32(c, a4);
  auto absdiff5 = abs_diff_32(c, a5);
  auto absdiff6 = abs_diff_32(c, a6);
  auto absdiff7 = abs_diff_32(c, a7);
  auto absdiff8 = abs_diff_32(c, a8);

  auto d1 = _mm_max_ps(absdiff1, absdiff8);
  auto d2 = _mm_max_ps(absdiff2, absdiff7);
  auto d3 = _mm_max_ps(absdiff3, absdiff6);
  auto d4 = _mm_max_ps(absdiff4, absdiff5);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);

  __m128 c1 = simd_clip_32(c, _mm_min_ps(a1, a8), _mm_max_ps(a1, a8));
  __m128 c2 = simd_clip_32(c, _mm_min_ps(a2, a7), _mm_max_ps(a2, a7));
  __m128 c3 = simd_clip_32(c, _mm_min_ps(a3, a6), _mm_max_ps(a3, a6));
  __m128 c4 = simd_clip_32(c, _mm_min_ps(a4, a5), _mm_max_ps(a4, a5));

  auto result = select_on_equal_32(mindiff, d1, c, c1);
  result = select_on_equal_32(mindiff, d3, result, c3);
  result = select_on_equal_32(mindiff, d2, result, c2);
  return _mm_castps_si128(select_on_equal_32(mindiff, d4, result, c4));
}


//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode19_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto a13    = _mm_avg_epu8 (a1, a3);
    auto a68    = _mm_avg_epu8 (a6, a8);
    auto a1368  = _mm_avg_epu8 (a13, a68);
    auto a1368b = _mm_subs_epu8 (a1368, _mm_set1_epi8(1));
    auto a25    = _mm_avg_epu8 (a2, a5);
    auto a47    = _mm_avg_epu8 (a4, a7);
    auto a2457  = _mm_avg_epu8 (a25, a47);
    auto val    = _mm_avg_epu8 (a1368b, a2457);

    return val;
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode19_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto a13 = _mm_avg_epu8(a1, a3);
  auto a68 = _mm_avg_epu8(a6, a8);
  auto a1368 = _mm_avg_epu8(a13, a68);
  auto a1368b = _mm_subs_epu8(a1368, _mm_set1_epi8(1));
  auto a25 = _mm_avg_epu8(a2, a5);
  auto a47 = _mm_avg_epu8(a4, a7);
  auto a2457 = _mm_avg_epu8(a25, a47);
  auto val = _mm_avg_epu8(a1368b, a2457);

  return val;
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode19_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto a13    = _mm_avg_epu16 (a1, a3);
  auto a68    = _mm_avg_epu16 (a6, a8);
  auto a1368  = _mm_avg_epu16 (a13, a68);
  auto a1368b = _mm_subs_epu16 (a1368, _mm_set1_epi8(1));
  auto a25    = _mm_avg_epu16 (a2, a5);
  auto a47    = _mm_avg_epu16 (a4, a7);
  auto a2457  = _mm_avg_epu16 (a25, a47);
  auto val    = _mm_avg_epu16 (a1368b, a2457);

  return val;
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode19_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto a13    = _mm_avg_ps (a1, a3);
  auto a68    = _mm_avg_ps (a6, a8);
  auto a1368  = _mm_avg_ps (a13, a68);
  // no rounding here auto a1368b = _mm_subs_ps (a1368, _mm_set1_epi8(1));
  auto a25    = _mm_avg_ps (a2, a5);
  auto a47    = _mm_avg_ps (a4, a7);
  auto a2457  = _mm_avg_ps (a25, a47);
  auto val    = _mm_avg_ps (a1368, a2457); // no b

  return _mm_castps_si128(val);
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode20_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto zero = _mm_setzero_si128();
    auto onenineth = _mm_set1_epi16((unsigned short)(((1u << 16) + 4) / 9));
    auto bias = _mm_set1_epi16(4);

    auto a1unpck_lo = _mm_unpacklo_epi8(a1, zero);
    auto a2unpck_lo = _mm_unpacklo_epi8(a2, zero);
    auto a3unpck_lo = _mm_unpacklo_epi8(a3, zero);
    auto a4unpck_lo = _mm_unpacklo_epi8(a4, zero);
    auto a5unpck_lo = _mm_unpacklo_epi8(a5, zero);
    auto a6unpck_lo = _mm_unpacklo_epi8(a6, zero);
    auto a7unpck_lo = _mm_unpacklo_epi8(a7, zero);
    auto a8unpck_lo = _mm_unpacklo_epi8(a8, zero);
    auto cunpck_lo  = _mm_unpacklo_epi8(c, zero);

    auto sum_t1 = _mm_adds_epu16(a1unpck_lo, a2unpck_lo);
    sum_t1 = _mm_adds_epu16(sum_t1, a3unpck_lo);
    sum_t1 = _mm_adds_epu16(sum_t1, a4unpck_lo);

    auto sum_t2 = _mm_adds_epu16(a5unpck_lo, a6unpck_lo);
    sum_t2 = _mm_adds_epu16(sum_t2, a7unpck_lo);
    sum_t2 = _mm_adds_epu16(sum_t2, a8unpck_lo);

    auto sum = _mm_adds_epu16(sum_t1, sum_t2);
    sum = _mm_adds_epu16(sum, cunpck_lo);
    sum = _mm_adds_epu16(sum, bias);
    
    auto result_lo = _mm_mulhi_epu16(sum, onenineth);
    

    auto a1unpck_hi = _mm_unpackhi_epi8(a1, zero);
    auto a2unpck_hi = _mm_unpackhi_epi8(a2, zero);
    auto a3unpck_hi = _mm_unpackhi_epi8(a3, zero);
    auto a4unpck_hi = _mm_unpackhi_epi8(a4, zero);
    auto a5unpck_hi = _mm_unpackhi_epi8(a5, zero);
    auto a6unpck_hi = _mm_unpackhi_epi8(a6, zero);
    auto a7unpck_hi = _mm_unpackhi_epi8(a7, zero);
    auto a8unpck_hi = _mm_unpackhi_epi8(a8, zero);
    auto cunpck_hi  = _mm_unpackhi_epi8(c, zero);

    sum_t1 = _mm_adds_epu16(a1unpck_hi, a2unpck_hi);
    sum_t1 = _mm_adds_epu16(sum_t1, a3unpck_hi);
    sum_t1 = _mm_adds_epu16(sum_t1, a4unpck_hi);

    sum_t2 = _mm_adds_epu16(a5unpck_hi, a6unpck_hi);
    sum_t2 = _mm_adds_epu16(sum_t2, a7unpck_hi);
    sum_t2 = _mm_adds_epu16(sum_t2, a8unpck_hi);

    sum = _mm_adds_epu16(sum_t1, sum_t2);
    sum = _mm_adds_epu16(sum, cunpck_hi);
    sum = _mm_adds_epu16(sum, bias);

    auto result_hi = _mm_mulhi_epu16(sum, onenineth);
    
    return _mm_packus_epi16(result_lo, result_hi);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode20_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto zero = _mm_setzero_si128();
  auto onenineth = _mm_set1_epi16((unsigned short)(((1u << 16) + 4) / 9));
  auto bias = _mm_set1_epi16(4);

  auto a1unpck_lo = _mm_unpacklo_epi8(a1, zero);
  auto a2unpck_lo = _mm_unpacklo_epi8(a2, zero);
  auto a3unpck_lo = _mm_unpacklo_epi8(a3, zero);
  auto a4unpck_lo = _mm_unpacklo_epi8(a4, zero);
  auto a5unpck_lo = _mm_unpacklo_epi8(a5, zero);
  auto a6unpck_lo = _mm_unpacklo_epi8(a6, zero);
  auto a7unpck_lo = _mm_unpacklo_epi8(a7, zero);
  auto a8unpck_lo = _mm_unpacklo_epi8(a8, zero);
  auto cunpck_lo = _mm_unpacklo_epi8(c, zero);

  auto sum_t1 = _mm_adds_epu16(a1unpck_lo, a2unpck_lo);
  sum_t1 = _mm_adds_epu16(sum_t1, a3unpck_lo);
  sum_t1 = _mm_adds_epu16(sum_t1, a4unpck_lo);

  auto sum_t2 = _mm_adds_epu16(a5unpck_lo, a6unpck_lo);
  sum_t2 = _mm_adds_epu16(sum_t2, a7unpck_lo);
  sum_t2 = _mm_adds_epu16(sum_t2, a8unpck_lo);

  auto sum = _mm_adds_epu16(sum_t1, sum_t2);
  sum = _mm_adds_epu16(sum, cunpck_lo);
  sum = _mm_adds_epu16(sum, bias);

  auto result_lo = _mm_mulhi_epu16(sum, onenineth);


  auto a1unpck_hi = _mm_unpackhi_epi8(a1, zero);
  auto a2unpck_hi = _mm_unpackhi_epi8(a2, zero);
  auto a3unpck_hi = _mm_unpackhi_epi8(a3, zero);
  auto a4unpck_hi = _mm_unpackhi_epi8(a4, zero);
  auto a5unpck_hi = _mm_unpackhi_epi8(a5, zero);
  auto a6unpck_hi = _mm_unpackhi_epi8(a6, zero);
  auto a7unpck_hi = _mm_unpackhi_epi8(a7, zero);
  auto a8unpck_hi = _mm_unpackhi_epi8(a8, zero);
  auto cunpck_hi = _mm_unpackhi_epi8(c, zero);

  sum_t1 = _mm_adds_epu16(a1unpck_hi, a2unpck_hi);
  sum_t1 = _mm_adds_epu16(sum_t1, a3unpck_hi);
  sum_t1 = _mm_adds_epu16(sum_t1, a4unpck_hi);

  sum_t2 = _mm_adds_epu16(a5unpck_hi, a6unpck_hi);
  sum_t2 = _mm_adds_epu16(sum_t2, a7unpck_hi);
  sum_t2 = _mm_adds_epu16(sum_t2, a8unpck_hi);

  sum = _mm_adds_epu16(sum_t1, sum_t2);
  sum = _mm_adds_epu16(sum, cunpck_hi);
  sum = _mm_adds_epu16(sum, bias);

  auto result_hi = _mm_mulhi_epu16(sum, onenineth);

  return _mm_packus_epi16(result_lo, result_hi);
}


template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode20_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  //    int sum = a1 + a2 + a3 + a4 + c + a5 + a6 + a7 + a8;
  //    int val = (sum + 4) / 9;

  // trick, but there is no _mm_mulhi_epi32
  // x / 9 = x * 1/9 = [x * ( (1/9)<<32 )] >> 32 = Hi32_part_of_64bit_result(x * ((1 << 32)/9))
  // instead: x less than 20 bits (9*65535), we have 15 bits to play, (1<<15)/9 < 4096 (12 bits)
  // ((1<<14)+4) / 9) = 0x71C (1820)
  // ((1<<15)+4) / 9) = 0xE39 (3641)
  // worst case: 9*FFFF * 71C = 3FFBC004
  // worst case: 9*FFFF * E39 = 8000B8E3 ( 8000B8E3 >> 15 = 10001, packus rounding to FFFF)
  // Try with
  // ((1<<15) / 9  + 4) = 0xE3C (3644)
  constexpr Byte FACTOR = 15;
  auto zero = _mm_setzero_si128();
  auto onenineth = _mm_set1_epi32(((1u << FACTOR) + 4) / 9);
  auto bias = _mm_set1_epi32(4);

  auto a1unpck_lo = _mm_unpacklo_epi16(a1, zero);
  auto a2unpck_lo = _mm_unpacklo_epi16(a2, zero);
  auto a3unpck_lo = _mm_unpacklo_epi16(a3, zero);
  auto a4unpck_lo = _mm_unpacklo_epi16(a4, zero);
  auto a5unpck_lo = _mm_unpacklo_epi16(a5, zero);
  auto a6unpck_lo = _mm_unpacklo_epi16(a6, zero);
  auto a7unpck_lo = _mm_unpacklo_epi16(a7, zero);
  auto a8unpck_lo = _mm_unpacklo_epi16(a8, zero);
  auto cunpck_lo  = _mm_unpacklo_epi16(c, zero);

  // lower 4x uint16_t -> 128 bit 4x uint32_t
  auto sum_t1 = _mm_add_epi32(a1unpck_lo, a2unpck_lo);
  sum_t1 = _mm_add_epi32(sum_t1, a3unpck_lo);
  sum_t1 = _mm_add_epi32(sum_t1, a4unpck_lo);
  
  auto sum_t2 = _mm_add_epi32(a5unpck_lo, a6unpck_lo);
  sum_t2 = _mm_add_epi32(sum_t2, a7unpck_lo);
  sum_t2 = _mm_add_epi32(sum_t2, a8unpck_lo);

  auto sum = _mm_add_epi32(sum_t1, sum_t2);
  sum = _mm_add_epi32(sum, cunpck_lo);
  sum = _mm_add_epi32(sum, bias);

  auto result_lo = _mm_srli_epi32(_mm_mullo_epi32(sum, onenineth),FACTOR);
  // we have sum of lower 4 pixels

  auto a1unpck_hi = _mm_unpackhi_epi16(a1, zero);
  auto a2unpck_hi = _mm_unpackhi_epi16(a2, zero);
  auto a3unpck_hi = _mm_unpackhi_epi16(a3, zero);
  auto a4unpck_hi = _mm_unpackhi_epi16(a4, zero);
  auto a5unpck_hi = _mm_unpackhi_epi16(a5, zero);
  auto a6unpck_hi = _mm_unpackhi_epi16(a6, zero);
  auto a7unpck_hi = _mm_unpackhi_epi16(a7, zero);
  auto a8unpck_hi = _mm_unpackhi_epi16(a8, zero);
  auto cunpck_hi  = _mm_unpackhi_epi16(c, zero);

  sum_t1 = _mm_add_epi32(a1unpck_hi, a2unpck_hi);
  sum_t1 = _mm_add_epi32(sum_t1, a3unpck_hi);
  sum_t1 = _mm_add_epi32(sum_t1, a4unpck_hi);
  
  sum_t2 = _mm_add_epi32(a5unpck_hi, a6unpck_hi);
  sum_t2 = _mm_add_epi32(sum_t2, a7unpck_hi);
  sum_t2 = _mm_add_epi32(sum_t2, a8unpck_hi);

  sum = _mm_add_epi32(sum_t1, sum_t2);
  sum = _mm_add_epi32(sum, cunpck_hi);
  sum = _mm_add_epi32(sum, bias);

  auto result_hi = _mm_srli_epi32(_mm_mullo_epi32(sum, onenineth),FACTOR);

  return _mm_packus_epi32(result_lo, result_hi);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode20_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto onenineth = _mm_set1_ps(1/9.0f);
  // float val = (a1 + a2 + a3 + a4 + c + a5 + a6 + a7 + a8) / 9.0f;

  auto a12 = _mm_add_ps(a1, a2);
  auto a34 = _mm_add_ps(a3, a4);
  auto a1234 = _mm_add_ps(a12, a34);
  auto a56 = _mm_add_ps(a5, a6);
  auto a78 = _mm_add_ps(a7, a8);
  auto a5678 = _mm_add_ps(a56, a78);
  auto a12345678 = _mm_add_ps(a1234, a5678);
  auto val = _mm_add_ps(a12345678, c);
  return _mm_castps_si128(_mm_mul_ps(val, onenineth));
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode21_sse2(const Byte* pSrc, int srcPitch) {
 
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto l1a = not_rounded_average(a1, a8);
    auto l2a = not_rounded_average(a2, a7);
    auto l3a = not_rounded_average(a3, a6);
    auto l4a = not_rounded_average(a4, a5);

    auto l1b = _mm_avg_epu8(a1, a8);
    auto l2b = _mm_avg_epu8(a2, a7);
    auto l3b = _mm_avg_epu8(a3, a6);
    auto l4b = _mm_avg_epu8(a4, a5);

    auto ma = _mm_max_epu8(l1b, l2b);
    ma = _mm_max_epu8(ma, l3b);
    ma = _mm_max_epu8(ma, l4b);

    auto mi = _mm_min_epu8(l1a, l2a);
    mi = _mm_min_epu8(mi, l3a);
    mi = _mm_min_epu8(mi, l4a);

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode21_sse(const Byte* pSrc, int srcPitch) {

  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto l1a = not_rounded_average(a1, a8);
  auto l2a = not_rounded_average(a2, a7);
  auto l3a = not_rounded_average(a3, a6);
  auto l4a = not_rounded_average(a4, a5);

  auto l1b = _mm_avg_epu8(a1, a8);
  auto l2b = _mm_avg_epu8(a2, a7);
  auto l3b = _mm_avg_epu8(a3, a6);
  auto l4b = _mm_avg_epu8(a4, a5);

  auto ma = _mm_max_epu8(l1b, l2b);
  ma = _mm_max_epu8(ma, l3b);
  ma = _mm_max_epu8(ma, l4b);

  auto mi = _mm_min_epu8(l1a, l2a);
  mi = _mm_min_epu8(mi, l3a);
  mi = _mm_min_epu8(mi, l4a);

  return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode21_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto l1a = not_rounded_average_16(a1, a8);
  auto l2a = not_rounded_average_16(a2, a7);
  auto l3a = not_rounded_average_16(a3, a6);
  auto l4a = not_rounded_average_16(a4, a5);

  auto l1b = _mm_avg_epu16(a1, a8);
  auto l2b = _mm_avg_epu16(a2, a7);
  auto l3b = _mm_avg_epu16(a3, a6);
  auto l4b = _mm_avg_epu16(a4, a5);

  auto ma = _mm_max_epu16(l1b, l2b);
  ma = _mm_max_epu16(ma, l3b);
  ma = _mm_max_epu16(ma, l4b);

  auto mi = _mm_min_epu16(l1a, l2a);
  mi = _mm_min_epu16(mi, l3a);
  mi = _mm_min_epu16(mi, l4a);

  return simd_clip_16(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode22_sse_32(const Byte* pSrc, int srcPitch);

  // float: no integer tricks, same as 22
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode21_sse_32(const Byte* pSrc, int srcPitch) {
  return rg_mode22_sse_32<aligned>(pSrc, srcPitch);
}

//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode22_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto l1 = _mm_avg_epu8(a1, a8);
    auto l2 = _mm_avg_epu8(a2, a7);

    auto ma = _mm_max_epu8(l1, l2);
    auto mi = _mm_min_epu8(l1, l2);

    auto l3 = _mm_avg_epu8(a3, a6);
    ma = _mm_max_epu8(ma, l3);
    mi = _mm_min_epu8(mi, l3);

    auto l4 = _mm_avg_epu8(a4, a5);
    ma = _mm_max_epu8(ma, l4);
    mi = _mm_min_epu8(mi, l4);

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode22_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto l1 = _mm_avg_epu8(a1, a8);
  auto l2 = _mm_avg_epu8(a2, a7);

  auto ma = _mm_max_epu8(l1, l2);
  auto mi = _mm_min_epu8(l1, l2);

  auto l3 = _mm_avg_epu8(a3, a6);
  ma = _mm_max_epu8(ma, l3);
  mi = _mm_min_epu8(mi, l3);

  auto l4 = _mm_avg_epu8(a4, a5);
  ma = _mm_max_epu8(ma, l4);
  mi = _mm_min_epu8(mi, l4);

  return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode22_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto l1 = _mm_avg_epu16(a1, a8);
  auto l2 = _mm_avg_epu16(a2, a7);

  auto ma = _mm_max_epu16(l1, l2);
  auto mi = _mm_min_epu16(l1, l2);

  auto l3 = _mm_avg_epu16(a3, a6);
  ma = _mm_max_epu16(ma, l3);
  mi = _mm_min_epu16(mi, l3);

  auto l4 = _mm_avg_epu16(a4, a5);
  ma = _mm_max_epu16(ma, l4);
  mi = _mm_min_epu16(mi, l4);

  return simd_clip_16(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode22_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto l1 = _mm_avg_ps(a1, a8);
  auto l2 = _mm_avg_ps(a2, a7);

  auto ma = _mm_max_ps(l1, l2);
  auto mi = _mm_min_ps(l1, l2);

  auto l3 = _mm_avg_ps(a3, a6);
  ma = _mm_max_ps(ma, l3);
  mi = _mm_min_ps(mi, l3);

  auto l4 = _mm_avg_ps(a4, a5);
  ma = _mm_max_ps(ma, l4);
  mi = _mm_min_ps(mi, l4);

  return _mm_castps_si128(simd_clip_32(c, mi, ma));
}


//-------------------

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode23_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto linediff1 = _mm_subs_epu8(mal1, mil1);
    auto linediff2 = _mm_subs_epu8(mal2, mil2);
    auto linediff3 = _mm_subs_epu8(mal3, mil3);
    auto linediff4 = _mm_subs_epu8(mal4, mil4);

    auto u1 = _mm_min_epu8(_mm_subs_epu8(c, mal1), linediff1);
    auto u2 = _mm_min_epu8(_mm_subs_epu8(c, mal2), linediff2);
    auto u3 = _mm_min_epu8(_mm_subs_epu8(c, mal3), linediff3);
    auto u4 = _mm_min_epu8(_mm_subs_epu8(c, mal4), linediff4);

    auto u = _mm_max_epu8(u1, u2);
    u = _mm_max_epu8(u, u3);
    u = _mm_max_epu8(u, u4);

    auto d1 = _mm_min_epu8(_mm_subs_epu8(mil1, c), linediff1);
    auto d2 = _mm_min_epu8(_mm_subs_epu8(mil2, c), linediff2);
    auto d3 = _mm_min_epu8(_mm_subs_epu8(mil3, c), linediff3);
    auto d4 = _mm_min_epu8(_mm_subs_epu8(mil4, c), linediff4);

    auto d = _mm_max_epu8(d1, d2);
    d = _mm_max_epu8(d, d3);
    d = _mm_max_epu8(d, d4);

    return _mm_adds_epu8(_mm_subs_epu8(c, u), d);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode23_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto linediff1 = _mm_subs_epu8(mal1, mil1);
  auto linediff2 = _mm_subs_epu8(mal2, mil2);
  auto linediff3 = _mm_subs_epu8(mal3, mil3);
  auto linediff4 = _mm_subs_epu8(mal4, mil4);

  auto u1 = _mm_min_epu8(_mm_subs_epu8(c, mal1), linediff1);
  auto u2 = _mm_min_epu8(_mm_subs_epu8(c, mal2), linediff2);
  auto u3 = _mm_min_epu8(_mm_subs_epu8(c, mal3), linediff3);
  auto u4 = _mm_min_epu8(_mm_subs_epu8(c, mal4), linediff4);

  auto u = _mm_max_epu8(u1, u2);
  u = _mm_max_epu8(u, u3);
  u = _mm_max_epu8(u, u4);

  auto d1 = _mm_min_epu8(_mm_subs_epu8(mil1, c), linediff1);
  auto d2 = _mm_min_epu8(_mm_subs_epu8(mil2, c), linediff2);
  auto d3 = _mm_min_epu8(_mm_subs_epu8(mil3, c), linediff3);
  auto d4 = _mm_min_epu8(_mm_subs_epu8(mil4, c), linediff4);

  auto d = _mm_max_epu8(d1, d2);
  d = _mm_max_epu8(d, d3);
  d = _mm_max_epu8(d, d4);

  return _mm_adds_epu8(_mm_subs_epu8(c, u), d);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode23_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto linediff1 = _mm_subs_epu16(mal1, mil1);
  auto linediff2 = _mm_subs_epu16(mal2, mil2);
  auto linediff3 = _mm_subs_epu16(mal3, mil3);
  auto linediff4 = _mm_subs_epu16(mal4, mil4);

  auto u1 = _mm_min_epu16(_mm_subs_epu16(c, mal1), linediff1);
  auto u2 = _mm_min_epu16(_mm_subs_epu16(c, mal2), linediff2);
  auto u3 = _mm_min_epu16(_mm_subs_epu16(c, mal3), linediff3);
  auto u4 = _mm_min_epu16(_mm_subs_epu16(c, mal4), linediff4);

  auto u = _mm_max_epu16(u1, u2);
  u = _mm_max_epu16(u, u3);
  u = _mm_max_epu16(u, u4);

  auto d1 = _mm_min_epu16(_mm_subs_epu16(mil1, c), linediff1);
  auto d2 = _mm_min_epu16(_mm_subs_epu16(mil2, c), linediff2);
  auto d3 = _mm_min_epu16(_mm_subs_epu16(mil3, c), linediff3);
  auto d4 = _mm_min_epu16(_mm_subs_epu16(mil4, c), linediff4);

  auto d = _mm_max_epu16(d1, d2);
  d = _mm_max_epu16(d, d3);
  d = _mm_max_epu16(d, d4);

  return _mm_adds_epu16(_mm_subs_epu16(c, u), d);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode23_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto linediff1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto linediff2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto linediff3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto linediff4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto u1 = _mm_min_ps(_mm_subs_ps_for_diff(c, mal1), linediff1);
  auto u2 = _mm_min_ps(_mm_subs_ps_for_diff(c, mal2), linediff2);
  auto u3 = _mm_min_ps(_mm_subs_ps_for_diff(c, mal3), linediff3);
  auto u4 = _mm_min_ps(_mm_subs_ps_for_diff(c, mal4), linediff4);

  auto u = _mm_max_ps(u1, u2);
  u = _mm_max_ps(u, u3);
  u = _mm_max_ps(u, u4);

  auto d1 = _mm_min_ps(_mm_subs_ps_for_diff(mil1, c), linediff1);
  auto d2 = _mm_min_ps(_mm_subs_ps_for_diff(mil2, c), linediff2);
  auto d3 = _mm_min_ps(_mm_subs_ps_for_diff(mil3, c), linediff3);
  auto d4 = _mm_min_ps(_mm_subs_ps_for_diff(mil4, c), linediff4);

  auto d = _mm_max_ps(d1, d2);
  d = _mm_max_ps(d, d3);
  d = _mm_max_ps(d, d4);

  return _mm_castps_si128(_mm_adds_ps<chroma>(_mm_subs_ps<chroma>(c, u), d));
}


//-------------------

//optimized, todo: decide how to name the function and extract this stuff. Order is important.
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode24_sse2(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal  = _mm_max_epu8(a1, a8);
    auto mil  = _mm_min_epu8(a1, a8);
    auto diff = _mm_subs_epu8(mal, mil);
    auto temp = _mm_subs_epu8(c, mal);
    auto u1   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
    temp      = _mm_subs_epu8(mil, c);
    auto d1   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

    mal       = _mm_max_epu8(a2, a7);
    mil       = _mm_min_epu8(a2, a7);
    diff      = _mm_subs_epu8(mal, mil);
    temp      = _mm_subs_epu8(c, mal);
    auto u2   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
    temp      = _mm_subs_epu8(mil, c);
    auto d2   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

    auto d = _mm_max_epu8(d1, d2);
    auto u = _mm_max_epu8(u1, u2);

    mal       = _mm_max_epu8(a3, a6);
    mil       = _mm_min_epu8(a3, a6);
    diff      = _mm_subs_epu8(mal, mil);
    temp      = _mm_subs_epu8(c, mal);
    auto u3   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
    temp      = _mm_subs_epu8(mil, c);
    auto d3   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

    d = _mm_max_epu8(d, d3);
    u = _mm_max_epu8(u, u3);

    mal       = _mm_max_epu8(a4, a5);
    mil       = _mm_min_epu8(a4, a5);
    diff      = _mm_subs_epu8(mal, mil);
    temp      = _mm_subs_epu8(c, mal);
    auto u4   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
    temp      = _mm_subs_epu8(mil, c);
    auto d4   = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

    d = _mm_max_epu8(d, d4);
    u = _mm_max_epu8(u, u4);

    return _mm_adds_epu8(_mm_subs_epu8(c, u), d);
}


template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode24_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal = _mm_max_epu8(a1, a8);
  auto mil = _mm_min_epu8(a1, a8);
  auto diff = _mm_subs_epu8(mal, mil);
  auto temp = _mm_subs_epu8(c, mal);
  auto u1 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
  temp = _mm_subs_epu8(mil, c);
  auto d1 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

  mal = _mm_max_epu8(a2, a7);
  mil = _mm_min_epu8(a2, a7);
  diff = _mm_subs_epu8(mal, mil);
  temp = _mm_subs_epu8(c, mal);
  auto u2 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
  temp = _mm_subs_epu8(mil, c);
  auto d2 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

  auto d = _mm_max_epu8(d1, d2);
  auto u = _mm_max_epu8(u1, u2);

  mal = _mm_max_epu8(a3, a6);
  mil = _mm_min_epu8(a3, a6);
  diff = _mm_subs_epu8(mal, mil);
  temp = _mm_subs_epu8(c, mal);
  auto u3 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
  temp = _mm_subs_epu8(mil, c);
  auto d3 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

  d = _mm_max_epu8(d, d3);
  u = _mm_max_epu8(u, u3);

  mal = _mm_max_epu8(a4, a5);
  mil = _mm_min_epu8(a4, a5);
  diff = _mm_subs_epu8(mal, mil);
  temp = _mm_subs_epu8(c, mal);
  auto u4 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));
  temp = _mm_subs_epu8(mil, c);
  auto d4 = _mm_min_epu8(temp, _mm_subs_epu8(diff, temp));

  d = _mm_max_epu8(d, d4);
  u = _mm_max_epu8(u, u4);

  return _mm_adds_epu8(_mm_subs_epu8(c, u), d);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode24_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal  = _mm_max_epu16(a1, a8);
  auto mil  = _mm_min_epu16(a1, a8);
  auto diff = _mm_subs_epu16(mal, mil);
  auto temp = _mm_subs_epu16(c, mal);
  auto u1   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));
  temp      = _mm_subs_epu16(mil, c);
  auto d1   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));

  mal       = _mm_max_epu16(a2, a7);
  mil       = _mm_min_epu16(a2, a7);
  diff      = _mm_subs_epu16(mal, mil);
  temp      = _mm_subs_epu16(c, mal);
  auto u2   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));
  temp      = _mm_subs_epu16(mil, c);
  auto d2   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));

  auto d = _mm_max_epu16(d1, d2);
  auto u = _mm_max_epu16(u1, u2);

  mal       = _mm_max_epu16(a3, a6);
  mil       = _mm_min_epu16(a3, a6);
  diff      = _mm_subs_epu16(mal, mil);
  temp      = _mm_subs_epu16(c, mal);
  auto u3   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));
  temp      = _mm_subs_epu16(mil, c);
  auto d3   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));

  d = _mm_max_epu16(d, d3);
  u = _mm_max_epu16(u, u3);

  mal       = _mm_max_epu16(a4, a5);
  mil       = _mm_min_epu16(a4, a5);
  diff      = _mm_subs_epu16(mal, mil);
  temp      = _mm_subs_epu16(c, mal);
  auto u4   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));
  temp      = _mm_subs_epu16(mil, c);
  auto d4   = _mm_min_epu16(temp, _mm_subs_epu16(diff, temp));

  d = _mm_max_epu16(d, d4);
  u = _mm_max_epu16(u, u4);

  return _mm_adds_epu16(_mm_subs_epu16(c, u), d);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode24_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal  = _mm_max_ps(a1, a8);
  auto mil  = _mm_min_ps(a1, a8);
  auto diff = _mm_subs_ps_for_diff(mal, mil);
  auto temp = _mm_subs_ps_for_diff(c, mal);
  auto u1   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));
  temp      = _mm_subs_ps_for_diff(mil, c);
  auto d1   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));

  mal       = _mm_max_ps(a2, a7);
  mil       = _mm_min_ps(a2, a7);
  diff      = _mm_subs_ps_for_diff(mal, mil);
  temp      = _mm_subs_ps_for_diff(c, mal);
  auto u2   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));
  temp      = _mm_subs_ps_for_diff(mil, c);
  auto d2   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));

  auto d = _mm_max_ps(d1, d2);
  auto u = _mm_max_ps(u1, u2);

  mal       = _mm_max_ps(a3, a6);
  mil       = _mm_min_ps(a3, a6);
  diff      = _mm_subs_ps_for_diff(mal, mil);
  temp      = _mm_subs_ps_for_diff(c, mal);
  auto u3   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));
  temp      = _mm_subs_ps_for_diff(mil, c);
  auto d3   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));

  d = _mm_max_ps(d, d3);
  u = _mm_max_ps(u, u3);

  mal       = _mm_max_ps(a4, a5);
  mil       = _mm_min_ps(a4, a5);
  diff      = _mm_subs_ps_for_diff(mal, mil);
  temp      = _mm_subs_ps_for_diff(c, mal);
  auto u4   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));
  temp      = _mm_subs_ps_for_diff(mil, c);
  auto d4   = _mm_min_ps(temp, _mm_subs_ps_for_diff(diff, temp));

  d = _mm_max_ps(d, d4);
  u = _mm_max_ps(u, u4);

  return _mm_castps_si128(_mm_adds_ps<chroma>(_mm_subs_ps<chroma>(c, u), d));
}

/*
Finally mode 25 is the minimal sharpening mode.
In this mode the neighbours n1,n2 are selected such that n1 <= c <= n2 and such that there
exists no further neighbour n with n1 < n < n2. If c is larger than all neighbours,
then n2 doesn't exist. In this case, we simply replace n2 by 255.
Similarily, if c is smaller than all neighbours, then n1 doesn't exist and n1 is replaced
by the value 0. Finally c is sharpened as described in the section Sharpening between two points.
*/
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode25_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  __m128i SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  __m128i SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results
  const __m128i zero = _mm_setzero_si128();

  neighbourdiff(SSE4, SSE5, c, a4, zero); // out out out in in in
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff(SSE6, SSE7, c, a5, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a1, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a2, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a3, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a6, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a7, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a8, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  auto result = sharpen(c, SSE4, SSE5);
  return result;
}

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode25_sse2(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  __m128i SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  __m128i SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results
  const __m128i zero = _mm_setzero_si128();

  neighbourdiff(SSE4, SSE5, c, a4, zero); // out out out in in in
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff(SSE6, SSE7, c, a5, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a1, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a2, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a3, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a6, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a7, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  neighbourdiff(SSE6, SSE7, c, a8, zero);
  SSE4 = _mm_min_epu8(SSE4, SSE6);
  SSE5 = _mm_min_epu8(SSE5, SSE7);

  auto result = sharpen(c, SSE4, SSE5);
  return result;
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode25_sse_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  __m128i SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  __m128i SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results
  const __m128i zero = _mm_setzero_si128();

  neighbourdiff_16(SSE4, SSE5, c, a4, zero); // out out out in in in
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff_16(SSE6, SSE7, c, a5, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a1, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a2, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a3, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a6, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a7, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  neighbourdiff_16(SSE6, SSE7, c, a8, zero);
  SSE4 = _mm_min_epu16(SSE4, SSE6);
  SSE5 = _mm_min_epu16(SSE5, SSE7);

  auto result = sharpen_16<bits_per_pixel>(c, SSE4, SSE5);

  return result;
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode25_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  __m128 SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  __m128 SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results
  const __m128 zero = _mm_setzero_ps();

  neighbourdiff_32(SSE4, SSE5, c, a4, zero); // out out out in in in
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff_32(SSE6, SSE7, c, a5, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a1, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a2, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a3, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a6, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a7, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  neighbourdiff_32(SSE6, SSE7, c, a8, zero);
  SSE4 = _mm_min_ps(SSE4, SSE6);
  SSE5 = _mm_min_ps(SSE5, SSE7);

  auto result = sharpen_32<chroma>(c, SSE4, SSE5);

  return _mm_castps_si128(result);
}



#if 0
// mode25
void	nondestructivesharpen(const Byte* pSrc, int srcPitch, BYTE* dp, int dpitch, const BYTE* _sp, int spitch, int hblocks, int remainder, int incpitch, int height)
{
  int eax, ebx, edx;
  const BYTE* esi;
  BYTE* edi;
  // integrated into later eax = hblocks; // __asm	mov			eax,				hblocks
  ebx = spitch; // __asm	mov			ebx, spitch
  edx = remainder; //  __asm	mov			edx, remainder
  auto sse0 = _mm_setzero_si128(); // __asm	pxor		SSE0, SSE0
  // integrated into later eax = eax * 2; //  __asm	add			eax, eax
  // integrated esi = _sp; //  __asm	mov			esi, _sp
  eax = hblocks * 16 + edx + 15; // __asm	lea			eax, [eax * 8 + edx + SSE_INCREMENT + 1]
  esi = _sp - ebx; // esi: PREV //  __asm	sub			esi, ebx
  dpitch -= eax; // __asm	sub			dpitch,				eax
  // integrated later eax = -eax; //  __asm	neg			eax
  edi = dp; // edi: DEST PTR //  __asm	mov			edi, dp
  eax = -eax + spitch + 1; // EAX //  __asm	lea			eax, [ebx + eax + 1]

  // __asm	align		16
  // __asm	column_loop:
  /*
  __asm	SSE3_MOVE	SSE1,				[esi + ebx + 1]
  __asm	SSE3_MOVE	SSE3,				[esi + ebx]
      neighbourdiff_w(SSE4, SSE5, SSE2, SSE1, [edi], SSE3, SSE0, movd)

  __asm	SSE3_MOVE	SSE3,				[esi + ebx + 2]
      neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi]
      neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi + 1]
      neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi + 2]
      neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi + 2*ebx]
      neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 1]
      neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7

  __asm	SSE3_MOVE	SSE3,				[esi + 2*ebx + 2]
      neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
  __asm	pminub		SSE4,				SSE6
  __asm	pminub		SSE5,				SSE7
      sharpen(SSE1, SSE4, SSE5, SSE6, SSE7)
  __asm	SSE_MOVE	[edi + 1],			SSE1
  */
  /*
  original:
  out    out     out    in        in         in
  #define neighbourdiff(minus, plus, center1_as_centerNext, center2, neighbour, nullreg)	\

  // center1 and center2 are changing during consecutive calls
  // helper for mode 25_mode24
  static RG_FORCEINLINE void neighbourdiff(__m128i & minus, __m128i & plus, __m128i & center_next, __m128i center2, __m128i neighbour, const __m128i & zero) {
  */
  LOAD_SQUARE_SSE_UA(pSrc, srcPitch /*, aligned*/, true);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  __m128i SSE1, SSE2, SSE3;
  __m128i SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  __m128i SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results
  __m128i SSE0 = _mm_setzero_si128();
  /*
    // now the pixels in the middle
  __asm	add			esi,				SSE_INCREMENT
  __asm	add			edi,				SSE_INCREMENT + 1
  __asm	mov			ecx,				hblocks
  __asm	align		16
  __asm	middle_loop:
  */
  SSE1 = c; // center middle __asm	SSE3_MOVE	SSE1, [esi + ebx + 1] // +ebx: NEXT
  SSE3 = a4; // center left__asm	SSE3_MOVE	SSE3, [esi + ebx]

  neighbourdiff(SSE4, SSE5, SSE2, SSE1, SSE3, SSE0); // out out out in in in
  // first result fill into collectors SSE4 and SSE5, no comparison
  SSE3 = a5; // center right _asm	SSE3_MOVE	SSE3,				[esi + ebx + 2]
  // SSE1 and SSE2 a changing SSE2,SSE1, then SSE1, SSE2, then SSE2,SSE1, internal asm optimization
  neighbourdiff(SSE4, SSE5, SSE1, SSE2, SSE3, SSE0); //  neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a1; // top left  __asm	SSE3_MOVE	SSE3, [esi]
  neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0); //    neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a2; // top center  __asm	SSE3_MOVE	SSE3, [esi + 1]
  neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0);
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a3; // top right  __asm	SSE3_MOVE	SSE3, [esi + 2]
  neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0);
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a6; // bottom left  __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx]
  neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0);
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a7; // bottom center __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx + 1]
  neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0);
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  SSE3 = a8; // bottom right  __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx + 2]
  neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0);
  SSE4 = _mm_min_epu8(SSE4, SSE6); // __asm	pminub		SSE4, SSE6
  SSE5 = _mm_min_epu8(SSE5, SSE7); //__asm	pminub		SSE5,				SSE7

  //__asm	add			esi,				SSE_INCREMENT
  /*
  in / out  in     in    tmp  tmp
  #define	sharpen(center, minus, plus, reg1, reg2)\
  //omit tmp, compiler solves it
  static RG_FORCEINLINE __m128i sharpen(const __m128i& center, const __m128i& minus, const __m128i& plus) {
  */
  sharpen(SSE1, SSE4, SSE5); // in/out, in in tmp tmp
  __asm	SSE_MOVE[edi], SSE1
  __asm	add			edi, SSE_INCREMENT
  __asm	dec			ecx
  __asm	jnz			middle_loop

  // the last pixels
  __asm	add			esi, edx
  __asm	add			edi, edx
  __asm	SSE3_MOVE	SSE1, [esi + ebx + 1]
    __asm	SSE3_MOVE	SSE3, [esi + ebx]
    neighbourdiff(SSE4, SSE5, SSE2, SSE1, SSE3, SSE0)

    __asm	SSE3_MOVE	SSE3, [esi + ebx + 2]
    neighbourdiff_w(SSE6, SSE7, SSE1, SSE2, [edi + 1], SSE3, SSE0, SSE_MOVE)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi]
    neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi + 1]
    neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi + 2]
    neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx]
    neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx + 1]
    neighbourdiff(SSE6, SSE7, SSE2, SSE1, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7

  __asm	SSE3_MOVE	SSE3, [esi + 2 * ebx + 2]
    neighbourdiff(SSE6, SSE7, SSE1, SSE2, SSE3, SSE0)
    __asm	pminub		SSE4, SSE6
  __asm	pminub		SSE5, SSE7
  __asm	add			esi, eax
  sharpen(SSE1, SSE4, SSE5, SSE6, SSE7)
    __asm	SSE_MOVE[edi], SSE1
  __asm	add			edi, dpitch
  __asm	dec			height
  __asm	jnz			column_loop
}
*/
#endif // 0

// Mode26_SmartRGC.cpp
// 26 = medianblur.Based off mode 17, but preserves corners, but not thin lines.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode26_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  // going clockwise
  auto mi12 = _mm_min_epu8(a1, a2);
  auto ma12 = _mm_max_epu8(a1, a2);
  
  auto mi23 = _mm_min_epu8(a2, a3);
  auto ma23 = _mm_max_epu8(a2, a3);
  auto lower = _mm_max_epu8(mi12, mi23);
  auto upper = _mm_min_epu8(ma12, ma23);

  auto mi35 = _mm_min_epu8(a3, a5);
  auto ma35 = _mm_max_epu8(a3, a5);
  lower = _mm_max_epu8(lower, mi35);
  upper = _mm_min_epu8(upper, ma35);

  auto mi58 = _mm_min_epu8(a5, a8);
  auto ma58 = _mm_max_epu8(a5, a8);
  lower = _mm_max_epu8(lower, mi58);
  upper = _mm_min_epu8(upper, ma58);

  auto mi78 = _mm_min_epu8(a7, a8);
  auto ma78 = _mm_max_epu8(a7, a8);
  lower = _mm_max_epu8(lower, mi78);
  upper = _mm_min_epu8(upper, ma78);

  auto mi67 = _mm_min_epu8(a6, a7);
  auto ma67 = _mm_max_epu8(a6, a7);
  lower = _mm_max_epu8(lower, mi67);
  upper = _mm_min_epu8(upper, ma67);

  auto mi46 = _mm_min_epu8(a4, a6);
  auto ma46 = _mm_max_epu8(a4, a6);
  lower = _mm_max_epu8(lower, mi46);
  upper = _mm_min_epu8(upper, ma46);

  auto mi14 = _mm_min_epu8(a1, a4);
  auto ma14 = _mm_max_epu8(a1, a4);
  lower = _mm_max_epu8(lower, mi14);
  upper = _mm_min_epu8(upper, ma14);

  auto real_lower = _mm_min_epu8(lower, upper);
  auto real_upper = _mm_max_epu8(lower, upper);

  return simd_clip(c, real_lower, real_upper);
}

#if mode26
__asm	align		16
__asm	middle_loop:
sse7 = a1; // __asm	SSE3_MOVE	SSE7, [esi]
sse6 = s2; // __asm	SSE3_MOVE	SSE6, [esi + 1]
sse0 = sse7; // __asm	SSE_RMOVE	SSE0, SSE7
sse1 = sse6; // __asm	SSE_RMOVE	SSE1, SSE6
sse5 = a3; // __asm	SSE3_MOVE	SSE5, [esi + 2]
sse0 = _mm_min_epu8(sse0, sse6); / __asm	pminub		SSE0, SSE6
sse2 = sse5; // __asm	SSE_RMOVE	SSE2, SSE5
sse1 = _mm_max_epu8(sse1, sse7); // __asm	pmaxub		SSE1, SSE7
sse2 = _mm_min_epu8(sse2, sse6); //  __asm	pminub		SSE2, SSE6
sse4 = a5; // __asm	SSE3_MOVE	SSE4, [esi + ebx + 2]
sse6 = _mm_max_epu8(sse6, sse5); // __asm	pmaxub		SSE6, SSE5
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse4; // __asm	SSE_RMOVE	SSE3, SSE4
sse1 = _mm_min_epu8(sse1, sse6); // __asm	pminub		SSE1, SSE6
sse3 = _mm_min_epu8(sse3, sse5); // __asm	pminub		SSE3, SSE5
sse6 = a8; // __asm	SSE3_MOVE	SSE6, [esi + 2 * ebx + 2]
sse5 = _mm_max_epu8(sse5, sse4); // __asm	pmaxub		SSE5, SSE4
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse2 = sse6; // __asm	SSE_RMOVE	SSE2, SSE6
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5
sse2 = _mm_min_epu8(sse2, sse4); // __asm	pminub		SSE2, SSE4
sse5 = a7; // __asm	SSE3_MOVE	SSE5, [esi + 2 * ebx + 1]
sse4 = _mm_max_epu8(sse4, sse6); // __asm	pmaxub		SSE4, SSE6
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse5; // __asm	SSE_RMOVE	SSE3, SSE5
sse1 = _mm_min_epu8(sse1, sse4); // __asm	pminub		SSE1, SSE4
sse3 = _mm_min_epu8(sse3, sse6); //  __asm	pminub		SSE3, SSE6
sse4 = a6; // __asm	SSE3_MOVE	SSE4, [esi + 2 * ebx]
sse6 = _mm_max_epu8(sse6, sse5); // __asm	pmaxub		SSE6, SSE5
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse2 = sse4; // __asm	SSE_RMOVE	SSE2, SSE4
sse1 = _mm_min_epu8(sse1, sse6); // __asm	pminub		SSE1, SSE6
sse2 = _mm_min_epu8(sse2, sse5); // __asm	pminub		SSE2, SSE5
sse6 = a4; // __asm	SSE3_MOVE	SSE6, [esi + ebx]
sse5 = _mm_max_epu8(sse5, sse4); // __asm	pmaxub		SSE5, SSE4
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse6; // __asm	SSE_RMOVE	SSE3, SSE6
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5
sse3 = _mm_min_epu8(sse3, sse4); // __asm	pminub		SSE3, SSE4
sse2 = sse7; // __asm	SSE_RMOVE	SSE2, SSE7
sse4 = _mm_max_epu8(sse4, sse6); // __asm	pmaxub		SSE4, SSE6
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse1 = _mm_min_epu8(sse1, sse4); // __asm	pminub		SSE1, SSE4
sse2 = _mm_min_epu8(sse2, sse6); // __asm	pminub		SSE2, SSE6
sse7 = _mm_max_epu8(sse7, sse6); // __asm	pmaxub		SSE7, SSE6
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse1 = _mm_min_epu8(sse1, sse7); // __asm	pminub		SSE1, SSE7

sse2 = sse0; //   __asm	SSE_RMOVE	SSE2, SSE0
//#if		(ISSE > 1) || defined(SHLUR)
//#ifdef	MODIFYPLUGIN
//__asm	SSE3_MOVE	SSE4, [edi] (repair)
//#else
sse4 = c; // __asm	SSE3_MOVE	SSE4, [esi + ebx + 1]
//#endif
//#endif
//#if		MODIFYPLUGIN > 0 (repair)
//sse5 = c; // __asm	SSE3_MOVE	SSE5, [esi + ebx + 1]
//#endif

sse0 = _mm_min_epu8(sse0, sse1); // __asm	pminub		SSE0, SSE1
sse2 = _mm_max_epu8(sse2, sse1); //  __asm	pmaxub		SSE2, SSE1
// #if		MODIFYPLUGIN > 0 (repair)
// sse1 = _mm_min_epu8(sse0, sse5); //__asm	pminub		SSE0, SSE5
// sse2 = _mm_max_epu8(sse2, sse5); //__asm	pmaxub		SSE2, SSE5
//#endif

//#ifdef	SHLUR
//sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
//__asm	add			esi, SSE_INCREMENT
//__asm	SSE_MOVE[edi], SSE4
//#else
//#if	ISSE > 1
sse0 = _mm_max_epu8(sse0, sse4); // __asm	pmaxub		SSE0, SSE4
//#else
//#ifdef	MODIFYPLUGIN (repair)
// sse0 = _mm_max_epu8(sse0, reference); // __asm	pmaxub		SSE0, [edi]
// #else
// sse0 = _mm_max_epu8(sse0, c); // __asm	pmaxub		SSE0, [esi + ebx + 1]
//#endif
//#endif
sse0 = _mm_min_epu8(sse0, sse2); // __asm	pminub		SSE0, SSE2
auto result = sse0; //  __asm	SSE_MOVE[edi], SSE0
// #endif	// SHLUR
//#endif
#endif

// Mode27_SmartRGCL.cpp
// 26 = medianblur.Based off mode 17, but preserves corners, but not thin lines.
// 27 = medianblur.Same as mode 26 but preserves thin lines.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode27_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */

  auto mi18 = _mm_min_epu8(a1, a8);
  auto ma18 = _mm_max_epu8(a1, a8);

  auto mi12 = _mm_min_epu8(a1, a2);
  auto ma12 = _mm_max_epu8(a1, a2);

  auto lower = _mm_max_epu8(mi18, mi12);
  auto upper = _mm_min_epu8(ma18, ma12);

  auto mi78 = _mm_min_epu8(a7, a8);
  auto ma78 = _mm_max_epu8(a7, a8);
  lower = _mm_max_epu8(lower, mi78);
  upper = _mm_min_epu8(upper, ma78);

  auto mi27 = _mm_min_epu8(a2, a7);
  auto ma27 = _mm_max_epu8(a2, a7);
  lower = _mm_max_epu8(lower, mi27);
  upper = _mm_min_epu8(upper, ma27);

  auto mi23 = _mm_min_epu8(a2, a3);
  auto ma23 = _mm_max_epu8(a2, a3);
  lower = _mm_max_epu8(lower, mi23);
  upper = _mm_min_epu8(upper, ma23);

  auto mi67 = _mm_min_epu8(a6, a7);
  auto ma67 = _mm_max_epu8(a6, a7);
  lower = _mm_max_epu8(lower, mi67);
  upper = _mm_min_epu8(upper, ma67);

  auto mi36 = _mm_min_epu8(a3, a6);
  auto ma36 = _mm_max_epu8(a3, a6);
  lower = _mm_max_epu8(lower, mi36);
  upper = _mm_min_epu8(upper, ma36);

  auto mi35 = _mm_min_epu8(a3, a5);
  auto ma35 = _mm_max_epu8(a3, a5);
  lower = _mm_max_epu8(lower, mi35);
  upper = _mm_min_epu8(upper, ma35);

  auto mi46 = _mm_min_epu8(a4, a6);
  auto ma46 = _mm_max_epu8(a4, a6);
  lower = _mm_max_epu8(lower, mi46);
  upper = _mm_min_epu8(upper, ma46);

  auto mi45 = _mm_min_epu8(a4, a5);
  auto ma45 = _mm_max_epu8(a4, a5);
  lower = _mm_max_epu8(lower, mi45);
  upper = _mm_min_epu8(upper, ma45);

  auto mi58 = _mm_min_epu8(a5, a8);
  auto ma58 = _mm_max_epu8(a5, a8);
  lower = _mm_max_epu8(lower, mi58);
  upper = _mm_min_epu8(upper, ma58);
  
  auto mi14 = _mm_min_epu8(a1, a4);
  auto ma14 = _mm_max_epu8(a1, a4);
  lower = _mm_max_epu8(lower, mi14);
  upper = _mm_min_epu8(upper, ma14);

  auto real_upper = _mm_max_epu8(upper, lower);
  auto real_lower = _mm_min_epu8(upper, lower);

  return simd_clip(c, real_lower, real_upper);
}

#ifdef mode27
__asm	align		16
__asm	middle_loop:

SSE4 = a1; // __asm	SSE3_MOVE	SSE4, [esi]
SSE5 = a8; //  __asm	SSE3_MOVE	SSE5, [esi + 2 * ebx + 2]
SSE0 = SSE4; //  __asm	SSE_RMOVE	SSE0, SSE4
SSE1 = SSE5; //  __asm	SSE_RMOVE	SSE1, SSE5
SSE6 = a2; //  __asm	SSE3_MOVE	SSE6, [esi + 1]
SSE0 = _mm_min_epu8(SSE0, SSE5); //  __asm	pminub		SSE0, SSE5
SSE2 = SSE6; //  __asm	SSE_RMOVE	SSE2, SSE6
SSE1 = _mm_max_epu8(SSE1, SSE4); // __asm	pmaxub		SSE1, SSE4
SSE7 = a7; // __asm	SSE3_MOVE	SSE7, [esi + 2 * ebx + 1]
SSE2 = _mm_min_epu8(SSE2, SSE4); //  __asm	pminub		SSE2, SSE4
SSE3 = SSE7; // __asm	SSE_RMOVE	SSE3, SSE7
SSE4 = _mm_max_epu8(SSE4, SSE6); // __asm	pmaxub		SSE4, SSE6
SSE3 = _mm_min_epu8(SSE3, SSE5); // __asm	pminub		SSE3, SSE5
SSE0 = _mm_max_epu8(SSE0, SSE2); //  __asm	pmaxub		SSE0, SSE2
SSE5 = _mm_max_epu8(SSE5, SSE7); // __asm	pmaxub		SSE5, SSE7
SSE1 = _mm_min_epu8(SSE1, SSE4); //  __asm	pminub		SSE1, SSE4
SSE0 = _mm_max_epu8(SSE0, SSE3); // __asm	pmaxub		SSE0, SSE3
SSE1 = _mm_min_epu8(SSE1, SSE5); // __asm	pminub		SSE1, SSE5

SSE4 = a3; // __asm	SSE3_MOVE	SSE4, [esi + 2]
SSE2 = SSE6; // __asm	SSE_RMOVE	SSE2, SSE6
SSE3 = SSE7; // __asm	SSE_RMOVE	SSE3, SSE7
SSE5 = SSE4; // __asm	SSE_RMOVE	SSE5, SSE4
SSE2 = _mm_min_epu8(SSE2, SSE7); // __asm	pminub		SSE2, SSE7
SSE3 = _mm_max_epu8(SSE3, SSE6); // __asm	pmaxub		SSE3, SSE6
SSE0 = _mm_max_epu8(SSE0, SSE2); // __asm	pmaxub		SSE0, SSE2
SSE5 = _mm_min_epu8(SSE5, SSE6); //  __asm	pminub		SSE5, SSE6
SSE1 = _mm_min_epu8(SSE1, SSE3); // __asm	pminub		SSE1, SSE3
SSE6 = _mm_max_epu8(SSE6, SSE4); // __asm	pmaxub		SSE6, SSE4
SSE0 = _mm_max_epu8(SSE0, SSE5); //  __asm	pmaxub		SSE0, SSE5
SSE5 = a6; // __asm	SSE3_MOVE	SSE5, [esi + 2 * ebx]
SSE1 = _mm_min_epu8(SSE1, SSE6); //  __asm	pminub		SSE1, SSE6
SSE2 = SSE5; // __asm	SSE_RMOVE	SSE2, SSE5
SSE5 = _mm_min_epu8(SSE5, SSE7); // __asm	pminub		SSE5, SSE7
SSE7 = _mm_max_epu8(SSE7, SSE2); // __asm	pmaxub		SSE7, SSE2
SSE0 = _mm_max_epu8(SSE0, SSE5); // __asm	pmaxub		SSE0, SSE5
SSE1 = _mm_min_epu8(SSE1, SSE7); // __asm	pminub		SSE1, SSE7
SSE5 = SSE2; // __asm	SSE_RMOVE	SSE5, SSE2
SSE3 = SSE4; // __asm	SSE_RMOVE	SSE3, SSE4

SSE6 = a5; // __asm	SSE3_MOVE	SSE6, [esi + ebx + 2]
SSE2 = _mm_min_epu8(SSE2, SSE4); // __asm	pminub		SSE2, SSE4
SSE3 = _mm_max_epu8(SSE3, SSE5); // __asm	pmaxub		SSE3, SSE5
SSE7 = SSE6; // __asm	SSE_RMOVE	SSE7, SSE6
SSE0 = _mm_max_epu8(SSE0, SSE2); // __asm	pmaxub		SSE0, SSE2
SSE7 = _mm_min_epu8(SSE7, SSE4); // __asm	pminub		SSE7, SSE4
SSE2 = a4; // __asm	SSE3_MOVE	SSE2, [esi + ebx]
SSE1 = _mm_min_epu8(SSE1, SSE3); // __asm	pminub		SSE1, SSE3
SSE4 = _mm_max_epu8(SSE4, SSE6); // __asm	pmaxub		SSE4, SSE6
SSE3 = SSE2; // __asm	SSE_RMOVE	SSE3, SSE2
SSE0 = _mm_max_epu8(SSE0, SSE7); // __asm	pmaxub		SSE0, SSE7
SSE3 = _mm_min_epu8(SSE3, SSE5); // __asm	pminub		SSE3, SSE5
SSE1 = _mm_min_epu8(SSE1, SSE4); // __asm	pminub		SSE1, SSE4
SSE5 = _mm_max_epu8(SSE5, SSE2); // __asm	pmaxub		SSE5, SSE2
SSE0 = _mm_max_epu8(SSE0, SSE3); // __asm	pmaxub		SSE0, SSE3
SSE1 = _mm_min_epu8(SSE1, SSE5); // __asm	pminub		SSE1, SSE5
SSE7 = SSE2; // __asm	SSE_RMOVE	SSE7, SSE2
SSE3 = SSE6; // __asm	SSE_RMOVE	SSE3, SSE6
SSE4 = a8; // __asm	SSE3_MOVE	SSE4, [esi + 2 * ebx + 2]
SSE2 = _mm_min_epu8(SSE2, SSE6); // __asm	pminub		SSE2, SSE6
SSE3 = _mm_max_epu8(SSE3, SSE7); // __asm	pmaxub		SSE3, SSE7
SSE5 = SSE4; // __asm	SSE_RMOVE	SSE5, SSE4
SSE0 = _mm_max_epu8(SSE0, SSE2); // __asm	pmaxub		SSE0, SSE2
SSE5 = _mm_min_epu8(SSE5, SSE6); // __asm	pminub		SSE5, SSE6
SSE2 = a1; // __asm	SSE3_MOVE	SSE2, [esi]
SSE1 = _mm_min_epu8(SSE1, SSE3); // __asm	pminub		SSE1, SSE3
SSE6 = _mm_max_epu8(SSE6, SSE4); // __asm	pmaxub		SSE6, SSE4
SSE0 = _mm_max_epu8(SSE0, SSE5); // __asm	pmaxub		SSE0, SSE5
SSE3 = SSE2; // __asm	SSE_RMOVE	SSE3, SSE2
SSE1 = _mm_min_epu8(SSE1, SSE6); // __asm	pminub		SSE1, SSE6
SSE2 = _mm_min_epu8(SSE2, SSE7); // __asm	pminub		SSE2, SSE7
SSE3 = _mm_max_epu8(SSE3, SSE7); // __asm	pmaxub		SSE3, SSE7
SSE0 = _mm_max_epu8(SSE0, SSE2); // __asm	pmaxub		SSE0, SSE2
SSE1 = _mm_min_epu8(SSE1, SSE3); // __asm	pminub		SSE1, SSE3
SSE2 = SSE0; //__asm	SSE_RMOVE	SSE2, SSE0

// #if		(ISSE > 1) || defined(SHLUR)
// #ifdef	MODIFYPLUGIN (repair)
// SSE4 = reference; // __asm	SSE3_MOVE	SSE4, [edi]
// #else
SSE4 = c; // __asm	SSE3_MOVE	SSE4, [esi + ebx + 1]
//#endif
//#endif
// #if		MODIFYPLUGIN > 0
// SSE5 = c; // __asm	SSE3_MOVE	SSE5, [esi + ebx + 1]
//#endif

SSE0 = _mm_min_epu8(SSE0, SSE1); //  __asm	pminub		SSE0, SSE1
SSE2 = _mm_max_epu8(SSE2, SSE1); // __asm	pmaxub		SSE2, SSE1
//#if		MODIFYPLUGIN > 0
// SSE0 = _mm_min_epu8(SSE0, SSE5); //__asm	pminub		SSE0, SSE5
// SSE2 = _mm_max_epu8(SSE2, SSE5); // __asm	pmaxub		SSE2, SSE5
//#endif

//#ifdef	SHLUR
//sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
//__asm	add			esi, SSE_INCREMENT
//__asm	SSE_MOVE[edi], SSE4
//#else
//#if	ISSE > 1
SSE0 = _mm_max_epu8(SSE0, SSE4); // __asm	pmaxub		SSE0, SSE4
//#else
  // #ifdef	MODIFYPLUGIN (repair)
  // SSE0 = _mm_max_epu8(SSE0, reference); //__asm	pmaxub		SSE0, [edi]
  //#else
  //SSE0 = _mm_max_epu8(SEE0, c); //__asm	pmaxub		SSE0, [esi + ebx + 1]
  //#endif
//#endif
SSE0 = _mm_min_epu8(SSE0, SSE2); // __asm	pminub		SSE0, SSE2
Store SSE0; // __asm	SSE_MOVE[edi], SSE0
//#endif	// SHLUR
#endif // mod27

// Mode28_SmartRGCL2.cpp
// For my sources it gave identical result as mode 27, even if I made intentional
// errors in source.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i rg_mode28_sse(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);
  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  auto mi12 = _mm_min_epu8(a1, a2);
  auto ma12 = _mm_max_epu8(a1, a2);

  auto mi23 = _mm_min_epu8(a2, a3);
  auto ma23 = _mm_max_epu8(a2, a3);
  auto lower = _mm_max_epu8(mi12, mi23);
  auto upper = _mm_min_epu8(ma12, ma23);

  auto mi35 = _mm_min_epu8(a3, a5);
  auto ma35 = _mm_max_epu8(a3, a5);
  lower = _mm_max_epu8(lower, mi35);
  upper = _mm_min_epu8(upper, ma35);

  auto mi58 = _mm_min_epu8(a5, a8);
  auto ma58 = _mm_max_epu8(a5, a8);
  lower = _mm_max_epu8(lower, mi58);
  upper = _mm_min_epu8(upper, ma58);
  
  auto mi78 = _mm_min_epu8(a7, a8);
  auto ma78 = _mm_max_epu8(a7, a8);
  lower = _mm_max_epu8(lower, mi78);
  upper = _mm_min_epu8(upper, ma78);
  
  auto mi67 = _mm_min_epu8(a6, a7);
  auto ma67 = _mm_max_epu8(a6, a7);
  lower = _mm_max_epu8(lower, mi67);
  upper = _mm_min_epu8(upper, ma67);
  
  auto mi46 = _mm_min_epu8(a4, a6);
  auto ma46 = _mm_max_epu8(a4, a6);
  lower = _mm_max_epu8(lower, mi46);
  upper = _mm_min_epu8(upper, ma46);
  
  auto mi14 = _mm_min_epu8(a1, a4);
  auto ma14 = _mm_max_epu8(a1, a4);
  lower = _mm_max_epu8(lower, mi14);
  upper = _mm_min_epu8(upper, ma14);

  auto mi18 = _mm_min_epu8(a1, a8);
  auto ma18 = _mm_max_epu8(a1, a8);
  lower = _mm_max_epu8(lower, mi18);
  upper = _mm_min_epu8(upper, ma18);
  
  auto mi36 = _mm_min_epu8(a3, a6);
  auto ma36 = _mm_max_epu8(a3, a6);
  lower = _mm_max_epu8(lower, mi36);
  upper = _mm_min_epu8(upper, ma36);
  
  auto mi27 = _mm_min_epu8(a2, a7);
  auto ma27 = _mm_max_epu8(a2, a7);
  lower = _mm_max_epu8(lower, mi27);
  upper = _mm_min_epu8(upper, ma27);
  
  auto mi45 = _mm_min_epu8(a4, a5);
  auto ma45 = _mm_max_epu8(a4, a5);
  lower = _mm_max_epu8(lower, mi45);
  upper = _mm_min_epu8(upper, ma45);

  auto real_upper = _mm_max_epu8(upper, lower);
  auto real_lower = _mm_min_epu8(upper, lower);

  return simd_clip(c, real_lower, real_upper);
}


#ifdef mode28
sse7 = a1; // __asm	SSE3_MOVE	SSE7, [esi]
sse6 = a2; // __asm	SSE3_MOVE	SSE6, [esi + 1]
sse0 = sse7; // __asm	SSE_RMOVE	SSE0, SSE7
sse1 = sse6; // __asm	SSE_RMOVE	SSE1, SSE6
sse5 = a3; //__asm	SSE3_MOVE	SSE5, [esi + 2]
sse0 = _mm_min_epu8(sse0, sse6); // __asm	pminub		SSE0, SSE6
sse2 = sse5; // __asm	SSE_RMOVE	SSE2, SSE5
sse1 = _mm_max_epu8(sse1, sse7); // __asm	pmaxub		SSE1, SSE7
sse2 = _mm_min_epu8(sse2, sse6); // __asm	pminub		SSE2, SSE6
sse4 = a5; //  __asm	SSE3_MOVE	SSE4, [esi + ebx + 2]
sse6 = _mm_max_epu8(sse6, sse5); // __asm	pmaxub		SSE6, SSE5
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse4; // __asm	SSE_RMOVE	SSE3, SSE4
sse1 = _mm_min_epu8(sse1, sse6); // __asm	pminub		SSE1, SSE6
sse3 = _mm_min_epu8(sse3, sse5); // __asm	pminub		SSE3, SSE5
sse6 = a8; // __asm	SSE3_MOVE	SSE6, [esi + 2 * ebx + 2]
sse5 = _mm_max_epu8(sse5, sse4); // __asm	pmaxub		SSE5, SSE4
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse2 = sse6; // __asm	SSE_RMOVE	SSE2, SSE6
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5
sse2 = _mm_min_epu8(sse2, sse4); // __asm	pminub		SSE2, SSE4
sse5 = a7; //  __asm	SSE3_MOVE	SSE5, [esi + 2 * ebx + 1]
sse4 = _mm_max_epu8(sse4, sse6); // __asm	pmaxub		SSE4, SSE6
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse5; //  __asm	SSE_RMOVE	SSE3, SSE5
sse1 = _mm_min_epu8(sse1, sse4); // __asm	pminub		SSE1, SSE4
sse3 = _mm_min_epu8(sse3, sse6); // __asm	pminub		SSE3, SSE6
sse4 = a6; // __asm	SSE3_MOVE	SSE4, [esi + 2 * ebx]
sse6 = _mm_max_epu8(sse6, sse5); // __asm	pmaxub		SSE6, SSE5
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse2 = sse4; //  __asm	SSE_RMOVE	SSE2, SSE4
sse1 = _mm_min_epu8(sse1, sse6); // __asm	pminub		SSE1, SSE6
sse2 = _mm_min_epu8(sse2, sse5); // __asm	pminub		SSE2, SSE5
sse6 = a4; //  __asm	SSE3_MOVE	SSE6, [esi + ebx]
sse5 = _mm_max_epu8(sse5, sse4); // __asm	pmaxub		SSE5, SSE4
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse3 = sse6; // __asm	SSE_RMOVE	SSE3, SSE6
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5
sse3 = _mm_min_epu8(sse3, sse4); // __asm	pminub		SSE3, SSE4
sse2 = sse7; // __asm	SSE_RMOVE	SSE2, SSE7
sse4 = _mm_max_epu8(sse4, sse6); // __asm	pmaxub		SSE4, SSE6
sse0 = _mm_max_epu8(sse0, sse3); // __asm	pmaxub		SSE0, SSE3
sse1 = _mm_min_epu8(sse1, sse4); // __asm	pminub		SSE1, SSE4
sse2 = _mm_min_epu8(sse2, sse6); // __asm	pminub		SSE2, SSE6
sse7 = _mm_max_epu8(sse7, sse6); // __asm	pmaxub		SSE7, SSE6
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse1 = _mm_min_epu8(sse1, sse7); // __asm	pminub		SSE1, SSE7

sse2 = a1; // __asm	SSE3_MOVE	SSE2, [esi]
sse7 = a8; // __asm	SSE3_MOVE	SSE7, [esi + 2 * ebx + 2]
sse3 = sse2; // __asm	SSE_RMOVE	SSE3, SSE2
sse4 = a3; // __asm	SSE3_MOVE	SSE4, [esi + 2]
sse2 = _mm_min_epu8(sse2, sse7); // __asm	pminub		SSE2, SSE7
sse6 = a6; // __asm	SSE3_MOVE	SSE6, [esi + 2 * ebx]
sse5 = sse4; // __asm	SSE_RMOVE	SSE5, SSE4
sse3 = _mm_max_epu8(sse3, sse7); //  __asm	pmaxub		SSE3, SSE7
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse1 = _mm_min_epu8(sse1, sse3); // __asm	pminub		SSE1, SSE3
sse4 = _mm_min_epu8(sse4, sse6); // __asm	pminub		SSE4, SSE6
sse2 = a2; // __asm	SSE3_MOVE	SSE2, [esi + 1]
sse5 = _mm_max_epu8(sse5, sse6); // __asm	pmaxub		SSE5, SSE6
sse0 = _mm_max_epu8(sse0, sse4); // __asm	pmaxub		SSE0, SSE4
sse7 = a7; // __asm	SSE3_MOVE	SSE7, [esi + 2 * ebx + 1]
sse3 = sse2; // __asm	SSE_RMOVE	SSE3, SSE2
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5
sse2 = _mm_min_epu8(sse2, sse7); // __asm	pminub		SSE2, SSE7
sse4 = a4; // __asm	SSE3_MOVE	SSE4, [esi + ebx]
sse3 = _mm_max_epu8(sse3, sse7); // __asm	pmaxub		SSE3, SSE7
sse0 = _mm_max_epu8(sse0, sse2); // __asm	pmaxub		SSE0, SSE2
sse7 = a5; // __asm	SSE3_MOVE	SSE7, [esi + ebx + 2]
sse5 = sse4; // __asm	SSE_RMOVE	SSE5, SSE4
sse1 = _mm_min_epu8(sse1, sse3); // __asm	pminub		SSE1, SSE3
sse4 = _mm_min_epu8(sse4, sse7); // __asm	pminub		SSE4, SSE7
sse5 = _mm_max_epu8(sse5, sse7); // __asm	pmaxub		SSE5, SSE7
sse0 = _mm_max_epu8(sse0, sse4); // __asm	pmaxub		SSE0, SSE4
sse1 = _mm_min_epu8(sse1, sse5); // __asm	pminub		SSE1, SSE5

sse2 = sse0; // __asm	SSE_RMOVE	SSE2, SSE0
//#if		(ISSE > 1) || defined(SHLUR)
//#ifdef	MODIFYPLUGIN (repair)
// sse4 = reference__asm	SSE3_MOVE	SSE4, [edi]
//#else
sse4 = c; // __asm	SSE3_MOVE	SSE4, [esi + ebx + 1]
//#endif
//#endif
//#if		MODIFYPLUGIN > 0(repair)
//sse5 = c; //__asm	SSE3_MOVE	SSE5, [esi + ebx + 1]
//#endif

sse0 = _mm_min_epu8(sse0, sse1); // __asm	pminub		SSE0, SSE1
sse2 = _mm_max_epu8(sse2, sse1); // __asm	pmaxub		SSE2, SSE1
//#if		MODIFYPLUGIN > 0 repair
// sse0 = _mm_min_epu8(sse0, sse5); //__asm	pminub		SSE0, SSE5
// sse2 = _mm_max_epu8(sse2, sse5); //__asm	pmaxub		SSE2, SSE5
//#endif

//#ifdef	SHLUR
//sharpen(SSE4, SSE0, SSE2, rshift[eax], shift_mask[eax], SSE7, SSE3)
//__asm	add			esi, SSE_INCREMENT
//__asm	SSE_MOVE[edi], SSE4
//#else
//#if	ISSE > 1
sse0 = _mm_max_epu8(sse0, sse4); // __asm	pmaxub		SSE0, SSE4
//#else
//#ifdef	MODIFYPLUGIN
//__asm	pmaxub		SSE0, [edi]
//#else
//__asm	pmaxub		SSE0, [esi + ebx + 1]
//#endif
//#endif
sse0 = _mm_min_epu8(sse0, sse2); // __asm	pminub		SSE0, SSE2
auto result = sse0; // __asm	SSE_MOVE[edi], SSE0
//#endif	// SHLUR
//#endif
#endif // mode28

#endif
