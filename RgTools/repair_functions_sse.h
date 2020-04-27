#ifndef __REPAIR_FUNCTIONS_SSE_H__
#define __REPAIR_FUNCTIONS_SSE_H__

#include "common.h"

typedef __m128i (SseModeProcessor)(const Byte*, const __m128i &val, int);

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode1_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    __m128i mi = _mm_min_epu8(_mm_min_epu8(
        _mm_min_epu8(_mm_min_epu8(a1, a2), _mm_min_epu8(a3, a4)),
        _mm_min_epu8(_mm_min_epu8(a5, a6), _mm_min_epu8(a7, a8))
        ), c);
    __m128i ma = _mm_max_epu8(_mm_max_epu8(
        _mm_max_epu8(_mm_max_epu8(a1, a2), _mm_max_epu8(a3, a4)),
        _mm_max_epu8(_mm_max_epu8(a5, a6), _mm_max_epu8(a7, a8))
        ), c);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode1_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  __m128i mi = _mm_min_epu8(_mm_min_epu8(
    _mm_min_epu8(_mm_min_epu8(a1, a2), _mm_min_epu8(a3, a4)),
    _mm_min_epu8(_mm_min_epu8(a5, a6), _mm_min_epu8(a7, a8))
  ), c);
  __m128i ma = _mm_max_epu8(_mm_max_epu8(
    _mm_max_epu8(_mm_max_epu8(a1, a2), _mm_max_epu8(a3, a4)),
    _mm_max_epu8(_mm_max_epu8(a5, a6), _mm_max_epu8(a7, a8))
  ), c);

  return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode1_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  __m128i mi = _mm_min_epu16(_mm_min_epu16(
    _mm_min_epu16(_mm_min_epu16(a1, a2), _mm_min_epu16(a3, a4)),
    _mm_min_epu16(_mm_min_epu16(a5, a6), _mm_min_epu16(a7, a8))
  ), c);
  __m128i ma = _mm_max_epu16(_mm_max_epu16(
    _mm_max_epu16(_mm_max_epu16(a1, a2), _mm_max_epu16(a3, a4)),
    _mm_max_epu16(_mm_max_epu16(a5, a6), _mm_max_epu16(a7, a8))
  ), c);

  return simd_clip_16(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode1_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  __m128 mi = _mm_min_ps(_mm_min_ps(
    _mm_min_ps(_mm_min_ps(a1, a2), _mm_min_ps(a3, a4)),
    _mm_min_ps(_mm_min_ps(a5, a6), _mm_min_ps(a7, a8))
  ), c);
  __m128 ma = _mm_max_ps(_mm_max_ps(
    _mm_max_ps(_mm_max_ps(a1, a2), _mm_max_ps(a3, a4)),
    _mm_max_ps(_mm_max_ps(a5, a6), _mm_max_ps(a7, a8))
  ), c);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode2_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair(a1, a8);

    sort_pair(a1,  c);
    sort_pair(a2, a5);
    sort_pair(a3, a6);
    sort_pair(a4, a7);
    sort_pair( c, a8);

    sort_pair(a1, a3);
    sort_pair( c, a6);
    sort_pair(a2, a4);
    sort_pair(a5, a7);

    sort_pair(a3, a8);

    sort_pair(a3,  c);
    sort_pair(a6, a8);
    sort_pair(a4, a5);

    a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
    a3 = _mm_min_epu8(a3, a4);	// sort_pair (a3, a4);
    sort_pair( c, a5);
    a7 = _mm_max_epu8(a6, a7);	// sort_pair (a6, a7);

    sort_pair(a2, a8);

    a2 = _mm_min_epu8(a2,  c);	// sort_pair (a2,  c);
    a8 = _mm_max_epu8(a5, a8);	// sort_pair (a5, a8);

    a2 = _mm_min_epu8(a2, a3);	// sort_pair (a2, a3);
    a7 = _mm_min_epu8(a7, a8);	// sort_pair (a7, a8);

    return simd_clip(val, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode2_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a8);

  sort_pair(a1, c);
  sort_pair(a2, a5);
  sort_pair(a3, a6);
  sort_pair(a4, a7);
  sort_pair(c, a8);

  sort_pair(a1, a3);
  sort_pair(c, a6);
  sort_pair(a2, a4);
  sort_pair(a5, a7);

  sort_pair(a3, a8);

  sort_pair(a3, c);
  sort_pair(a6, a8);
  sort_pair(a4, a5);

  a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
  a3 = _mm_min_epu8(a3, a4);	// sort_pair (a3, a4);
  sort_pair(c, a5);
  a7 = _mm_max_epu8(a6, a7);	// sort_pair (a6, a7);

  sort_pair(a2, a8);

  a2 = _mm_min_epu8(a2, c);	// sort_pair (a2,  c);
  a8 = _mm_max_epu8(a5, a8);	// sort_pair (a5, a8);

  a2 = _mm_min_epu8(a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_min_epu8(a7, a8);	// sort_pair (a7, a8);

  return simd_clip(val, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode2_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16(a1, a8);

  sort_pair_16(a1,  c);
  sort_pair_16(a2, a5);
  sort_pair_16(a3, a6);
  sort_pair_16(a4, a7);
  sort_pair_16( c, a8);

  sort_pair_16(a1, a3);
  sort_pair_16( c, a6);
  sort_pair_16(a2, a4);
  sort_pair_16(a5, a7);

  sort_pair_16(a3, a8);

  sort_pair_16(a3,  c);
  sort_pair_16(a6, a8);
  sort_pair_16(a4, a5);

  a2 = _mm_max_epu16(a1, a2);	// sort_pair_16 (a1, a2);
  a3 = _mm_min_epu16(a3, a4);	// sort_pair_16 (a3, a4);
  sort_pair_16( c, a5);
  a7 = _mm_max_epu16(a6, a7);	// sort_pair_16 (a6, a7);

  sort_pair_16(a2, a8);

  a2 = _mm_min_epu16(a2,  c);	// sort_pair_16 (a2,  c);
  a8 = _mm_max_epu16(a5, a8);	// sort_pair_16 (a5, a8);

  a2 = _mm_min_epu16(a2, a3);	// sort_pair_16 (a2, a3);
  a7 = _mm_min_epu16(a7, a8);	// sort_pair_16 (a7, a8);

  return simd_clip_16(val, a2, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode2_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32(a1, a8);

  sort_pair_32(a1,  c);
  sort_pair_32(a2, a5);
  sort_pair_32(a3, a6);
  sort_pair_32(a4, a7);
  sort_pair_32( c, a8);

  sort_pair_32(a1, a3);
  sort_pair_32( c, a6);
  sort_pair_32(a2, a4);
  sort_pair_32(a5, a7);

  sort_pair_32(a3, a8);

  sort_pair_32(a3,  c);
  sort_pair_32(a6, a8);
  sort_pair_32(a4, a5);

  a2 = _mm_max_ps(a1, a2);	// sort_pair_32 (a1, a2);
  a3 = _mm_min_ps(a3, a4);	// sort_pair_32 (a3, a4);
  sort_pair_32( c, a5);
  a7 = _mm_max_ps(a6, a7);	// sort_pair_32 (a6, a7);

  sort_pair_32(a2, a8);

  a2 = _mm_min_ps(a2,  c);	// sort_pair_32 (a2,  c);
  a8 = _mm_max_ps(a5, a8);	// sort_pair_32 (a5, a8);

  a2 = _mm_min_ps(a2, a3);	// sort_pair_32 (a2, a3);
  a7 = _mm_min_ps(a7, a8);	// sort_pair_32 (a7, a8);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), a2, a7));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode3_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair(a1, a8);

    sort_pair(a1,  c);
    sort_pair(a2, a5);
    sort_pair(a3, a6);
    sort_pair(a4, a7);
    sort_pair( c, a8);

    sort_pair(a1, a3);
    sort_pair( c, a6);
    sort_pair(a2, a4);
    sort_pair(a5, a7);

    sort_pair(a3, a8);

    sort_pair(a3,  c);
    sort_pair(a6, a8);
    sort_pair(a4, a5);

    a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
    sort_pair(a3, a4);
    sort_pair( c, a5);
    a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

    sort_pair(a2, a8);

    a2 = _mm_min_epu8(a2,  c);	// sort_pair (a2,  c);
    a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);
    a5 = _mm_min_epu8(a5, a8);	// sort_pair (a5, a8);

    a3 = _mm_max_epu8(a2, a3);	// sort_pair (a2, a3);
    a6 = _mm_max_epu8(a5, a6);	// sort_pair (a5, a6);

    return simd_clip(val, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode3_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a8);

  sort_pair(a1, c);
  sort_pair(a2, a5);
  sort_pair(a3, a6);
  sort_pair(a4, a7);
  sort_pair(c, a8);

  sort_pair(a1, a3);
  sort_pair(c, a6);
  sort_pair(a2, a4);
  sort_pair(a5, a7);

  sort_pair(a3, a8);

  sort_pair(a3, c);
  sort_pair(a6, a8);
  sort_pair(a4, a5);

  a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
  sort_pair(a3, a4);
  sort_pair(c, a5);
  a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

  sort_pair(a2, a8);

  a2 = _mm_min_epu8(a2, c);	// sort_pair (a2,  c);
  a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);
  a5 = _mm_min_epu8(a5, a8);	// sort_pair (a5, a8);

  a3 = _mm_max_epu8(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_max_epu8(a5, a6);	// sort_pair (a5, a6);

  return simd_clip(val, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode3_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16(a1, a8);

  sort_pair_16(a1,  c);
  sort_pair_16(a2, a5);
  sort_pair_16(a3, a6);
  sort_pair_16(a4, a7);
  sort_pair_16( c, a8);

  sort_pair_16(a1, a3);
  sort_pair_16( c, a6);
  sort_pair_16(a2, a4);
  sort_pair_16(a5, a7);

  sort_pair_16(a3, a8);

  sort_pair_16(a3,  c);
  sort_pair_16(a6, a8);
  sort_pair_16(a4, a5);

  a2 = _mm_max_epu16(a1, a2);	// sort_pair_16 (a1, a2);
  sort_pair_16(a3, a4);
  sort_pair_16( c, a5);
  a6 = _mm_min_epu16(a6, a7);	// sort_pair_16 (a6, a7);

  sort_pair_16(a2, a8);

  a2 = _mm_min_epu16(a2,  c);	// sort_pair_16 (a2,  c);
  a6 = _mm_max_epu16(a4, a6);	// sort_pair_16 (a4, a6);
  a5 = _mm_min_epu16(a5, a8);	// sort_pair_16 (a5, a8);

  a3 = _mm_max_epu16(a2, a3);	// sort_pair_16 (a2, a3);
  a6 = _mm_max_epu16(a5, a6);	// sort_pair_16 (a5, a6);

  return simd_clip_16(val, a3, a6);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode3_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32(a1, a8);

  sort_pair_32(a1,  c);
  sort_pair_32(a2, a5);
  sort_pair_32(a3, a6);
  sort_pair_32(a4, a7);
  sort_pair_32( c, a8);

  sort_pair_32(a1, a3);
  sort_pair_32( c, a6);
  sort_pair_32(a2, a4);
  sort_pair_32(a5, a7);

  sort_pair_32(a3, a8);

  sort_pair_32(a3,  c);
  sort_pair_32(a6, a8);
  sort_pair_32(a4, a5);

  a2 = _mm_max_ps(a1, a2);	// sort_pair_32 (a1, a2);
  sort_pair_32(a3, a4);
  sort_pair_32( c, a5);
  a6 = _mm_min_ps(a6, a7);	// sort_pair_32 (a6, a7);

  sort_pair_32(a2, a8);

  a2 = _mm_min_ps(a2,  c);	// sort_pair_32 (a2,  c);
  a6 = _mm_max_ps(a4, a6);	// sort_pair_32 (a4, a6);
  a5 = _mm_min_ps(a5, a8);	// sort_pair_32 (a5, a8);

  a3 = _mm_max_ps(a2, a3);	// sort_pair_32 (a2, a3);
  a6 = _mm_max_ps(a5, a6);	// sort_pair_32 (a5, a6);
  
  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), a3, a6));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode4_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    sort_pair(a1, a8);

    sort_pair(a1,  c);
    sort_pair(a2, a5);
    sort_pair(a3, a6);
    sort_pair(a4, a7);
    sort_pair( c, a8);

    sort_pair(a1, a3);
    sort_pair( c, a6);
    sort_pair(a2, a4);
    sort_pair(a5, a7);

    sort_pair(a3, a8);

    sort_pair(a3,  c);
    sort_pair(a6, a8);
    sort_pair(a4, a5);

    a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
    a4 = _mm_max_epu8(a3, a4);	// sort_pair (a3, a4);
    sort_pair ( c, a5);
    a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

    sort_pair (a2, a8);

    c  = _mm_max_epu8(a2,  c);	// sort_pair (a2,  c);
    sort_pair (a4, a6);
    a5 = _mm_min_epu8(a5, a8);	// sort_pair (a5, a8);

    a4 = _mm_min_epu8(a4,  c);	// sort_pair (a4,  c);
    a5 = _mm_min_epu8(a5, a6);	// sort_pair (a5, a6);

    return simd_clip(val, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode4_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  sort_pair(a1, a8);

  sort_pair(a1, c);
  sort_pair(a2, a5);
  sort_pair(a3, a6);
  sort_pair(a4, a7);
  sort_pair(c, a8);

  sort_pair(a1, a3);
  sort_pair(c, a6);
  sort_pair(a2, a4);
  sort_pair(a5, a7);

  sort_pair(a3, a8);

  sort_pair(a3, c);
  sort_pair(a6, a8);
  sort_pair(a4, a5);

  a2 = _mm_max_epu8(a1, a2);	// sort_pair (a1, a2);
  a4 = _mm_max_epu8(a3, a4);	// sort_pair (a3, a4);
  sort_pair(c, a5);
  a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

  sort_pair(a2, a8);

  c = _mm_max_epu8(a2, c);	// sort_pair (a2,  c);
  sort_pair(a4, a6);
  a5 = _mm_min_epu8(a5, a8);	// sort_pair (a5, a8);

  a4 = _mm_min_epu8(a4, c);	// sort_pair (a4,  c);
  a5 = _mm_min_epu8(a5, a6);	// sort_pair (a5, a6);

  return simd_clip(val, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode4_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  sort_pair_16(a1, a8);

  sort_pair_16(a1,  c);
  sort_pair_16(a2, a5);
  sort_pair_16(a3, a6);
  sort_pair_16(a4, a7);
  sort_pair_16( c, a8);

  sort_pair_16(a1, a3);
  sort_pair_16( c, a6);
  sort_pair_16(a2, a4);
  sort_pair_16(a5, a7);

  sort_pair_16(a3, a8);

  sort_pair_16(a3,  c);
  sort_pair_16(a6, a8);
  sort_pair_16(a4, a5);

  a2 = _mm_max_epu16(a1, a2);	// sort_pair_16 (a1, a2);
  a4 = _mm_max_epu16(a3, a4);	// sort_pair_16 (a3, a4);
  sort_pair_16 ( c, a5);
  a6 = _mm_min_epu16(a6, a7);	// sort_pair_16 (a6, a7);

  sort_pair_16 (a2, a8);

  c  = _mm_max_epu16(a2,  c);	// sort_pair_16 (a2,  c);
  sort_pair_16 (a4, a6);
  a5 = _mm_min_epu16(a5, a8);	// sort_pair_16 (a5, a8);

  a4 = _mm_min_epu16(a4,  c);	// sort_pair_16 (a4,  c);
  a5 = _mm_min_epu16(a5, a6);	// sort_pair_16 (a5, a6);

  return simd_clip_16(val, a4, a5);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode4_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  sort_pair_32(a1, a8);

  sort_pair_32(a1,  c);
  sort_pair_32(a2, a5);
  sort_pair_32(a3, a6);
  sort_pair_32(a4, a7);
  sort_pair_32( c, a8);

  sort_pair_32(a1, a3);
  sort_pair_32( c, a6);
  sort_pair_32(a2, a4);
  sort_pair_32(a5, a7);

  sort_pair_32(a3, a8);

  sort_pair_32(a3,  c);
  sort_pair_32(a6, a8);
  sort_pair_32(a4, a5);

  a2 = _mm_max_ps(a1, a2);	// sort_pair_32 (a1, a2);
  a4 = _mm_max_ps(a3, a4);	// sort_pair_32 (a3, a4);
  sort_pair_32 ( c, a5);
  a6 = _mm_min_ps(a6, a7);	// sort_pair_32 (a6, a7);

  sort_pair_32 (a2, a8);

  c  = _mm_max_ps(a2,  c);	// sort_pair_32 (a2,  c);
  sort_pair_32 (a4, a6);
  a5 = _mm_min_ps(a5, a8);	// sort_pair_32 (a5, a8);

  a4 = _mm_min_ps(a4,  c);	// sort_pair_32 (a4,  c);
  a5 = _mm_min_ps(a5, a6);	// sort_pair_32 (a5, a6);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), a4, a5));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode5_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
    auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

    auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
    auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

    auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
    auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

    auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
    auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

    auto clipped1 = simd_clip(val, mil1, mal1);
    auto clipped2 = simd_clip(val, mil2, mal2);
    auto clipped3 = simd_clip(val, mil3, mal3);
    auto clipped4 = simd_clip(val, mil4, mal4);

    auto c1 = abs_diff(val, clipped1);
    auto c2 = abs_diff(val, clipped2);
    auto c3 = abs_diff(val, clipped3);
    auto c4 = abs_diff(val, clipped4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, val, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode5_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
  auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

  auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
  auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

  auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
  auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

  auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
  auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

  auto clipped1 = simd_clip(val, mil1, mal1);
  auto clipped2 = simd_clip(val, mil2, mal2);
  auto clipped3 = simd_clip(val, mil3, mal3);
  auto clipped4 = simd_clip(val, mil4, mal4);

  auto c1 = abs_diff(val, clipped1);
  auto c2 = abs_diff(val, clipped2);
  auto c3 = abs_diff(val, clipped3);
  auto c4 = abs_diff(val, clipped4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, val, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode5_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(_mm_max_epu16(a1, a8), c);
  auto mil1 = _mm_min_epu16(_mm_min_epu16(a1, a8), c);

  auto mal2 = _mm_max_epu16(_mm_max_epu16(a2, a7), c);
  auto mil2 = _mm_min_epu16(_mm_min_epu16(a2, a7), c);

  auto mal3 = _mm_max_epu16(_mm_max_epu16(a3, a6), c);
  auto mil3 = _mm_min_epu16(_mm_min_epu16(a3, a6), c);

  auto mal4 = _mm_max_epu16(_mm_max_epu16(a4, a5), c);
  auto mil4 = _mm_min_epu16(_mm_min_epu16(a4, a5), c);

  auto clipped1 = simd_clip_16(val, mil1, mal1);
  auto clipped2 = simd_clip_16(val, mil2, mal2);
  auto clipped3 = simd_clip_16(val, mil3, mal3);
  auto clipped4 = simd_clip_16(val, mil4, mal4);

  auto c1 = abs_diff_16(val, clipped1);
  auto c2 = abs_diff_16(val, clipped2);
  auto c3 = abs_diff_16(val, clipped3);
  auto c4 = abs_diff_16(val, clipped4);

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, val, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode5_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(_mm_max_ps(a1, a8), c);
  auto mil1 = _mm_min_ps(_mm_min_ps(a1, a8), c);

  auto mal2 = _mm_max_ps(_mm_max_ps(a2, a7), c);
  auto mil2 = _mm_min_ps(_mm_min_ps(a2, a7), c);

  auto mal3 = _mm_max_ps(_mm_max_ps(a3, a6), c);
  auto mil3 = _mm_min_ps(_mm_min_ps(a3, a6), c);

  auto mal4 = _mm_max_ps(_mm_max_ps(a4, a5), c);
  auto mil4 = _mm_min_ps(_mm_min_ps(a4, a5), c);

  auto clipped1 = simd_clip_32(_mm_castsi128_ps(val), mil1, mal1);
  auto clipped2 = simd_clip_32(_mm_castsi128_ps(val), mil2, mal2);
  auto clipped3 = simd_clip_32(_mm_castsi128_ps(val), mil3, mal3);
  auto clipped4 = simd_clip_32(_mm_castsi128_ps(val), mil4, mal4);

  auto c1 = abs_diff_32(_mm_castsi128_ps(val), clipped1);
  auto c2 = abs_diff_32(_mm_castsi128_ps(val), clipped2);
  auto c3 = abs_diff_32(_mm_castsi128_ps(val), clipped3);
  auto c4 = abs_diff_32(_mm_castsi128_ps(val), clipped4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val), clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode6_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
    auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

    auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
    auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

    auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
    auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

    auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
    auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(val, mil1, mal1);
    auto clipped2 = simd_clip(val, mil2, mal2);
    auto clipped3 = simd_clip(val, mil3, mal3);
    auto clipped4 = simd_clip(val, mil4, mal4);

    auto absdiff1 = abs_diff(val, clipped1);
    auto absdiff2 = abs_diff(val, clipped2);
    auto absdiff3 = abs_diff(val, clipped3);
    auto absdiff4 = abs_diff(val, clipped4);

    auto c1 = _mm_adds_epu8(_mm_adds_epu8(absdiff1, absdiff1), d1);
    auto c2 = _mm_adds_epu8(_mm_adds_epu8(absdiff2, absdiff2), d2);
    auto c3 = _mm_adds_epu8(_mm_adds_epu8(absdiff3, absdiff3), d3);
    auto c4 = _mm_adds_epu8(_mm_adds_epu8(absdiff4, absdiff4), d4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, val, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode6_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
  auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

  auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
  auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

  auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
  auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

  auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
  auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(val, mil1, mal1);
  auto clipped2 = simd_clip(val, mil2, mal2);
  auto clipped3 = simd_clip(val, mil3, mal3);
  auto clipped4 = simd_clip(val, mil4, mal4);

  auto absdiff1 = abs_diff(val, clipped1);
  auto absdiff2 = abs_diff(val, clipped2);
  auto absdiff3 = abs_diff(val, clipped3);
  auto absdiff4 = abs_diff(val, clipped4);

  auto c1 = _mm_adds_epu8(_mm_adds_epu8(absdiff1, absdiff1), d1);
  auto c2 = _mm_adds_epu8(_mm_adds_epu8(absdiff2, absdiff2), d2);
  auto c3 = _mm_adds_epu8(_mm_adds_epu8(absdiff3, absdiff3), d3);
  auto c4 = _mm_adds_epu8(_mm_adds_epu8(absdiff4, absdiff4), d4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, val, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}


template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode6_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(_mm_max_epu16(a1, a8), c);
  auto mil1 = _mm_min_epu16(_mm_min_epu16(a1, a8), c);

  auto mal2 = _mm_max_epu16(_mm_max_epu16(a2, a7), c);
  auto mil2 = _mm_min_epu16(_mm_min_epu16(a2, a7), c);

  auto mal3 = _mm_max_epu16(_mm_max_epu16(a3, a6), c);
  auto mil3 = _mm_min_epu16(_mm_min_epu16(a3, a6), c);

  auto mal4 = _mm_max_epu16(_mm_max_epu16(a4, a5), c);
  auto mil4 = _mm_min_epu16(_mm_min_epu16(a4, a5), c);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(val, mil1, mal1);
  auto clipped2 = simd_clip_16(val, mil2, mal2);
  auto clipped3 = simd_clip_16(val, mil3, mal3);
  auto clipped4 = simd_clip_16(val, mil4, mal4);

  auto absdiff1 = abs_diff_16(val, clipped1);
  auto absdiff2 = abs_diff_16(val, clipped2);
  auto absdiff3 = abs_diff_16(val, clipped3);
  auto absdiff4 = abs_diff_16(val, clipped4);

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

  auto result = select_on_equal_16(mindiff, c1, val, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode6_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(_mm_max_ps(a1, a8), c);
  auto mil1 = _mm_min_ps(_mm_min_ps(a1, a8), c);

  auto mal2 = _mm_max_ps(_mm_max_ps(a2, a7), c);
  auto mil2 = _mm_min_ps(_mm_min_ps(a2, a7), c);

  auto mal3 = _mm_max_ps(_mm_max_ps(a3, a6), c);
  auto mil3 = _mm_min_ps(_mm_min_ps(a3, a6), c);

  auto mal4 = _mm_max_ps(_mm_max_ps(a4, a5), c);
  auto mil4 = _mm_min_ps(_mm_min_ps(a4, a5), c);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(_mm_castsi128_ps(val), mil1, mal1);
  auto clipped2 = simd_clip_32(_mm_castsi128_ps(val), mil2, mal2);
  auto clipped3 = simd_clip_32(_mm_castsi128_ps(val), mil3, mal3);
  auto clipped4 = simd_clip_32(_mm_castsi128_ps(val), mil4, mal4);

  auto absdiff1 = abs_diff_32(_mm_castsi128_ps(val), clipped1);
  auto absdiff2 = abs_diff_32(_mm_castsi128_ps(val), clipped2);
  auto absdiff3 = abs_diff_32(_mm_castsi128_ps(val), clipped3);
  auto absdiff4 = abs_diff_32(_mm_castsi128_ps(val), clipped4);

  auto c1 = _mm_add_ps(_mm_add_ps(absdiff1, absdiff1), d1); // no adds needed, only for comparison
  auto c2 = _mm_add_ps(_mm_add_ps(absdiff2, absdiff2), d2);
  auto c3 = _mm_add_ps(_mm_add_ps(absdiff3, absdiff3), d3);
  auto c4 = _mm_add_ps(_mm_add_ps(absdiff4, absdiff4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val), clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode7_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
    auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

    auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
    auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

    auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
    auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

    auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
    auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(val, mil1, mal1);
    auto clipped2 = simd_clip(val, mil2, mal2);
    auto clipped3 = simd_clip(val, mil3, mal3);
    auto clipped4 = simd_clip(val, mil4, mal4);
    //todo: what happens when this overflows?
    auto c1 = _mm_adds_epu8(abs_diff(val, clipped1), d1);
    auto c2 = _mm_adds_epu8(abs_diff(val, clipped2), d2);
    auto c3 = _mm_adds_epu8(abs_diff(val, clipped3), d3);
    auto c4 = _mm_adds_epu8(abs_diff(val, clipped4), d4);

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, val, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode7_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
  auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

  auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
  auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

  auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
  auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

  auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
  auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(val, mil1, mal1);
  auto clipped2 = simd_clip(val, mil2, mal2);
  auto clipped3 = simd_clip(val, mil3, mal3);
  auto clipped4 = simd_clip(val, mil4, mal4);
  //todo: what happens when this overflows?
  auto c1 = _mm_adds_epu8(abs_diff(val, clipped1), d1);
  auto c2 = _mm_adds_epu8(abs_diff(val, clipped2), d2);
  auto c3 = _mm_adds_epu8(abs_diff(val, clipped3), d3);
  auto c4 = _mm_adds_epu8(abs_diff(val, clipped4), d4);

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, val, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode7_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(_mm_max_epu16(a1, a8), c);
  auto mil1 = _mm_min_epu16(_mm_min_epu16(a1, a8), c);

  auto mal2 = _mm_max_epu16(_mm_max_epu16(a2, a7), c);
  auto mil2 = _mm_min_epu16(_mm_min_epu16(a2, a7), c);

  auto mal3 = _mm_max_epu16(_mm_max_epu16(a3, a6), c);
  auto mil3 = _mm_min_epu16(_mm_min_epu16(a3, a6), c);

  auto mal4 = _mm_max_epu16(_mm_max_epu16(a4, a5), c);
  auto mil4 = _mm_min_epu16(_mm_min_epu16(a4, a5), c);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(val, mil1, mal1);
  auto clipped2 = simd_clip_16(val, mil2, mal2);
  auto clipped3 = simd_clip_16(val, mil3, mal3);
  auto clipped4 = simd_clip_16(val, mil4, mal4);
  //todo: what happens when this overflows?
  auto c1 = _mm_adds_epu16(abs_diff_16(val, clipped1), d1);
  auto c2 = _mm_adds_epu16(abs_diff_16(val, clipped2), d2);
  auto c3 = _mm_adds_epu16(abs_diff_16(val, clipped3), d3);
  auto c4 = _mm_adds_epu16(abs_diff_16(val, clipped4), d4);

  auto mindiff = _mm_min_epu16(c1, c2);
  mindiff = _mm_min_epu16(mindiff, c3);
  mindiff = _mm_min_epu16(mindiff, c4);

  auto result = select_on_equal_16(mindiff, c1, val, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode7_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(_mm_max_ps(a1, a8), c);
  auto mil1 = _mm_min_ps(_mm_min_ps(a1, a8), c);

  auto mal2 = _mm_max_ps(_mm_max_ps(a2, a7), c);
  auto mil2 = _mm_min_ps(_mm_min_ps(a2, a7), c);

  auto mal3 = _mm_max_ps(_mm_max_ps(a3, a6), c);
  auto mil3 = _mm_min_ps(_mm_min_ps(a3, a6), c);

  auto mal4 = _mm_max_ps(_mm_max_ps(a4, a5), c);
  auto mil4 = _mm_min_ps(_mm_min_ps(a4, a5), c);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(_mm_castsi128_ps(val), mil1, mal1);
  auto clipped2 = simd_clip_32(_mm_castsi128_ps(val), mil2, mal2);
  auto clipped3 = simd_clip_32(_mm_castsi128_ps(val), mil3, mal3);
  auto clipped4 = simd_clip_32(_mm_castsi128_ps(val), mil4, mal4);

  auto c1 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped1), d1); // no adds needed, only for comparison
  auto c2 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped2), d2);
  auto c3 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped3), d3);
  auto c4 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val), clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode8_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
    auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

    auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
    auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

    auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
    auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

    auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
    auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto clipped1 = simd_clip(val, mil1, mal1);
    auto clipped2 = simd_clip(val, mil2, mal2);
    auto clipped3 = simd_clip(val, mil3, mal3);
    auto clipped4 = simd_clip(val, mil4, mal4);

    auto c1 = _mm_adds_epu8(abs_diff(val, clipped1), _mm_adds_epu8(d1, d1));
    auto c2 = _mm_adds_epu8(abs_diff(val, clipped2), _mm_adds_epu8(d2, d2));
    auto c3 = _mm_adds_epu8(abs_diff(val, clipped3), _mm_adds_epu8(d3, d3));
    auto c4 = _mm_adds_epu8(abs_diff(val, clipped4), _mm_adds_epu8(d4, d4));

    auto mindiff = _mm_min_epu8(c1, c2);
    mindiff = _mm_min_epu8(mindiff, c3);
    mindiff = _mm_min_epu8(mindiff, c4);

    auto result = select_on_equal(mindiff, c1, val, clipped1);
    result = select_on_equal(mindiff, c3, result, clipped3);
    result = select_on_equal(mindiff, c2, result, clipped2);
    return select_on_equal(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode8_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
  auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

  auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
  auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

  auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
  auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

  auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
  auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto clipped1 = simd_clip(val, mil1, mal1);
  auto clipped2 = simd_clip(val, mil2, mal2);
  auto clipped3 = simd_clip(val, mil3, mal3);
  auto clipped4 = simd_clip(val, mil4, mal4);

  auto c1 = _mm_adds_epu8(abs_diff(val, clipped1), _mm_adds_epu8(d1, d1));
  auto c2 = _mm_adds_epu8(abs_diff(val, clipped2), _mm_adds_epu8(d2, d2));
  auto c3 = _mm_adds_epu8(abs_diff(val, clipped3), _mm_adds_epu8(d3, d3));
  auto c4 = _mm_adds_epu8(abs_diff(val, clipped4), _mm_adds_epu8(d4, d4));

  auto mindiff = _mm_min_epu8(c1, c2);
  mindiff = _mm_min_epu8(mindiff, c3);
  mindiff = _mm_min_epu8(mindiff, c4);

  auto result = select_on_equal_sse4(mindiff, c1, val, clipped1);
  result = select_on_equal_sse4(mindiff, c3, result, clipped3);
  result = select_on_equal_sse4(mindiff, c2, result, clipped2);
  return select_on_equal_sse4(mindiff, c4, result, clipped4);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode8_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(_mm_max_epu16(a1, a8), c);
  auto mil1 = _mm_min_epu16(_mm_min_epu16(a1, a8), c);

  auto mal2 = _mm_max_epu16(_mm_max_epu16(a2, a7), c);
  auto mil2 = _mm_min_epu16(_mm_min_epu16(a2, a7), c);

  auto mal3 = _mm_max_epu16(_mm_max_epu16(a3, a6), c);
  auto mil3 = _mm_min_epu16(_mm_min_epu16(a3, a6), c);

  auto mal4 = _mm_max_epu16(_mm_max_epu16(a4, a5), c);
  auto mil4 = _mm_min_epu16(_mm_min_epu16(a4, a5), c);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto clipped1 = simd_clip_16(val, mil1, mal1);
  auto clipped2 = simd_clip_16(val, mil2, mal2);
  auto clipped3 = simd_clip_16(val, mil3, mal3);
  auto clipped4 = simd_clip_16(val, mil4, mal4);

  auto c1 = _mm_adds_epu16(abs_diff_16(val, clipped1), _mm_adds_epu16(d1, d1));
  auto c2 = _mm_adds_epu16(abs_diff_16(val, clipped2), _mm_adds_epu16(d2, d2));
  auto c3 = _mm_adds_epu16(abs_diff_16(val, clipped3), _mm_adds_epu16(d3, d3));
  auto c4 = _mm_adds_epu16(abs_diff_16(val, clipped4), _mm_adds_epu16(d4, d4));

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

  auto result = select_on_equal_16(mindiff, c1, val, clipped1);
  result = select_on_equal_16(mindiff, c3, result, clipped3);
  result = select_on_equal_16(mindiff, c2, result, clipped2);
  return select_on_equal_16(mindiff, c4, result, clipped4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode8_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(_mm_max_ps(a1, a8), c);
  auto mil1 = _mm_min_ps(_mm_min_ps(a1, a8), c);

  auto mal2 = _mm_max_ps(_mm_max_ps(a2, a7), c);
  auto mil2 = _mm_min_ps(_mm_min_ps(a2, a7), c);

  auto mal3 = _mm_max_ps(_mm_max_ps(a3, a6), c);
  auto mil3 = _mm_min_ps(_mm_min_ps(a3, a6), c);

  auto mal4 = _mm_max_ps(_mm_max_ps(a4, a5), c);
  auto mil4 = _mm_min_ps(_mm_min_ps(a4, a5), c);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto clipped1 = simd_clip_32(_mm_castsi128_ps(val), mil1, mal1);
  auto clipped2 = simd_clip_32(_mm_castsi128_ps(val), mil2, mal2);
  auto clipped3 = simd_clip_32(_mm_castsi128_ps(val), mil3, mal3);
  auto clipped4 = simd_clip_32(_mm_castsi128_ps(val), mil4, mal4);

  auto c1 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped1), _mm_add_ps(d1, d1)); // no adds needed only for comparison
  auto c2 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped2), _mm_add_ps(d2, d2));
  auto c3 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped3), _mm_add_ps(d3, d3));
  auto c4 = _mm_add_ps(abs_diff_32(_mm_castsi128_ps(val), clipped4), _mm_add_ps(d4, d4));

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val), clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}



// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode9_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
    auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

    auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
    auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

    auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
    auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

    auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
    auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);

    auto result = select_on_equal(mindiff, d1, val, simd_clip(val, mil1, mal1));
    result = select_on_equal(mindiff, d3, result, simd_clip(val, mil3, mal3));
    result = select_on_equal(mindiff, d2, result, simd_clip(val, mil2, mal2));
    return select_on_equal(mindiff, d4, result, simd_clip(val, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode9_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(_mm_max_epu8(a1, a8), c);
  auto mil1 = _mm_min_epu8(_mm_min_epu8(a1, a8), c);

  auto mal2 = _mm_max_epu8(_mm_max_epu8(a2, a7), c);
  auto mil2 = _mm_min_epu8(_mm_min_epu8(a2, a7), c);

  auto mal3 = _mm_max_epu8(_mm_max_epu8(a3, a6), c);
  auto mil3 = _mm_min_epu8(_mm_min_epu8(a3, a6), c);

  auto mal4 = _mm_max_epu8(_mm_max_epu8(a4, a5), c);
  auto mil4 = _mm_min_epu8(_mm_min_epu8(a4, a5), c);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);

  auto result = select_on_equal_sse4(mindiff, d1, val, simd_clip(val, mil1, mal1));
  result = select_on_equal_sse4(mindiff, d3, result, simd_clip(val, mil3, mal3));
  result = select_on_equal_sse4(mindiff, d2, result, simd_clip(val, mil2, mal2));
  return select_on_equal_sse4(mindiff, d4, result, simd_clip(val, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode9_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(_mm_max_epu16(a1, a8), c);
  auto mil1 = _mm_min_epu16(_mm_min_epu16(a1, a8), c);

  auto mal2 = _mm_max_epu16(_mm_max_epu16(a2, a7), c);
  auto mil2 = _mm_min_epu16(_mm_min_epu16(a2, a7), c);

  auto mal3 = _mm_max_epu16(_mm_max_epu16(a3, a6), c);
  auto mil3 = _mm_min_epu16(_mm_min_epu16(a3, a6), c);

  auto mal4 = _mm_max_epu16(_mm_max_epu16(a4, a5), c);
  auto mil4 = _mm_min_epu16(_mm_min_epu16(a4, a5), c);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);

  auto result = select_on_equal_16(mindiff, d1, val, simd_clip_16(val, mil1, mal1));
  result = select_on_equal_16(mindiff, d3, result, simd_clip_16(val, mil3, mal3));
  result = select_on_equal_16(mindiff, d2, result, simd_clip_16(val, mil2, mal2));
  return select_on_equal_16(mindiff, d4, result, simd_clip_16(val, mil4, mal4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode9_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(_mm_max_ps(a1, a8), c);
  auto mil1 = _mm_min_ps(_mm_min_ps(a1, a8), c);

  auto mal2 = _mm_max_ps(_mm_max_ps(a2, a7), c);
  auto mil2 = _mm_min_ps(_mm_min_ps(a2, a7), c);

  auto mal3 = _mm_max_ps(_mm_max_ps(a3, a6), c);
  auto mil3 = _mm_min_ps(_mm_min_ps(a3, a6), c);

  auto mal4 = _mm_max_ps(_mm_max_ps(a4, a5), c);
  auto mil4 = _mm_min_ps(_mm_min_ps(a4, a5), c);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);

  auto result = select_on_equal_32(mindiff, d1, _mm_castsi128_ps(val), simd_clip_32(_mm_castsi128_ps(val), mil1, mal1));
  result = select_on_equal_32(mindiff, d3, result, simd_clip_32(_mm_castsi128_ps(val), mil3, mal3));
  result = select_on_equal_32(mindiff, d2, result, simd_clip_32(_mm_castsi128_ps(val), mil2, mal2));
  return _mm_castps_si128(select_on_equal_32(mindiff, d4, result, simd_clip_32(_mm_castsi128_ps(val), mil4, mal4)));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode10_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto d1 = abs_diff(val, a1);
    auto d2 = abs_diff(val, a2);
    auto d3 = abs_diff(val, a3);
    auto d4 = abs_diff(val, a4);
    auto d5 = abs_diff(val, a5);
    auto d6 = abs_diff(val, a6);
    auto d7 = abs_diff(val, a7);
    auto d8 = abs_diff(val, a8);
    auto dc = abs_diff(val, c);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);
    mindiff = _mm_min_epu8(mindiff, d5);
    mindiff = _mm_min_epu8(mindiff, d6);
    mindiff = _mm_min_epu8(mindiff, d7);
    mindiff = _mm_min_epu8(mindiff, d8);
    mindiff = _mm_min_epu8(mindiff, dc);

    auto result = select_on_equal(mindiff, d4, c, a4);
    result = select_on_equal(mindiff, dc, result, c);
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
RG_FORCEINLINE __m128i repair_mode10_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff(val, a1);
  auto d2 = abs_diff(val, a2);
  auto d3 = abs_diff(val, a3);
  auto d4 = abs_diff(val, a4);
  auto d5 = abs_diff(val, a5);
  auto d6 = abs_diff(val, a6);
  auto d7 = abs_diff(val, a7);
  auto d8 = abs_diff(val, a8);
  auto dc = abs_diff(val, c);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);
  mindiff = _mm_min_epu8(mindiff, d5);
  mindiff = _mm_min_epu8(mindiff, d6);
  mindiff = _mm_min_epu8(mindiff, d7);
  mindiff = _mm_min_epu8(mindiff, d8);
  mindiff = _mm_min_epu8(mindiff, dc);

  auto result = select_on_equal_sse4(mindiff, d4, c, a4);
  result = select_on_equal_sse4(mindiff, dc, result, c);
  result = select_on_equal_sse4(mindiff, d5, result, a5);
  result = select_on_equal_sse4(mindiff, d1, result, a1);
  result = select_on_equal_sse4(mindiff, d3, result, a3);
  result = select_on_equal_sse4(mindiff, d2, result, a2);
  result = select_on_equal_sse4(mindiff, d6, result, a6);
  result = select_on_equal(mindiff, d8, result, a8);
  return select_on_equal_sse4(mindiff, d7, result, a7);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode10_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_16(val, a1);
  auto d2 = abs_diff_16(val, a2);
  auto d3 = abs_diff_16(val, a3);
  auto d4 = abs_diff_16(val, a4);
  auto d5 = abs_diff_16(val, a5);
  auto d6 = abs_diff_16(val, a6);
  auto d7 = abs_diff_16(val, a7);
  auto d8 = abs_diff_16(val, a8);
  auto dc = abs_diff_16(val, c);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);
  mindiff = _mm_min_epu16(mindiff, d5);
  mindiff = _mm_min_epu16(mindiff, d6);
  mindiff = _mm_min_epu16(mindiff, d7);
  mindiff = _mm_min_epu16(mindiff, d8);
  mindiff = _mm_min_epu16(mindiff, dc);

  auto result = select_on_equal_16(mindiff, d4, c, a4);
  result = select_on_equal_16(mindiff, dc, result, c);
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
RG_FORCEINLINE __m128i repair_mode10_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_32(_mm_castsi128_ps(val), a1);
  auto d2 = abs_diff_32(_mm_castsi128_ps(val), a2);
  auto d3 = abs_diff_32(_mm_castsi128_ps(val), a3);
  auto d4 = abs_diff_32(_mm_castsi128_ps(val), a4);
  auto d5 = abs_diff_32(_mm_castsi128_ps(val), a5);
  auto d6 = abs_diff_32(_mm_castsi128_ps(val), a6);
  auto d7 = abs_diff_32(_mm_castsi128_ps(val), a7);
  auto d8 = abs_diff_32(_mm_castsi128_ps(val), a8);
  auto dc = abs_diff_32(_mm_castsi128_ps(val), c);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);
  mindiff = _mm_min_ps(mindiff, d5);
  mindiff = _mm_min_ps(mindiff, d6);
  mindiff = _mm_min_ps(mindiff, d7);
  mindiff = _mm_min_ps(mindiff, d8);
  mindiff = _mm_min_ps(mindiff, dc);

  auto result = select_on_equal_32(mindiff, d4, c, a4);
  result = select_on_equal_32(mindiff, dc, result, c);
  result = select_on_equal_32(mindiff, d5, result, a5);
  result = select_on_equal_32(mindiff, d1, result, a1);
  result = select_on_equal_32(mindiff, d3, result, a3);
  result = select_on_equal_32(mindiff, d2, result, a2);
  result = select_on_equal_32(mindiff, d6, result, a6);
  result = select_on_equal_32(mindiff, d8, result, a8);
  return _mm_castps_si128(select_on_equal_32(mindiff, d7, result, a7));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode12_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
    sort_pair(a2, a6);
    sort_pair(a3, a7);
    a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

    a3 = _mm_min_epu8(a3, a5);	// sort_pair (a3, a5);
    a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);

    a2 = _mm_min_epu8(a2, a3);	// sort_pair (a2, a3);
    a7 = _mm_max_epu8(a6, a7);	// sort_pair (a6, a7);

    __m128i mi = _mm_min_epu8(c, a2);
    __m128i ma = _mm_max_epu8(c, a7);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode12_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  __m128i mi = _mm_min_epu8(c, a2);
  __m128i ma = _mm_max_epu8(c, a7);

  return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode12_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  sort_pair_16(a2, a6);
  sort_pair_16(a3, a7);
  a4 = _mm_min_epu16(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu16(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu16(a4, a6);	// sort_pair (a4, a6);

  a2 = _mm_min_epu16(a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_max_epu16(a6, a7);	// sort_pair (a6, a7);

  __m128i mi = _mm_min_epu16(c, a2);
  __m128i ma = _mm_max_epu16(c, a7);

  return simd_clip_16(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode12_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  sort_pair_32(a2, a6);
  sort_pair_32(a3, a7);
  a4 = _mm_min_ps(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_ps(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_ps(a4, a6);	// sort_pair (a4, a6);

  a2 = _mm_min_ps(a2, a3);	// sort_pair (a2, a3);
  a7 = _mm_max_ps(a6, a7);	// sort_pair (a6, a7);

  __m128 mi = _mm_min_ps(c, a2);
  __m128 ma = _mm_max_ps(c, a7);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode13_sse2(const Byte* pSrc, const __m128i& val, int srcPitch) {
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
  sort_pair(a2, a6);
  sort_pair(a3, a7);
  a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu8(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu8(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_epu8(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_epu8(a6, a7);	// sort_pair (a6, a7);

  __m128i mi = _mm_min_epu8(c, a3);
  __m128i ma = _mm_max_epu8(c, a6);

  return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode13_sse(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

    __m128i mi = _mm_min_epu8(c, a3);
    __m128i ma = _mm_max_epu8(c, a6);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode13_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  sort_pair_16(a2, a6);
  sort_pair_16(a3, a7);
  a4 = _mm_min_epu16(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_epu16(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_epu16(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_epu16(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_epu16(a6, a7);	// sort_pair (a6, a7);

  __m128i mi = _mm_min_epu16(c, a3);
  __m128i ma = _mm_max_epu16(c, a6);

  return simd_clip_16(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode13_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  sort_pair_32(a2, a6);
  sort_pair_32(a3, a7);
  a4 = _mm_min_ps(a4, a8);	// sort_pair (a4, a8);

  a3 = _mm_min_ps(a3, a5);	// sort_pair (a3, a5);
  a6 = _mm_max_ps(a4, a6);	// sort_pair (a4, a6);

  a3 = _mm_max_ps(a2, a3);	// sort_pair (a2, a3);
  a6 = _mm_min_ps(a6, a7);	// sort_pair (a6, a7);

  __m128 mi = _mm_min_ps(c, a3);
  __m128 ma = _mm_max_ps(c, a6);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode14_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
    a6 = _mm_max_epu8(a2, a6);	// sort_pair (a2, a6);
    a3 = _mm_min_epu8(a3, a7);	// sort_pair (a3, a7);
    a4 = _mm_min_epu8(a4, a8);	// sort_pair (a4, a8);

    a5 = _mm_max_epu8(a3, a5);	// sort_pair (a3, a5);
    a4 = _mm_min_epu8(a4, a6);	// sort_pair (a4, a6);

    sort_pair(a4, a5);

    __m128i mi = _mm_min_epu8(c, a4);
    __m128i ma = _mm_max_epu8(c, a5);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode14_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  sort_pair(a4, a5);

  __m128i mi = _mm_min_epu8(c, a4);
  __m128i ma = _mm_max_epu8(c, a5);

  return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode14_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  a6 = _mm_max_epu16(a2, a6);	// sort_pair (a2, a6);
  a3 = _mm_min_epu16(a3, a7);	// sort_pair (a3, a7);
  a4 = _mm_min_epu16(a4, a8);	// sort_pair (a4, a8);

  a5 = _mm_max_epu16(a3, a5);	// sort_pair (a3, a5);
  a4 = _mm_min_epu16(a4, a6);	// sort_pair (a4, a6);

  sort_pair_16(a4, a5);

  __m128i mi = _mm_min_epu16(c, a4);
  __m128i ma = _mm_max_epu16(c, a5);

  return simd_clip_16(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode14_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  a6 = _mm_max_ps(a2, a6);	// sort_pair (a2, a6);
  a3 = _mm_min_ps(a3, a7);	// sort_pair (a3, a7);
  a4 = _mm_min_ps(a4, a8);	// sort_pair (a4, a8);

  a5 = _mm_max_ps(a3, a5);	// sort_pair (a3, a5);
  a4 = _mm_min_ps(a4, a6);	// sort_pair (a4, a6);

  sort_pair_32(a4, a5);

  __m128 mi = _mm_min_ps(c, a4);
  __m128 ma = _mm_max_ps(c, a5);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode15_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto cma1 = _mm_max_epu8(c, mal1);
    auto cma2 = _mm_max_epu8(c, mal2);
    auto cma3 = _mm_max_epu8(c, mal3);
    auto cma4 = _mm_max_epu8(c, mal4);

    auto cmi1 = _mm_min_epu8(c, mil1);
    auto cmi2 = _mm_min_epu8(c, mil2);
    auto cmi3 = _mm_min_epu8(c, mil3);
    auto cmi4 = _mm_min_epu8(c, mil4);

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

    auto result = select_on_equal(mindiff, c1, val,    simd_clip(val, cmi1, cma1));
    result      = select_on_equal(mindiff, c3, result, simd_clip(val, cmi3, cma3));
    result      = select_on_equal(mindiff, c2, result, simd_clip(val, cmi2, cma2));
    return        select_on_equal(mindiff, c4, result, simd_clip(val, cmi4, cma4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode15_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto cma1 = _mm_max_epu8(c, mal1);
  auto cma2 = _mm_max_epu8(c, mal2);
  auto cma3 = _mm_max_epu8(c, mal3);
  auto cma4 = _mm_max_epu8(c, mal4);

  auto cmi1 = _mm_min_epu8(c, mil1);
  auto cmi2 = _mm_min_epu8(c, mil2);
  auto cmi3 = _mm_min_epu8(c, mil3);
  auto cmi4 = _mm_min_epu8(c, mil4);

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

  auto result = select_on_equal_sse4(mindiff, c1, val, simd_clip(val, cmi1, cma1));
  result = select_on_equal_sse4(mindiff, c3, result, simd_clip(val, cmi3, cma3));
  result = select_on_equal_sse4(mindiff, c2, result, simd_clip(val, cmi2, cma2));
  return select_on_equal_sse4(mindiff, c4, result, simd_clip(val, cmi4, cma4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode15_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto cma1 = _mm_max_epu16(c, mal1);
  auto cma2 = _mm_max_epu16(c, mal2);
  auto cma3 = _mm_max_epu16(c, mal3);
  auto cma4 = _mm_max_epu16(c, mal4);

  auto cmi1 = _mm_min_epu16(c, mil1);
  auto cmi2 = _mm_min_epu16(c, mil2);
  auto cmi3 = _mm_min_epu16(c, mil3);
  auto cmi4 = _mm_min_epu16(c, mil4);

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

  auto result = select_on_equal_16(mindiff, c1, val,    simd_clip_16(val, cmi1, cma1));
  result      = select_on_equal_16(mindiff, c3, result, simd_clip_16(val, cmi3, cma3));
  result      = select_on_equal_16(mindiff, c2, result, simd_clip_16(val, cmi2, cma2));
  return        select_on_equal_16(mindiff, c4, result, simd_clip_16(val, cmi4, cma4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode15_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto cma1 = _mm_max_ps(c, mal1);
  auto cma2 = _mm_max_ps(c, mal2);
  auto cma3 = _mm_max_ps(c, mal3);
  auto cma4 = _mm_max_ps(c, mal4);

  auto cmi1 = _mm_min_ps(c, mil1);
  auto cmi2 = _mm_min_ps(c, mil2);
  auto cmi3 = _mm_min_ps(c, mil3);
  auto cmi4 = _mm_min_ps(c, mil4);

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

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val),    simd_clip_32(_mm_castsi128_ps(val), cmi1, cma1));
  result      = select_on_equal_32(mindiff, c3, result, simd_clip_32(_mm_castsi128_ps(val), cmi3, cma3));
  result      = select_on_equal_32(mindiff, c2, result, simd_clip_32(_mm_castsi128_ps(val), cmi2, cma2));
  return        _mm_castps_si128(select_on_equal_32(mindiff, c4, result, simd_clip_32(_mm_castsi128_ps(val), cmi4, cma4)));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode16_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto cma1 = _mm_max_epu8(c, mal1);
    auto cma2 = _mm_max_epu8(c, mal2);
    auto cma3 = _mm_max_epu8(c, mal3);
    auto cma4 = _mm_max_epu8(c, mal4);

    auto cmi1 = _mm_min_epu8(c, mil1);
    auto cmi2 = _mm_min_epu8(c, mil2);
    auto cmi3 = _mm_min_epu8(c, mil3);
    auto cmi4 = _mm_min_epu8(c, mil4);

    auto clipped1 = simd_clip(c, mil1, mal1);
    auto clipped2 = simd_clip(c, mil2, mal2);
    auto clipped3 = simd_clip(c, mil3, mal3);
    auto clipped4 = simd_clip(c, mil4, mal4);

    auto d1 = _mm_subs_epu8(mal1, mil1);
    auto d2 = _mm_subs_epu8(mal2, mil2);
    auto d3 = _mm_subs_epu8(mal3, mil3);
    auto d4 = _mm_subs_epu8(mal4, mil4);

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

    auto result = select_on_equal(mindiff, c1, val,    simd_clip(val, cmi1, cma1));
    result      = select_on_equal(mindiff, c3, result, simd_clip(val, cmi3, cma3));
    result      = select_on_equal(mindiff, c2, result, simd_clip(val, cmi2, cma2));
    return        select_on_equal(mindiff, c4, result, simd_clip(val, cmi4, cma4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode16_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto cma1 = _mm_max_epu8(c, mal1);
  auto cma2 = _mm_max_epu8(c, mal2);
  auto cma3 = _mm_max_epu8(c, mal3);
  auto cma4 = _mm_max_epu8(c, mal4);

  auto cmi1 = _mm_min_epu8(c, mil1);
  auto cmi2 = _mm_min_epu8(c, mil2);
  auto cmi3 = _mm_min_epu8(c, mil3);
  auto cmi4 = _mm_min_epu8(c, mil4);

  auto clipped1 = simd_clip(c, mil1, mal1);
  auto clipped2 = simd_clip(c, mil2, mal2);
  auto clipped3 = simd_clip(c, mil3, mal3);
  auto clipped4 = simd_clip(c, mil4, mal4);

  auto d1 = _mm_subs_epu8(mal1, mil1);
  auto d2 = _mm_subs_epu8(mal2, mil2);
  auto d3 = _mm_subs_epu8(mal3, mil3);
  auto d4 = _mm_subs_epu8(mal4, mil4);

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

  auto result = select_on_equal_sse4(mindiff, c1, val, simd_clip(val, cmi1, cma1));
  result = select_on_equal_sse4(mindiff, c3, result, simd_clip(val, cmi3, cma3));
  result = select_on_equal_sse4(mindiff, c2, result, simd_clip(val, cmi2, cma2));
  return select_on_equal_sse4(mindiff, c4, result, simd_clip(val, cmi4, cma4));
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode16_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto cma1 = _mm_max_epu16(c, mal1);
  auto cma2 = _mm_max_epu16(c, mal2);
  auto cma3 = _mm_max_epu16(c, mal3);
  auto cma4 = _mm_max_epu16(c, mal4);

  auto cmi1 = _mm_min_epu16(c, mil1);
  auto cmi2 = _mm_min_epu16(c, mil2);
  auto cmi3 = _mm_min_epu16(c, mil3);
  auto cmi4 = _mm_min_epu16(c, mil4);

  auto clipped1 = simd_clip_16(c, mil1, mal1);
  auto clipped2 = simd_clip_16(c, mil2, mal2);
  auto clipped3 = simd_clip_16(c, mil3, mal3);
  auto clipped4 = simd_clip_16(c, mil4, mal4);

  auto d1 = _mm_subs_epu16(mal1, mil1);
  auto d2 = _mm_subs_epu16(mal2, mil2);
  auto d3 = _mm_subs_epu16(mal3, mil3);
  auto d4 = _mm_subs_epu16(mal4, mil4);

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

  auto result = select_on_equal_16(mindiff, c1, val,    simd_clip_16(val, cmi1, cma1));
  result      = select_on_equal_16(mindiff, c3, result, simd_clip_16(val, cmi3, cma3));
  result      = select_on_equal_16(mindiff, c2, result, simd_clip_16(val, cmi2, cma2));
  return        select_on_equal_16(mindiff, c4, result, simd_clip_16(val, cmi4, cma4));
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode16_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto cma1 = _mm_max_ps(c, mal1);
  auto cma2 = _mm_max_ps(c, mal2);
  auto cma3 = _mm_max_ps(c, mal3);
  auto cma4 = _mm_max_ps(c, mal4);

  auto cmi1 = _mm_min_ps(c, mil1);
  auto cmi2 = _mm_min_ps(c, mil2);
  auto cmi3 = _mm_min_ps(c, mil3);
  auto cmi4 = _mm_min_ps(c, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto d1 = _mm_subs_ps_for_diff(mal1, mil1);
  auto d2 = _mm_subs_ps_for_diff(mal2, mil2);
  auto d3 = _mm_subs_ps_for_diff(mal3, mil3);
  auto d4 = _mm_subs_ps_for_diff(mal4, mil4);

  auto absdiff1 = abs_diff_32(c, clipped1);
  auto absdiff2 = abs_diff_32(c, clipped2);
  auto absdiff3 = abs_diff_32(c, clipped3);
  auto absdiff4 = abs_diff_32(c, clipped4);

  auto c1 = _mm_add_ps(_mm_add_ps(absdiff1, absdiff1), d1); // no adds needed, only comparison
  auto c2 = _mm_add_ps(_mm_add_ps(absdiff2, absdiff2), d2);
  auto c3 = _mm_add_ps(_mm_add_ps(absdiff3, absdiff3), d3);
  auto c4 = _mm_add_ps(_mm_add_ps(absdiff4, absdiff4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, _mm_castsi128_ps(val),    simd_clip_32(_mm_castsi128_ps(val), cmi1, cma1));
  result      = select_on_equal_32(mindiff, c3, result, simd_clip_32(_mm_castsi128_ps(val), cmi3, cma3));
  result      = select_on_equal_32(mindiff, c2, result, simd_clip_32(_mm_castsi128_ps(val), cmi2, cma2));
  return        _mm_castps_si128(select_on_equal_32(mindiff, c4, result, simd_clip_32(_mm_castsi128_ps(val), cmi4, cma4)));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode17_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

    auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
    auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

    return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode17_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode17_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto real_upper = _mm_max_epu16(_mm_max_epu16(upper, lower), c);
  auto real_lower = _mm_min_epu16(_mm_min_epu16(upper, lower), c);

  return simd_clip_16(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode17_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto real_upper = _mm_max_ps(_mm_max_ps(upper, lower), c);
  auto real_lower = _mm_min_ps(_mm_min_ps(upper, lower), c);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), real_lower, real_upper));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode18_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

    auto mi1 = _mm_min_epu8(c, _mm_min_epu8(a1, a8));
    auto mi2 = _mm_min_epu8(c, _mm_min_epu8(a2, a7));
    auto mi3 = _mm_min_epu8(c, _mm_min_epu8(a3, a6));
    auto mi4 = _mm_min_epu8(c, _mm_min_epu8(a4, a5));

    auto ma1 = _mm_max_epu8(c, _mm_max_epu8(a1, a8));
    auto ma2 = _mm_max_epu8(c, _mm_max_epu8(a2, a7));
    auto ma3 = _mm_max_epu8(c, _mm_max_epu8(a3, a6));
    auto ma4 = _mm_max_epu8(c, _mm_max_epu8(a4, a5));

    __m128i c1 = simd_clip(val, mi1, ma1);
    __m128i c2 = simd_clip(val, mi2, ma2);
    __m128i c3 = simd_clip(val, mi3, ma3);
    __m128i c4 = simd_clip(val, mi4, ma4);

    auto result = select_on_equal(mindiff, d1, val, c1);
    result = select_on_equal(mindiff, d3, result, c3);
    result = select_on_equal(mindiff, d2, result, c2);
    return select_on_equal(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode18_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto mi1 = _mm_min_epu8(c, _mm_min_epu8(a1, a8));
  auto mi2 = _mm_min_epu8(c, _mm_min_epu8(a2, a7));
  auto mi3 = _mm_min_epu8(c, _mm_min_epu8(a3, a6));
  auto mi4 = _mm_min_epu8(c, _mm_min_epu8(a4, a5));

  auto ma1 = _mm_max_epu8(c, _mm_max_epu8(a1, a8));
  auto ma2 = _mm_max_epu8(c, _mm_max_epu8(a2, a7));
  auto ma3 = _mm_max_epu8(c, _mm_max_epu8(a3, a6));
  auto ma4 = _mm_max_epu8(c, _mm_max_epu8(a4, a5));

  __m128i c1 = simd_clip(val, mi1, ma1);
  __m128i c2 = simd_clip(val, mi2, ma2);
  __m128i c3 = simd_clip(val, mi3, ma3);
  __m128i c4 = simd_clip(val, mi4, ma4);

  auto result = select_on_equal_sse4(mindiff, d1, val, c1);
  result = select_on_equal_sse4(mindiff, d3, result, c3);
  result = select_on_equal_sse4(mindiff, d2, result, c2);
  return select_on_equal_sse4(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode18_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto mi1 = _mm_min_epu16(c, _mm_min_epu16(a1, a8));
  auto mi2 = _mm_min_epu16(c, _mm_min_epu16(a2, a7));
  auto mi3 = _mm_min_epu16(c, _mm_min_epu16(a3, a6));
  auto mi4 = _mm_min_epu16(c, _mm_min_epu16(a4, a5));

  auto ma1 = _mm_max_epu16(c, _mm_max_epu16(a1, a8));
  auto ma2 = _mm_max_epu16(c, _mm_max_epu16(a2, a7));
  auto ma3 = _mm_max_epu16(c, _mm_max_epu16(a3, a6));
  auto ma4 = _mm_max_epu16(c, _mm_max_epu16(a4, a5));

  __m128i c1 = simd_clip_16(val, mi1, ma1);
  __m128i c2 = simd_clip_16(val, mi2, ma2);
  __m128i c3 = simd_clip_16(val, mi3, ma3);
  __m128i c4 = simd_clip_16(val, mi4, ma4);

  auto result = select_on_equal_16(mindiff, d1, val, c1);
  result = select_on_equal_16(mindiff, d3, result, c3);
  result = select_on_equal_16(mindiff, d2, result, c2);
  return select_on_equal_16(mindiff, d4, result, c4);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode18_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto mi1 = _mm_min_ps(c, _mm_min_ps(a1, a8));
  auto mi2 = _mm_min_ps(c, _mm_min_ps(a2, a7));
  auto mi3 = _mm_min_ps(c, _mm_min_ps(a3, a6));
  auto mi4 = _mm_min_ps(c, _mm_min_ps(a4, a5));

  auto ma1 = _mm_max_ps(c, _mm_max_ps(a1, a8));
  auto ma2 = _mm_max_ps(c, _mm_max_ps(a2, a7));
  auto ma3 = _mm_max_ps(c, _mm_max_ps(a3, a6));
  auto ma4 = _mm_max_ps(c, _mm_max_ps(a4, a5));

  __m128 c1 = simd_clip_32(_mm_castsi128_ps(val), mi1, ma1);
  __m128 c2 = simd_clip_32(_mm_castsi128_ps(val), mi2, ma2);
  __m128 c3 = simd_clip_32(_mm_castsi128_ps(val), mi3, ma3);
  __m128 c4 = simd_clip_32(_mm_castsi128_ps(val), mi4, ma4);

  auto result = select_on_equal_32(mindiff, d1, _mm_castsi128_ps(val), c1);
  result = select_on_equal_32(mindiff, d3, result, c3);
  result = select_on_equal_32(mindiff, d2, result, c2);
  return _mm_castps_si128(select_on_equal_32(mindiff, d4, result, c4));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode19_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

    auto mi = _mm_subs_epu8(c, mindiff);
    auto ma = _mm_adds_epu8(c, mindiff);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode19_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto mi = _mm_subs_epu8(c, mindiff);
  auto ma = _mm_adds_epu8(c, mindiff);

  return simd_clip(val, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode19_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto mi = _mm_subs_epu16(c, mindiff);
  auto ma = _mm_adds_epu16(c, mindiff);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(val, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode19_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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

  auto mi = _mm_subs_ps<chroma>(c, mindiff);
  auto ma = _mm_adds_ps<chroma>(c, mindiff);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode20_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
    auto maxdiff = _mm_max_epu8(d1, d2);

    maxdiff = simd_clip(maxdiff, mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d3);

    maxdiff = simd_clip(maxdiff, mindiff, d4);
    mindiff = _mm_min_epu8(mindiff, d4);

    maxdiff = simd_clip(maxdiff, mindiff, d5);
    mindiff = _mm_min_epu8(mindiff, d5);

    maxdiff = simd_clip(maxdiff, mindiff, d6);
    mindiff = _mm_min_epu8(mindiff, d6);

    maxdiff = simd_clip(maxdiff, mindiff, d7);
    mindiff = _mm_min_epu8(mindiff, d7);

    maxdiff = simd_clip(maxdiff, mindiff, d8);

    auto mi = _mm_subs_epu8(c, maxdiff);
    auto ma = _mm_adds_epu8(c, maxdiff);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode20_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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
  auto maxdiff = _mm_max_epu8(d1, d2);

  maxdiff = simd_clip(maxdiff, mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d3);

  maxdiff = simd_clip(maxdiff, mindiff, d4);
  mindiff = _mm_min_epu8(mindiff, d4);

  maxdiff = simd_clip(maxdiff, mindiff, d5);
  mindiff = _mm_min_epu8(mindiff, d5);

  maxdiff = simd_clip(maxdiff, mindiff, d6);
  mindiff = _mm_min_epu8(mindiff, d6);

  maxdiff = simd_clip(maxdiff, mindiff, d7);
  mindiff = _mm_min_epu8(mindiff, d7);

  maxdiff = simd_clip(maxdiff, mindiff, d8);

  auto mi = _mm_subs_epu8(c, maxdiff);
  auto ma = _mm_adds_epu8(c, maxdiff);

  return simd_clip(val, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode20_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  auto maxdiff = _mm_max_epu16(d1, d2);

  maxdiff = simd_clip_16(maxdiff, mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d3);

  maxdiff = simd_clip_16(maxdiff, mindiff, d4);
  mindiff = _mm_min_epu16(mindiff, d4);

  maxdiff = simd_clip_16(maxdiff, mindiff, d5);
  mindiff = _mm_min_epu16(mindiff, d5);

  maxdiff = simd_clip_16(maxdiff, mindiff, d6);
  mindiff = _mm_min_epu16(mindiff, d6);

  maxdiff = simd_clip_16(maxdiff, mindiff, d7);
  mindiff = _mm_min_epu16(mindiff, d7);

  maxdiff = simd_clip_16(maxdiff, mindiff, d8);

  auto mi = _mm_subs_epu16(c, maxdiff);
  auto ma = _mm_adds_epu16(c, maxdiff);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(val, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode20_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
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
  auto maxdiff = _mm_max_ps(d1, d2);

  maxdiff = simd_clip_32(maxdiff, mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d3);

  maxdiff = simd_clip_32(maxdiff, mindiff, d4);
  mindiff = _mm_min_ps(mindiff, d4);

  maxdiff = simd_clip_32(maxdiff, mindiff, d5);
  mindiff = _mm_min_ps(mindiff, d5);

  maxdiff = simd_clip_32(maxdiff, mindiff, d6);
  mindiff = _mm_min_ps(mindiff, d6);

  maxdiff = simd_clip_32(maxdiff, mindiff, d7);
  mindiff = _mm_min_ps(mindiff, d7);

  maxdiff = simd_clip_32(maxdiff, mindiff, d8);

  auto mi = _mm_subs_ps<chroma>(c, maxdiff);
  auto ma = _mm_adds_ps<chroma>(c, maxdiff);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}

// ------------


template<bool aligned>
RG_FORCEINLINE __m128i repair_mode21_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, c);
    auto d2 = _mm_subs_epu8(mal2, c);
    auto d3 = _mm_subs_epu8(mal3, c);
    auto d4 = _mm_subs_epu8(mal4, c);

    auto rd1 = _mm_subs_epu8(c, mil1);
    auto rd2 = _mm_subs_epu8(c, mil2);
    auto rd3 = _mm_subs_epu8(c, mil3);
    auto rd4 = _mm_subs_epu8(c, mil4);

    auto u1 = _mm_max_epu8(d1, rd1);
    auto u2 = _mm_max_epu8(d2, rd2);
    auto u3 = _mm_max_epu8(d3, rd3);
    auto u4 = _mm_max_epu8(d4, rd4);

    auto u = _mm_min_epu8(u1, u2);
    u = _mm_min_epu8(u, u3);
    u = _mm_min_epu8(u, u4);

    auto mi = _mm_subs_epu8(c, u);
    auto ma = _mm_adds_epu8(c, u);

    return simd_clip(val, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode21_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, c);
  auto d2 = _mm_subs_epu8(mal2, c);
  auto d3 = _mm_subs_epu8(mal3, c);
  auto d4 = _mm_subs_epu8(mal4, c);

  auto rd1 = _mm_subs_epu8(c, mil1);
  auto rd2 = _mm_subs_epu8(c, mil2);
  auto rd3 = _mm_subs_epu8(c, mil3);
  auto rd4 = _mm_subs_epu8(c, mil4);

  auto u1 = _mm_max_epu8(d1, rd1);
  auto u2 = _mm_max_epu8(d2, rd2);
  auto u3 = _mm_max_epu8(d3, rd3);
  auto u4 = _mm_max_epu8(d4, rd4);

  auto u = _mm_min_epu8(u1, u2);
  u = _mm_min_epu8(u, u3);
  u = _mm_min_epu8(u, u4);

  auto mi = _mm_subs_epu8(c, u);
  auto ma = _mm_adds_epu8(c, u);

  return simd_clip(val, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode21_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, c);
  auto d2 = _mm_subs_epu16(mal2, c);
  auto d3 = _mm_subs_epu16(mal3, c);
  auto d4 = _mm_subs_epu16(mal4, c);

  auto rd1 = _mm_subs_epu16(c, mil1);
  auto rd2 = _mm_subs_epu16(c, mil2);
  auto rd3 = _mm_subs_epu16(c, mil3);
  auto rd4 = _mm_subs_epu16(c, mil4);

  auto u1 = _mm_max_epu16(d1, rd1);
  auto u2 = _mm_max_epu16(d2, rd2);
  auto u3 = _mm_max_epu16(d3, rd3);
  auto u4 = _mm_max_epu16(d4, rd4);

  auto u = _mm_min_epu16(u1, u2);
  u = _mm_min_epu16(u, u3);
  u = _mm_min_epu16(u, u4);

  auto mi = _mm_subs_epu16(c, u);
  auto ma = _mm_adds_epu16(c, u);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(val, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode21_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, c);
  auto d2 = _mm_subs_ps_for_diff(mal2, c);
  auto d3 = _mm_subs_ps_for_diff(mal3, c);
  auto d4 = _mm_subs_ps_for_diff(mal4, c);

  auto rd1 = _mm_subs_ps_for_diff(c, mil1);
  auto rd2 = _mm_subs_ps_for_diff(c, mil2);
  auto rd3 = _mm_subs_ps_for_diff(c, mil3);
  auto rd4 = _mm_subs_ps_for_diff(c, mil4);

  auto u1 = _mm_max_ps(d1, rd1);
  auto u2 = _mm_max_ps(d2, rd2);
  auto u3 = _mm_max_ps(d3, rd3);
  auto u4 = _mm_max_ps(d4, rd4);

  auto u = _mm_min_ps(u1, u2);
  u = _mm_min_ps(u, u3);
  u = _mm_min_ps(u, u4);

  auto mi = _mm_subs_ps<chroma>(c, u);
  auto ma = _mm_adds_ps<chroma>(c, u);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), mi, ma));
}

// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode22_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto d1 = abs_diff(val, a1);
    auto d2 = abs_diff(val, a2);
    auto d3 = abs_diff(val, a3);
    auto d4 = abs_diff(val, a4);
    auto d5 = abs_diff(val, a5);
    auto d6 = abs_diff(val, a6);
    auto d7 = abs_diff(val, a7);
    auto d8 = abs_diff(val, a8);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d4);
    mindiff = _mm_min_epu8(mindiff, d5);
    mindiff = _mm_min_epu8(mindiff, d6);
    mindiff = _mm_min_epu8(mindiff, d7);
    mindiff = _mm_min_epu8(mindiff, d8);

    auto mi = _mm_subs_epu8(val, mindiff);
    auto ma = _mm_adds_epu8(val, mindiff);

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode22_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff(val, a1);
  auto d2 = abs_diff(val, a2);
  auto d3 = abs_diff(val, a3);
  auto d4 = abs_diff(val, a4);
  auto d5 = abs_diff(val, a5);
  auto d6 = abs_diff(val, a6);
  auto d7 = abs_diff(val, a7);
  auto d8 = abs_diff(val, a8);

  auto mindiff = _mm_min_epu8(d1, d2);
  mindiff = _mm_min_epu8(mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d4);
  mindiff = _mm_min_epu8(mindiff, d5);
  mindiff = _mm_min_epu8(mindiff, d6);
  mindiff = _mm_min_epu8(mindiff, d7);
  mindiff = _mm_min_epu8(mindiff, d8);

  auto mi = _mm_subs_epu8(val, mindiff);
  auto ma = _mm_adds_epu8(val, mindiff);

  return simd_clip(c, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode22_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_16(val, a1);
  auto d2 = abs_diff_16(val, a2);
  auto d3 = abs_diff_16(val, a3);
  auto d4 = abs_diff_16(val, a4);
  auto d5 = abs_diff_16(val, a5);
  auto d6 = abs_diff_16(val, a6);
  auto d7 = abs_diff_16(val, a7);
  auto d8 = abs_diff_16(val, a8);

  auto mindiff = _mm_min_epu16(d1, d2);
  mindiff = _mm_min_epu16(mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d4);
  mindiff = _mm_min_epu16(mindiff, d5);
  mindiff = _mm_min_epu16(mindiff, d6);
  mindiff = _mm_min_epu16(mindiff, d7);
  mindiff = _mm_min_epu16(mindiff, d8);

  auto mi = _mm_subs_epu16(val, mindiff);
  auto ma = _mm_adds_epu16(val, mindiff);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(c, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode22_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_32(_mm_castsi128_ps(val), a1);
  auto d2 = abs_diff_32(_mm_castsi128_ps(val), a2);
  auto d3 = abs_diff_32(_mm_castsi128_ps(val), a3);
  auto d4 = abs_diff_32(_mm_castsi128_ps(val), a4);
  auto d5 = abs_diff_32(_mm_castsi128_ps(val), a5);
  auto d6 = abs_diff_32(_mm_castsi128_ps(val), a6);
  auto d7 = abs_diff_32(_mm_castsi128_ps(val), a7);
  auto d8 = abs_diff_32(_mm_castsi128_ps(val), a8);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);
  mindiff = _mm_min_ps(mindiff, d5);
  mindiff = _mm_min_ps(mindiff, d6);
  mindiff = _mm_min_ps(mindiff, d7);
  mindiff = _mm_min_ps(mindiff, d8);

  auto mi = _mm_subs_ps<chroma>(_mm_castsi128_ps(val), mindiff);
  auto ma = _mm_adds_ps<chroma>(_mm_castsi128_ps(val), mindiff);

  return _mm_castps_si128(simd_clip_32(c, mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode23_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto d1 = abs_diff(val, a1);
    auto d2 = abs_diff(val, a2);
    auto d3 = abs_diff(val, a3);
    auto d4 = abs_diff(val, a4);
    auto d5 = abs_diff(val, a5);
    auto d6 = abs_diff(val, a6);
    auto d7 = abs_diff(val, a7);
    auto d8 = abs_diff(val, a8);

    auto mindiff = _mm_min_epu8(d1, d2);
    auto maxdiff = _mm_max_epu8(d1, d2);

    maxdiff = simd_clip(maxdiff, mindiff, d3);
    mindiff = _mm_min_epu8(mindiff, d3);

    maxdiff = simd_clip(maxdiff, mindiff, d4);
    mindiff = _mm_min_epu8(mindiff, d4);

    maxdiff = simd_clip(maxdiff, mindiff, d5);
    mindiff = _mm_min_epu8(mindiff, d5);

    maxdiff = simd_clip(maxdiff, mindiff, d6);
    mindiff = _mm_min_epu8(mindiff, d6);

    maxdiff = simd_clip(maxdiff, mindiff, d7);
    mindiff = _mm_min_epu8(mindiff, d7);

    maxdiff = simd_clip(maxdiff, mindiff, d8);

    auto mi = _mm_subs_epu8(val, maxdiff);
    auto ma = _mm_adds_epu8(val, maxdiff);

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode23_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff(val, a1);
  auto d2 = abs_diff(val, a2);
  auto d3 = abs_diff(val, a3);
  auto d4 = abs_diff(val, a4);
  auto d5 = abs_diff(val, a5);
  auto d6 = abs_diff(val, a6);
  auto d7 = abs_diff(val, a7);
  auto d8 = abs_diff(val, a8);

  auto mindiff = _mm_min_epu8(d1, d2);
  auto maxdiff = _mm_max_epu8(d1, d2);

  maxdiff = simd_clip(maxdiff, mindiff, d3);
  mindiff = _mm_min_epu8(mindiff, d3);

  maxdiff = simd_clip(maxdiff, mindiff, d4);
  mindiff = _mm_min_epu8(mindiff, d4);

  maxdiff = simd_clip(maxdiff, mindiff, d5);
  mindiff = _mm_min_epu8(mindiff, d5);

  maxdiff = simd_clip(maxdiff, mindiff, d6);
  mindiff = _mm_min_epu8(mindiff, d6);

  maxdiff = simd_clip(maxdiff, mindiff, d7);
  mindiff = _mm_min_epu8(mindiff, d7);

  maxdiff = simd_clip(maxdiff, mindiff, d8);

  auto mi = _mm_subs_epu8(val, maxdiff);
  auto ma = _mm_adds_epu8(val, maxdiff);

  return simd_clip(c, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode23_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_16(val, a1);
  auto d2 = abs_diff_16(val, a2);
  auto d3 = abs_diff_16(val, a3);
  auto d4 = abs_diff_16(val, a4);
  auto d5 = abs_diff_16(val, a5);
  auto d6 = abs_diff_16(val, a6);
  auto d7 = abs_diff_16(val, a7);
  auto d8 = abs_diff_16(val, a8);

  auto mindiff = _mm_min_epu16(d1, d2);
  auto maxdiff = _mm_max_epu16(d1, d2);

  maxdiff = simd_clip_16(maxdiff, mindiff, d3);
  mindiff = _mm_min_epu16(mindiff, d3);

  maxdiff = simd_clip_16(maxdiff, mindiff, d4);
  mindiff = _mm_min_epu16(mindiff, d4);

  maxdiff = simd_clip_16(maxdiff, mindiff, d5);
  mindiff = _mm_min_epu16(mindiff, d5);

  maxdiff = simd_clip_16(maxdiff, mindiff, d6);
  mindiff = _mm_min_epu16(mindiff, d6);

  maxdiff = simd_clip_16(maxdiff, mindiff, d7);
  mindiff = _mm_min_epu16(mindiff, d7);

  maxdiff = simd_clip_16(maxdiff, mindiff, d8);

  auto mi = _mm_subs_epu16(val, maxdiff);
  auto ma = _mm_adds_epu16(val, maxdiff);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(c, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode23_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto d1 = abs_diff_32(_mm_castsi128_ps(val), a1);
  auto d2 = abs_diff_32(_mm_castsi128_ps(val), a2);
  auto d3 = abs_diff_32(_mm_castsi128_ps(val), a3);
  auto d4 = abs_diff_32(_mm_castsi128_ps(val), a4);
  auto d5 = abs_diff_32(_mm_castsi128_ps(val), a5);
  auto d6 = abs_diff_32(_mm_castsi128_ps(val), a6);
  auto d7 = abs_diff_32(_mm_castsi128_ps(val), a7);
  auto d8 = abs_diff_32(_mm_castsi128_ps(val), a8);

  auto mindiff = _mm_min_ps(d1, d2);
  auto maxdiff = _mm_max_ps(d1, d2);

  maxdiff = simd_clip_32(maxdiff, mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d3);

  maxdiff = simd_clip_32(maxdiff, mindiff, d4);
  mindiff = _mm_min_ps(mindiff, d4);

  maxdiff = simd_clip_32(maxdiff, mindiff, d5);
  mindiff = _mm_min_ps(mindiff, d5);

  maxdiff = simd_clip_32(maxdiff, mindiff, d6);
  mindiff = _mm_min_ps(mindiff, d6);

  maxdiff = simd_clip_32(maxdiff, mindiff, d7);
  mindiff = _mm_min_ps(mindiff, d7);

  maxdiff = simd_clip_32(maxdiff, mindiff, d8);

  auto mi = _mm_subs_ps<chroma>(_mm_castsi128_ps(val), maxdiff);
  auto ma = _mm_adds_ps<chroma>(_mm_castsi128_ps(val), maxdiff);

  return _mm_castps_si128(simd_clip_32(c, mi, ma));
}


// ------------

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode24_sse2(const Byte* pSrc, const __m128i &val, int srcPitch) {
    LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);

    auto mal1 = _mm_max_epu8(a1, a8);
    auto mil1 = _mm_min_epu8(a1, a8);

    auto mal2 = _mm_max_epu8(a2, a7);
    auto mil2 = _mm_min_epu8(a2, a7);

    auto mal3 = _mm_max_epu8(a3, a6);
    auto mil3 = _mm_min_epu8(a3, a6);

    auto mal4 = _mm_max_epu8(a4, a5);
    auto mil4 = _mm_min_epu8(a4, a5);

    auto d1 = _mm_subs_epu8(mal1, val);
    auto d2 = _mm_subs_epu8(mal2, val);
    auto d3 = _mm_subs_epu8(mal3, val);
    auto d4 = _mm_subs_epu8(mal4, val);

    auto rd1 = _mm_subs_epu8(val, mil1);
    auto rd2 = _mm_subs_epu8(val, mil2);
    auto rd3 = _mm_subs_epu8(val, mil3);
    auto rd4 = _mm_subs_epu8(val, mil4);

    auto u1 = _mm_max_epu8(d1, rd1);
    auto u2 = _mm_max_epu8(d2, rd2);
    auto u3 = _mm_max_epu8(d3, rd3);
    auto u4 = _mm_max_epu8(d4, rd4);

    auto u = _mm_min_epu8(u1, u2);
    u = _mm_min_epu8(u, u3);
    u = _mm_min_epu8(u, u4);

    auto mi = _mm_subs_epu8(val, u);
    auto ma = _mm_adds_epu8(val, u);

    return simd_clip(c, mi, ma);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode24_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE3_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu8(a1, a8);
  auto mil1 = _mm_min_epu8(a1, a8);

  auto mal2 = _mm_max_epu8(a2, a7);
  auto mil2 = _mm_min_epu8(a2, a7);

  auto mal3 = _mm_max_epu8(a3, a6);
  auto mil3 = _mm_min_epu8(a3, a6);

  auto mal4 = _mm_max_epu8(a4, a5);
  auto mil4 = _mm_min_epu8(a4, a5);

  auto d1 = _mm_subs_epu8(mal1, val);
  auto d2 = _mm_subs_epu8(mal2, val);
  auto d3 = _mm_subs_epu8(mal3, val);
  auto d4 = _mm_subs_epu8(mal4, val);

  auto rd1 = _mm_subs_epu8(val, mil1);
  auto rd2 = _mm_subs_epu8(val, mil2);
  auto rd3 = _mm_subs_epu8(val, mil3);
  auto rd4 = _mm_subs_epu8(val, mil4);

  auto u1 = _mm_max_epu8(d1, rd1);
  auto u2 = _mm_max_epu8(d2, rd2);
  auto u3 = _mm_max_epu8(d3, rd3);
  auto u4 = _mm_max_epu8(d4, rd4);

  auto u = _mm_min_epu8(u1, u2);
  u = _mm_min_epu8(u, u3);
  u = _mm_min_epu8(u, u4);

  auto mi = _mm_subs_epu8(val, u);
  auto ma = _mm_adds_epu8(val, u);

  return simd_clip(c, mi, ma);
}

template<int bits_per_pixel, bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode24_sse_16(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_epu16(a1, a8);
  auto mil1 = _mm_min_epu16(a1, a8);

  auto mal2 = _mm_max_epu16(a2, a7);
  auto mil2 = _mm_min_epu16(a2, a7);

  auto mal3 = _mm_max_epu16(a3, a6);
  auto mil3 = _mm_min_epu16(a3, a6);

  auto mal4 = _mm_max_epu16(a4, a5);
  auto mil4 = _mm_min_epu16(a4, a5);

  auto d1 = _mm_subs_epu16(mal1, val);
  auto d2 = _mm_subs_epu16(mal2, val);
  auto d3 = _mm_subs_epu16(mal3, val);
  auto d4 = _mm_subs_epu16(mal4, val);

  auto rd1 = _mm_subs_epu16(val, mil1);
  auto rd2 = _mm_subs_epu16(val, mil2);
  auto rd3 = _mm_subs_epu16(val, mil3);
  auto rd4 = _mm_subs_epu16(val, mil4);

  auto u1 = _mm_max_epu16(d1, rd1);
  auto u2 = _mm_max_epu16(d2, rd2);
  auto u3 = _mm_max_epu16(d3, rd3);
  auto u4 = _mm_max_epu16(d4, rd4);

  auto u = _mm_min_epu16(u1, u2);
  u = _mm_min_epu16(u, u3);
  u = _mm_min_epu16(u, u4);

  auto mi = _mm_subs_epu16(val, u);
  auto ma = _mm_adds_epu16(val, u);
  if (bits_per_pixel < 16) { // adds saturates to FFFF
    const __m128i pixel_max = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1));
    ma = _mm_min_epu16(ma, pixel_max);
  }

  return simd_clip_16(c, mi, ma);
}

template<bool aligned, bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode24_sse_32(const Byte* pSrc, const __m128i &val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps_for_diff(mal1, _mm_castsi128_ps(val));
  auto d2 = _mm_subs_ps_for_diff(mal2, _mm_castsi128_ps(val));
  auto d3 = _mm_subs_ps_for_diff(mal3, _mm_castsi128_ps(val));
  auto d4 = _mm_subs_ps_for_diff(mal4, _mm_castsi128_ps(val));

  auto rd1 = _mm_subs_ps_for_diff(_mm_castsi128_ps(val), mil1);
  auto rd2 = _mm_subs_ps_for_diff(_mm_castsi128_ps(val), mil2);
  auto rd3 = _mm_subs_ps_for_diff(_mm_castsi128_ps(val), mil3);
  auto rd4 = _mm_subs_ps_for_diff(_mm_castsi128_ps(val), mil4);

  auto u1 = _mm_max_ps(d1, rd1);
  auto u2 = _mm_max_ps(d2, rd2);
  auto u3 = _mm_max_ps(d3, rd3);
  auto u4 = _mm_max_ps(d4, rd4);

  auto u = _mm_min_ps(u1, u2);
  u = _mm_min_ps(u, u3);
  u = _mm_min_ps(u, u4);

  auto mi = _mm_subs_ps<chroma>(_mm_castsi128_ps(val), u);
  auto ma = _mm_adds_ps<chroma>(_mm_castsi128_ps(val), u);

  return _mm_castps_si128(simd_clip_32(c, mi, ma));
}

//----------------------------------
// mode 25 repair does not exist
//----------------------------------

// Mode26_SmartRGC.cpp
// 26 = medianblur.Based off mode 17, but preserves corners, but not thin lines.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode26_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}


template<bool aligned>
RG_FORCEINLINE __m128i repair_mode26_sse2(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode26_sse_16(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mi12 = _mm_min_epu16(a1, a2);
  auto ma12 = _mm_max_epu16(a1, a2);

  auto mi23 = _mm_min_epu16(a2, a3);
  auto ma23 = _mm_max_epu16(a2, a3);
  auto lower = _mm_max_epu16(mi12, mi23);
  auto upper = _mm_min_epu16(ma12, ma23);

  auto mi35 = _mm_min_epu16(a3, a5);
  auto ma35 = _mm_max_epu16(a3, a5);
  lower = _mm_max_epu16(lower, mi35);
  upper = _mm_min_epu16(upper, ma35);

  auto mi58 = _mm_min_epu16(a5, a8);
  auto ma58 = _mm_max_epu16(a5, a8);
  lower = _mm_max_epu16(lower, mi58);
  upper = _mm_min_epu16(upper, ma58);

  auto mi78 = _mm_min_epu16(a7, a8);
  auto ma78 = _mm_max_epu16(a7, a8);
  lower = _mm_max_epu16(lower, mi78);
  upper = _mm_min_epu16(upper, ma78);

  auto mi67 = _mm_min_epu16(a6, a7);
  auto ma67 = _mm_max_epu16(a6, a7);
  lower = _mm_max_epu16(lower, mi67);
  upper = _mm_min_epu16(upper, ma67);

  auto mi46 = _mm_min_epu16(a4, a6);
  auto ma46 = _mm_max_epu16(a4, a6);
  lower = _mm_max_epu16(lower, mi46);
  upper = _mm_min_epu16(upper, ma46);

  auto mi14 = _mm_min_epu16(a1, a4);
  auto ma14 = _mm_max_epu16(a1, a4);
  lower = _mm_max_epu16(lower, mi14);
  upper = _mm_min_epu16(upper, ma14);

  auto real_upper = _mm_max_epu16(_mm_max_epu16(upper, lower), c);
  auto real_lower = _mm_min_epu16(_mm_min_epu16(upper, lower), c);

  return simd_clip_16(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode26_sse_32(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mi12 = _mm_min_ps(a1, a2);
  auto ma12 = _mm_max_ps(a1, a2);

  auto mi23 = _mm_min_ps(a2, a3);
  auto ma23 = _mm_max_ps(a2, a3);
  auto lower = _mm_max_ps(mi12, mi23);
  auto upper = _mm_min_ps(ma12, ma23);

  auto mi35 = _mm_min_ps(a3, a5);
  auto ma35 = _mm_max_ps(a3, a5);
  lower = _mm_max_ps(lower, mi35);
  upper = _mm_min_ps(upper, ma35);

  auto mi58 = _mm_min_ps(a5, a8);
  auto ma58 = _mm_max_ps(a5, a8);
  lower = _mm_max_ps(lower, mi58);
  upper = _mm_min_ps(upper, ma58);

  auto mi78 = _mm_min_ps(a7, a8);
  auto ma78 = _mm_max_ps(a7, a8);
  lower = _mm_max_ps(lower, mi78);
  upper = _mm_min_ps(upper, ma78);

  auto mi67 = _mm_min_ps(a6, a7);
  auto ma67 = _mm_max_ps(a6, a7);
  lower = _mm_max_ps(lower, mi67);
  upper = _mm_min_ps(upper, ma67);

  auto mi46 = _mm_min_ps(a4, a6);
  auto ma46 = _mm_max_ps(a4, a6);
  lower = _mm_max_ps(lower, mi46);
  upper = _mm_min_ps(upper, ma46);

  auto mi14 = _mm_min_ps(a1, a4);
  auto ma14 = _mm_max_ps(a1, a4);
  lower = _mm_max_ps(lower, mi14);
  upper = _mm_min_ps(upper, ma14);

  auto real_upper = _mm_max_ps(_mm_max_ps(upper, lower), c);
  auto real_lower = _mm_min_ps(_mm_min_ps(upper, lower), c);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), real_lower, real_upper));
}


// Mode27_SmartRGCL.cpp
// 26 = medianblur.Based off mode 17, but preserves corners, but not thin lines.
// 27 = medianblur.Same as mode 26 but preserves thin lines.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode27_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode27_sse2(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode27_sse_16(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mi18 = _mm_min_epu16(a1, a8);
  auto ma18 = _mm_max_epu16(a1, a8);

  auto mi12 = _mm_min_epu16(a1, a2);
  auto ma12 = _mm_max_epu16(a1, a2);

  auto lower = _mm_max_epu16(mi18, mi12);
  auto upper = _mm_min_epu16(ma18, ma12);

  auto mi78 = _mm_min_epu16(a7, a8);
  auto ma78 = _mm_max_epu16(a7, a8);
  lower = _mm_max_epu16(lower, mi78);
  upper = _mm_min_epu16(upper, ma78);

  auto mi27 = _mm_min_epu16(a2, a7);
  auto ma27 = _mm_max_epu16(a2, a7);
  lower = _mm_max_epu16(lower, mi27);
  upper = _mm_min_epu16(upper, ma27);

  auto mi23 = _mm_min_epu16(a2, a3);
  auto ma23 = _mm_max_epu16(a2, a3);
  lower = _mm_max_epu16(lower, mi23);
  upper = _mm_min_epu16(upper, ma23);

  auto mi67 = _mm_min_epu16(a6, a7);
  auto ma67 = _mm_max_epu16(a6, a7);
  lower = _mm_max_epu16(lower, mi67);
  upper = _mm_min_epu16(upper, ma67);

  auto mi36 = _mm_min_epu16(a3, a6);
  auto ma36 = _mm_max_epu16(a3, a6);
  lower = _mm_max_epu16(lower, mi36);
  upper = _mm_min_epu16(upper, ma36);

  auto mi35 = _mm_min_epu16(a3, a5);
  auto ma35 = _mm_max_epu16(a3, a5);
  lower = _mm_max_epu16(lower, mi35);
  upper = _mm_min_epu16(upper, ma35);

  auto mi46 = _mm_min_epu16(a4, a6);
  auto ma46 = _mm_max_epu16(a4, a6);
  lower = _mm_max_epu16(lower, mi46);
  upper = _mm_min_epu16(upper, ma46);

  auto mi45 = _mm_min_epu16(a4, a5);
  auto ma45 = _mm_max_epu16(a4, a5);
  lower = _mm_max_epu16(lower, mi45);
  upper = _mm_min_epu16(upper, ma45);

  auto mi58 = _mm_min_epu16(a5, a8);
  auto ma58 = _mm_max_epu16(a5, a8);
  lower = _mm_max_epu16(lower, mi58);
  upper = _mm_min_epu16(upper, ma58);

  auto mi14 = _mm_min_epu16(a1, a4);
  auto ma14 = _mm_max_epu16(a1, a4);
  lower = _mm_max_epu16(lower, mi14);
  upper = _mm_min_epu16(upper, ma14);

  auto real_upper = _mm_max_epu16(_mm_max_epu16(upper, lower), c);
  auto real_lower = _mm_min_epu16(_mm_min_epu16(upper, lower), c);

  return simd_clip_16(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode27_sse_32(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mi18 = _mm_min_ps(a1, a8);
  auto ma18 = _mm_max_ps(a1, a8);

  auto mi12 = _mm_min_ps(a1, a2);
  auto ma12 = _mm_max_ps(a1, a2);

  auto lower = _mm_max_ps(mi18, mi12);
  auto upper = _mm_min_ps(ma18, ma12);

  auto mi78 = _mm_min_ps(a7, a8);
  auto ma78 = _mm_max_ps(a7, a8);
  lower = _mm_max_ps(lower, mi78);
  upper = _mm_min_ps(upper, ma78);

  auto mi27 = _mm_min_ps(a2, a7);
  auto ma27 = _mm_max_ps(a2, a7);
  lower = _mm_max_ps(lower, mi27);
  upper = _mm_min_ps(upper, ma27);

  auto mi23 = _mm_min_ps(a2, a3);
  auto ma23 = _mm_max_ps(a2, a3);
  lower = _mm_max_ps(lower, mi23);
  upper = _mm_min_ps(upper, ma23);

  auto mi67 = _mm_min_ps(a6, a7);
  auto ma67 = _mm_max_ps(a6, a7);
  lower = _mm_max_ps(lower, mi67);
  upper = _mm_min_ps(upper, ma67);

  auto mi36 = _mm_min_ps(a3, a6);
  auto ma36 = _mm_max_ps(a3, a6);
  lower = _mm_max_ps(lower, mi36);
  upper = _mm_min_ps(upper, ma36);

  auto mi35 = _mm_min_ps(a3, a5);
  auto ma35 = _mm_max_ps(a3, a5);
  lower = _mm_max_ps(lower, mi35);
  upper = _mm_min_ps(upper, ma35);

  auto mi46 = _mm_min_ps(a4, a6);
  auto ma46 = _mm_max_ps(a4, a6);
  lower = _mm_max_ps(lower, mi46);
  upper = _mm_min_ps(upper, ma46);

  auto mi45 = _mm_min_ps(a4, a5);
  auto ma45 = _mm_max_ps(a4, a5);
  lower = _mm_max_ps(lower, mi45);
  upper = _mm_min_ps(upper, ma45);

  auto mi58 = _mm_min_ps(a5, a8);
  auto ma58 = _mm_max_ps(a5, a8);
  lower = _mm_max_ps(lower, mi58);
  upper = _mm_min_ps(upper, ma58);

  auto mi14 = _mm_min_ps(a1, a4);
  auto ma14 = _mm_max_ps(a1, a4);
  lower = _mm_max_ps(lower, mi14);
  upper = _mm_min_ps(upper, ma14);

  auto real_upper = _mm_max_ps(_mm_max_ps(upper, lower), c);
  auto real_lower = _mm_min_ps(_mm_min_ps(upper, lower), c);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), real_lower, real_upper));
}


// Mode28_SmartRGCL2.cpp
// For my sources it gave identical result as mode 27, even if I made intentional
// errors in source.
template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode28_sse(const Byte* pSrc, const __m128i& val, int srcPitch) {
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
RG_FORCEINLINE __m128i repair_mode28_sse2(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_UA(pSrc, srcPitch, aligned);
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

  auto real_upper = _mm_max_epu8(_mm_max_epu8(upper, lower), c);
  auto real_lower = _mm_min_epu8(_mm_min_epu8(upper, lower), c);

  return simd_clip(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode28_sse_16(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_16_UA(pSrc, srcPitch, aligned);

  auto mi12 = _mm_min_epu16(a1, a2);
  auto ma12 = _mm_max_epu16(a1, a2);

  auto mi23 = _mm_min_epu16(a2, a3);
  auto ma23 = _mm_max_epu16(a2, a3);
  auto lower = _mm_max_epu16(mi12, mi23);
  auto upper = _mm_min_epu16(ma12, ma23);

  auto mi35 = _mm_min_epu16(a3, a5);
  auto ma35 = _mm_max_epu16(a3, a5);
  lower = _mm_max_epu16(lower, mi35);
  upper = _mm_min_epu16(upper, ma35);

  auto mi58 = _mm_min_epu16(a5, a8);
  auto ma58 = _mm_max_epu16(a5, a8);
  lower = _mm_max_epu16(lower, mi58);
  upper = _mm_min_epu16(upper, ma58);

  auto mi78 = _mm_min_epu16(a7, a8);
  auto ma78 = _mm_max_epu16(a7, a8);
  lower = _mm_max_epu16(lower, mi78);
  upper = _mm_min_epu16(upper, ma78);

  auto mi67 = _mm_min_epu16(a6, a7);
  auto ma67 = _mm_max_epu16(a6, a7);
  lower = _mm_max_epu16(lower, mi67);
  upper = _mm_min_epu16(upper, ma67);

  auto mi46 = _mm_min_epu16(a4, a6);
  auto ma46 = _mm_max_epu16(a4, a6);
  lower = _mm_max_epu16(lower, mi46);
  upper = _mm_min_epu16(upper, ma46);

  auto mi14 = _mm_min_epu16(a1, a4);
  auto ma14 = _mm_max_epu16(a1, a4);
  lower = _mm_max_epu16(lower, mi14);
  upper = _mm_min_epu16(upper, ma14);

  auto mi18 = _mm_min_epu16(a1, a8);
  auto ma18 = _mm_max_epu16(a1, a8);
  lower = _mm_max_epu16(lower, mi18);
  upper = _mm_min_epu16(upper, ma18);

  auto mi36 = _mm_min_epu16(a3, a6);
  auto ma36 = _mm_max_epu16(a3, a6);
  lower = _mm_max_epu16(lower, mi36);
  upper = _mm_min_epu16(upper, ma36);

  auto mi27 = _mm_min_epu16(a2, a7);
  auto ma27 = _mm_max_epu16(a2, a7);
  lower = _mm_max_epu16(lower, mi27);
  upper = _mm_min_epu16(upper, ma27);

  auto mi45 = _mm_min_epu16(a4, a5);
  auto ma45 = _mm_max_epu16(a4, a5);
  lower = _mm_max_epu16(lower, mi45);
  upper = _mm_min_epu16(upper, ma45);

  auto real_upper = _mm_max_epu16(_mm_max_epu16(upper, lower), c);
  auto real_lower = _mm_min_epu16(_mm_min_epu16(upper, lower), c);

  return simd_clip_16(val, real_lower, real_upper);
}

template<bool aligned>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i repair_mode28_sse_32(const Byte* pSrc, const __m128i& val, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mi12 = _mm_min_ps(a1, a2);
  auto ma12 = _mm_max_ps(a1, a2);

  auto mi23 = _mm_min_ps(a2, a3);
  auto ma23 = _mm_max_ps(a2, a3);
  auto lower = _mm_max_ps(mi12, mi23);
  auto upper = _mm_min_ps(ma12, ma23);

  auto mi35 = _mm_min_ps(a3, a5);
  auto ma35 = _mm_max_ps(a3, a5);
  lower = _mm_max_ps(lower, mi35);
  upper = _mm_min_ps(upper, ma35);

  auto mi58 = _mm_min_ps(a5, a8);
  auto ma58 = _mm_max_ps(a5, a8);
  lower = _mm_max_ps(lower, mi58);
  upper = _mm_min_ps(upper, ma58);

  auto mi78 = _mm_min_ps(a7, a8);
  auto ma78 = _mm_max_ps(a7, a8);
  lower = _mm_max_ps(lower, mi78);
  upper = _mm_min_ps(upper, ma78);

  auto mi67 = _mm_min_ps(a6, a7);
  auto ma67 = _mm_max_ps(a6, a7);
  lower = _mm_max_ps(lower, mi67);
  upper = _mm_min_ps(upper, ma67);

  auto mi46 = _mm_min_ps(a4, a6);
  auto ma46 = _mm_max_ps(a4, a6);
  lower = _mm_max_ps(lower, mi46);
  upper = _mm_min_ps(upper, ma46);

  auto mi14 = _mm_min_ps(a1, a4);
  auto ma14 = _mm_max_ps(a1, a4);
  lower = _mm_max_ps(lower, mi14);
  upper = _mm_min_ps(upper, ma14);

  auto mi18 = _mm_min_ps(a1, a8);
  auto ma18 = _mm_max_ps(a1, a8);
  lower = _mm_max_ps(lower, mi18);
  upper = _mm_min_ps(upper, ma18);

  auto mi36 = _mm_min_ps(a3, a6);
  auto ma36 = _mm_max_ps(a3, a6);
  lower = _mm_max_ps(lower, mi36);
  upper = _mm_min_ps(upper, ma36);

  auto mi27 = _mm_min_ps(a2, a7);
  auto ma27 = _mm_max_ps(a2, a7);
  lower = _mm_max_ps(lower, mi27);
  upper = _mm_min_ps(upper, ma27);

  auto mi45 = _mm_min_ps(a4, a5);
  auto ma45 = _mm_max_ps(a4, a5);
  lower = _mm_max_ps(lower, mi45);
  upper = _mm_min_ps(upper, ma45);

  auto real_upper = _mm_max_ps(_mm_max_ps(upper, lower), c);
  auto real_lower = _mm_min_ps(_mm_min_ps(upper, lower), c);

  return _mm_castps_si128(simd_clip_32(_mm_castsi128_ps(val), real_lower, real_upper));
}

#endif