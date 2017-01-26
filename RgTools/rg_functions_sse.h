#ifndef __RG_FUNCTIONS_SSE_H__
#define __RG_FUNCTIONS_SSE_H__

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
// - AVX/AVX2
// - uint16_t saturates to 65535, bit we have 10, 12 and 14 bot formats
// - todo: Float parts were blindly converted, should be optimized,
//         check averaging simplifications, clamping

//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode1_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode2_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode3_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode4_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode5_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode6_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<int bits_per_pixel, bool aligned>
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
RG_FORCEINLINE __m128i rg_mode6_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto zero = _mm_setzero_ps();

  auto mal1 = _mm_max_ps(a1, a8);
  auto mil1 = _mm_min_ps(a1, a8);

  auto mal2 = _mm_max_ps(a2, a7);
  auto mil2 = _mm_min_ps(a2, a7);

  auto mal3 = _mm_max_ps(a3, a6);
  auto mil3 = _mm_min_ps(a3, a6);

  auto mal4 = _mm_max_ps(a4, a5);
  auto mil4 = _mm_min_ps(a4, a5);

  auto d1 = _mm_subs_ps(mal1, mil1);
  auto d2 = _mm_subs_ps(mal2, mil2);
  auto d3 = _mm_subs_ps(mal3, mil3);
  auto d4 = _mm_subs_ps(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto absdiff1 = abs_diff_32(c, clipped1);
  auto absdiff2 = abs_diff_32(c, clipped2);
  auto absdiff3 = abs_diff_32(c, clipped3);
  auto absdiff4 = abs_diff_32(c, clipped4);

  auto c1 = _mm_adds_ps(_mm_adds_ps(absdiff1, absdiff1), d1);
  auto c2 = _mm_adds_ps(_mm_adds_ps(absdiff2, absdiff2), d2);
  auto c3 = _mm_adds_ps(_mm_adds_ps(absdiff3, absdiff3), d3);
  auto c4 = _mm_adds_ps(_mm_adds_ps(absdiff4, absdiff4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}


//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode7_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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
  //todo: what happens when this overflows?
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

  auto d1 = _mm_subs_ps(mal1, mil1);
  auto d2 = _mm_subs_ps(mal2, mil2);
  auto d3 = _mm_subs_ps(mal3, mil3);
  auto d4 = _mm_subs_ps(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);
  //todo: what happens when this overflows?
  auto c1 = _mm_adds_ps(abs_diff_32(c, clipped1), d1);
  auto c2 = _mm_adds_ps(abs_diff_32(c, clipped2), d2);
  auto c3 = _mm_adds_ps(abs_diff_32(c, clipped3), d3);
  auto c4 = _mm_adds_ps(abs_diff_32(c, clipped4), d4);

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}

//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode8_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<int bits_per_pixel, bool aligned>
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

  auto d1 = _mm_subs_ps(mal1, mil1);
  auto d2 = _mm_subs_ps(mal2, mil2);
  auto d3 = _mm_subs_ps(mal3, mil3);
  auto d4 = _mm_subs_ps(mal4, mil4);

  auto clipped1 = simd_clip_32(c, mil1, mal1);
  auto clipped2 = simd_clip_32(c, mil2, mal2);
  auto clipped3 = simd_clip_32(c, mil3, mal3);
  auto clipped4 = simd_clip_32(c, mil4, mal4);

  auto c1 = _mm_adds_ps(abs_diff_32(c, clipped1), _mm_adds_ps(d1, d1));
  auto c2 = _mm_adds_ps(abs_diff_32(c, clipped2), _mm_adds_ps(d2, d2));
  auto c3 = _mm_adds_ps(abs_diff_32(c, clipped3), _mm_adds_ps(d3, d3));
  auto c4 = _mm_adds_ps(abs_diff_32(c, clipped4), _mm_adds_ps(d4, d4));

  auto mindiff = _mm_min_ps(c1, c2);
  mindiff = _mm_min_ps(mindiff, c3);
  mindiff = _mm_min_ps(mindiff, c4);

  auto result = select_on_equal_32(mindiff, c1, c, clipped1);
  result = select_on_equal_32(mindiff, c3, result, clipped3);
  result = select_on_equal_32(mindiff, c2, result, clipped2);
  return _mm_castps_si128(select_on_equal_32(mindiff, c4, result, clipped4));
}

//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode9_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

  auto d1 = _mm_subs_ps(mal1, mil1);
  auto d2 = _mm_subs_ps(mal2, mil2);
  auto d3 = _mm_subs_ps(mal3, mil3);
  auto d4 = _mm_subs_ps(mal4, mil4);

  auto mindiff = _mm_min_ps(d1, d2);
  mindiff = _mm_min_ps(mindiff, d3);
  mindiff = _mm_min_ps(mindiff, d4);

  auto result = select_on_equal_32(mindiff, d1, c, simd_clip_32(c, mil1, mal1));
  result = select_on_equal_32(mindiff, d3, result, simd_clip_32(c, mil3, mal3));
  result = select_on_equal_32(mindiff, d2, result, simd_clip_32(c, mil2, mal2));
  return _mm_castps_si128(select_on_equal_32(mindiff, d4, result, simd_clip_32(c, mil4, mal4)));
}


//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode10_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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
RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch);
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode12_sse_16(const Byte* pSrc, int srcPitch);
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode12_sse_32(const Byte* pSrc, int srcPitch);

//todo: actually implement is as mode 11
template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode11_sse(const Byte* pSrc, int srcPitch) {
    return rg_mode12_sse<aligned, optLevel>(pSrc, srcPitch);
}
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode11_sse_16(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse_16<aligned>(pSrc, srcPitch);
}
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode11_sse_32(const Byte* pSrc, int srcPitch) {
  return rg_mode12_sse_32<aligned>(pSrc, srcPitch);
}

//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode13_and14_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

//rounding does not match
template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode15_and16_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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
RG_FORCEINLINE __m128i rg_mode15_and16_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto max18 = _mm_max_ps(a1, a8);
  auto min18 = _mm_min_ps(a1, a8);

  auto max27 = _mm_max_ps(a2, a7);
  auto min27 = _mm_min_ps(a2, a7);

  auto max36 = _mm_max_ps(a3, a6);
  auto min36 = _mm_min_ps(a3, a6);

  auto d1 = _mm_subs_ps(max18, min18);
  auto d2 = _mm_subs_ps(max27, min27);
  auto d3 = _mm_subs_ps(max36, min36);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode17_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode18_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode19_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

//todo: probably extract a function with 12 arguments?
template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode20_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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
  const byte FACTOR = 15;
  auto zero = _mm_setzero_si128();
  auto onenineth = _mm_set1_epi32((unsigned short)(((1u << FACTOR) + 4) / 9));
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

  auto sum = _mm_adds_epu16(sum_t1, sum_t2);
  sum = _mm_adds_epu16(sum, cunpck_lo);
  sum = _mm_adds_epu16(sum, bias);

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

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode21_sse(const Byte* pSrc, int srcPitch) {
  /*
  LOAD_SQUARE_SSE_UA_18(optLevel, pSrc, srcPitch, aligned);

  auto l1a = not_rounded_average(a1, a8);
  auto l1b = _mm_avg_epu8(a1, a8);

  LOAD_SQUARE_SSE_UA_27(optLevel, pSrc, srcPitch, aligned);
  auto l2a = not_rounded_average(a2, a7);
  auto l2b = _mm_avg_epu8(a2, a7);

  LOAD_SQUARE_SSE_UA_36(optLevel, pSrc, srcPitch, aligned);
  auto l3a = not_rounded_average(a3, a6);
  auto l3b = _mm_avg_epu8(a3, a6);

  LOAD_SQUARE_SSE_UA_45(optLevel, pSrc, srcPitch, aligned);
  auto l4a = not_rounded_average(a4, a5);
  auto l4b = _mm_avg_epu8(a4, a5);

  auto ma = _mm_max_epu8(l1b, l2b);
  ma = _mm_max_epu8(ma, l3b);
  ma = _mm_max_epu8(ma, l4b);

  auto mi = _mm_min_epu8(l1a, l2a);
  mi = _mm_min_epu8(mi, l3a);
  mi = _mm_min_epu8(mi, l4a);

  LOAD_SQUARE_SSE_UA_Cent(optLevel, pSrc, srcPitch, aligned);
  return simd_clip(c, mi, ma);
*/
  
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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
RG_FORCEINLINE __m128i rg_mode22_sse_32(const Byte* pSrc, int srcPitch);
  // float: no integer tricks, same like 22
template<bool aligned>
RG_FORCEINLINE __m128i rg_mode21_sse_32(const Byte* pSrc, int srcPitch) {
  return rg_mode22_sse_32<aligned>(pSrc, srcPitch);
}

//-------------------

template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode22_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

//optimized. 
template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode23_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned>
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

  auto linediff1 = _mm_subs_ps(mal1, mil1);
  auto linediff2 = _mm_subs_ps(mal2, mil2);
  auto linediff3 = _mm_subs_ps(mal3, mil3);
  auto linediff4 = _mm_subs_ps(mal4, mil4);

  auto u1 = _mm_min_ps(_mm_subs_ps(c, mal1), linediff1);
  auto u2 = _mm_min_ps(_mm_subs_ps(c, mal2), linediff2);
  auto u3 = _mm_min_ps(_mm_subs_ps(c, mal3), linediff3);
  auto u4 = _mm_min_ps(_mm_subs_ps(c, mal4), linediff4);

  auto u = _mm_max_ps(u1, u2);
  u = _mm_max_ps(u, u3);
  u = _mm_max_ps(u, u4);

  auto d1 = _mm_min_ps(_mm_subs_ps(mil1, c), linediff1);
  auto d2 = _mm_min_ps(_mm_subs_ps(mil2, c), linediff2);
  auto d3 = _mm_min_ps(_mm_subs_ps(mil3, c), linediff3);
  auto d4 = _mm_min_ps(_mm_subs_ps(mil4, c), linediff4);

  auto d = _mm_max_ps(d1, d2);
  d = _mm_max_ps(d, d3);
  d = _mm_max_ps(d, d4);

  return _mm_castps_si128(_mm_adds_ps(_mm_subs_ps(c, u), d));
}


//-------------------

//optimized, todo: decide how to name the function and extract this stuff. Order is important.
template<bool aligned, InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode24_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE_UA(optLevel, pSrc, srcPitch, aligned);

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

template<bool aligned>
RG_FORCEINLINE __m128i rg_mode24_sse_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_SSE_32_UA(pSrc, srcPitch, aligned);

  auto mal  = _mm_max_ps(a1, a8);
  auto mil  = _mm_min_ps(a1, a8);
  auto diff = _mm_subs_ps(mal, mil);
  auto temp = _mm_subs_ps(c, mal);
  auto u1   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));
  temp      = _mm_subs_ps(mil, c);
  auto d1   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));

  mal       = _mm_max_ps(a2, a7);
  mil       = _mm_min_ps(a2, a7);
  diff      = _mm_subs_ps(mal, mil);
  temp      = _mm_subs_ps(c, mal);
  auto u2   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));
  temp      = _mm_subs_ps(mil, c);
  auto d2   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));

  auto d = _mm_max_ps(d1, d2);
  auto u = _mm_max_ps(u1, u2);

  mal       = _mm_max_ps(a3, a6);
  mil       = _mm_min_ps(a3, a6);
  diff      = _mm_subs_ps(mal, mil);
  temp      = _mm_subs_ps(c, mal);
  auto u3   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));
  temp      = _mm_subs_ps(mil, c);
  auto d3   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));

  d = _mm_max_ps(d, d3);
  u = _mm_max_ps(u, u3);

  mal       = _mm_max_ps(a4, a5);
  mil       = _mm_min_ps(a4, a5);
  diff      = _mm_subs_ps(mal, mil);
  temp      = _mm_subs_ps(c, mal);
  auto u4   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));
  temp      = _mm_subs_ps(mil, c);
  auto d4   = _mm_min_ps(temp, _mm_subs_ps(diff, temp));

  d = _mm_max_ps(d, d4);
  u = _mm_max_ps(u, u4);

  return _mm_castps_si128(_mm_adds_ps(_mm_subs_ps(c, u), d));
}


#endif