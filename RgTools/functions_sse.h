#ifndef __FUNCTIONS_SSE_H__
#define __FUNCTIONS_SSE_H__

#include "common.h"
#include <smmintrin.h>

typedef __m128i (SseModeProcessor)(const Byte*, int);

#define LOAD_SQUARE_SSE(optLevel, ptr, pitch) \
    __m128i a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - 1); \
    __m128i a2 = simd_loadu_si128<optLevel>((ptr) - (pitch)); \
    __m128i a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + 1); \
    __m128i a4 = simd_loadu_si128<optLevel>((ptr) - 1); \
    __m128i c  = simd_loadu_si128<optLevel>((ptr) ); \
    __m128i a5 = simd_loadu_si128<optLevel>((ptr) + 1); \
    __m128i a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - 1); \
    __m128i a7 = simd_loadu_si128<optLevel>((ptr) + (pitch)); \
    __m128i a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + 1); 



static RG_FORCEINLINE void sort_pair(__m128i &a1, __m128i &a2)
{
    const __m128i tmp = _mm_min_epu8 (a1, a2);
    a2 = _mm_max_epu8 (a1, a2);
    a1 = tmp;
}

template<InstructionSet optLevel>
static RG_FORCEINLINE __m128i simd_loadu_si128(const Byte* ptr) {
    if (optLevel == SSE2) {
#ifdef USE_MOVPS
        return _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<const float*>(ptr)));
#else
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
    }
    return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(ptr));
}

static RG_FORCEINLINE __m128i simd_clip(__m128i &val, __m128i &minimum, __m128i &maximum) {
    return _mm_max_epu8(_mm_min_epu8(val, maximum), minimum);
}

//mask ? a : b
static RG_FORCEINLINE __m128i blend(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
   //return  _mm_blendv_epi8 (otherwise, desired, mask);
    auto andop = _mm_and_si128(mask , desired);
    auto andnop = _mm_andnot_si128(mask, otherwise);
    return _mm_or_si128(andop, andnop);
}

static RG_FORCEINLINE __m128i abs_diff(__m128i a, __m128i b) {
    auto positive = _mm_subs_epu8(a, b);
    auto negative = _mm_subs_epu8(b, a);
    return _mm_or_si128(positive, negative);
}

static RG_FORCEINLINE __m128i select_on_equal(__m128i &cmp1, __m128i &cmp2, __m128i &current, __m128i &desired) {
    auto eq = _mm_cmpeq_epi8(cmp1, cmp2);
    return blend(eq, desired, current);
}

//(x&y)+((x^y)/2) for (a+b)/2
static RG_FORCEINLINE __m128i not_rounded_average(__m128i a, __m128i b) {
    auto andop = _mm_and_si128(a, b);
    auto xorop = _mm_xor_si128(a, b);
    //kinda psrlb, probably not optimal but works
    xorop = _mm_srli_epi16(xorop, 1);
    xorop = _mm_and_si128(xorop, _mm_set1_epi8(0x7F));
    return _mm_adds_epu8(xorop, andop);
}


RG_FORCEINLINE __m128i rg_nothing_sse(const Byte*, int) {
    return _mm_set1_epi32(0xBAD);
}


template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode1_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode2_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode3_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode4_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode5_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode6_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode7_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode8_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode9_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode10_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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


RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch);
//todo: actually implement is as mode 11
template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode11_sse(const Byte* pSrc, int srcPitch) {
    return rg_mode12_sse<optLevel>(pSrc, srcPitch);
}

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode12_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode13_and14_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

    auto d1 = abs_diff(a1, a8);
    auto d2 = abs_diff(a2, a7);
    auto d3 = abs_diff(a3, a6);

    auto mindiff = _mm_min_epu8(d1, d2);
    mindiff = _mm_min_epu8(mindiff, d3);

    auto result = select_on_equal(mindiff, d1, c, _mm_avg_epu8(a1, a8));
    result = select_on_equal(mindiff, d3, result,  _mm_avg_epu8(a3, a6));
    return select_on_equal(mindiff, d2, result,  _mm_avg_epu8(a2, a7));
}

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode17_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode18_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode19_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode21_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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


template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode22_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

//optimized. 
template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode23_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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

//optimized, todo: decide how to name the function and extract this stuff. Order is important.
template<InstructionSet optLevel>
RG_FORCEINLINE __m128i rg_mode24_sse(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_SSE(optLevel, pSrc, srcPitch);

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


#endif