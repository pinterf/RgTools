#ifndef __COMMON_AVX2_H__
#define __COMMON_AVX2_H__

#include <algorithm>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include "avs/config.h"

// experimental simd includes for avx2 compiled files
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__))  && ! defined (__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__ 


#include "common.h"

typedef unsigned char Byte;

static RG_FORCEINLINE bool is_32byte_aligned(const void *ptr) {
    return (((uintptr_t)ptr) & 31) == 0;
}

static RG_FORCEINLINE __m256i simd_clip(const __m256i &val, const __m256i &minimum, const __m256i &maximum) {
  return _mm256_max_epu8(_mm256_min_epu8(val, maximum), minimum);
}

// SSE4!
// PF avs+
static RG_FORCEINLINE __m256i simd_clip_16(const __m256i &val, const __m256i &minimum, const __m256i &maximum) {
  return _mm256_max_epu16(_mm256_min_epu16(val, maximum), minimum); // SSE4
}

// PF avs+
static RG_FORCEINLINE __m256 simd_clip_32(const __m256 &val, const __m256 &minimum, const __m256 &maximum) {
  return _mm256_max_ps(_mm256_min_ps(val, maximum), minimum); 
}



static RG_FORCEINLINE void sort_pair(__m256i &a1, __m256i &a2)
{
  const __m256i tmp = _mm256_min_epu8 (a1, a2);
  a2 = _mm256_max_epu8 (a1, a2);
  a1 = tmp;
}

// SSE4
// PF avs+
static RG_FORCEINLINE void sort_pair_16(__m256i &a1, __m256i &a2)
{
  const __m256i tmp = _mm256_min_epu16 (a1, a2);
  a2 = _mm256_max_epu16 (a1, a2);
  a1 = tmp;
}

// PF avs+
static RG_FORCEINLINE void sort_pair_32(__m256 &a1, __m256 &a2)
{
  const __m256 tmp = _mm256_min_ps (a1, a2);
  a2 = _mm256_max_ps (a1, a2);
  a1 = tmp;
}

static RG_FORCEINLINE __m256i simd_loadu_si256(const Byte* ptr) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
}

static RG_FORCEINLINE __m256i simd_loada_si256(const Byte* ptr) {
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
}

//mask ? a : b
static RG_FORCEINLINE __m256i blend(__m256i const &mask, __m256i const &desired, __m256i const &otherwise) {
  return  _mm256_blendv_epi8 (otherwise, desired, mask);
}

static RG_FORCEINLINE __m256i blend_16(__m256i const &mask, __m256i const &desired, __m256i const &otherwise) {
  return  _mm256_blendv_epi8 (otherwise, desired, mask); // no need for epi 16 here
}

static RG_FORCEINLINE __m256 blend_32(__m256 const &mask, __m256 const &desired, __m256 const &otherwise) {
  return  _mm256_blendv_ps (otherwise, desired, mask);
}


static RG_FORCEINLINE __m256i abs_diff(__m256i a, __m256i b) {
  auto positive = _mm256_subs_epu8(a, b);
  auto negative = _mm256_subs_epu8(b, a);
  return _mm256_or_si256(positive, negative);
}

static RG_FORCEINLINE __m256i abs_diff_16(__m256i a, __m256i b) {
  auto positive = _mm256_subs_epu16(a, b);
  auto negative = _mm256_subs_epu16(b, a);
  return _mm256_or_si256(positive, negative);
}

static RG_FORCEINLINE __m256 abs_diff_32(__m256 a, __m256 b) {
  // maybe not optimal, mask may be generated 
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(~(1<<31))); // 0x7FFFFFFF
  return _mm256_and_ps(_mm256_sub_ps(a, b), absmask);
}

template<bool chroma>
static RG_FORCEINLINE __m256 _mm256_subs_ps(__m256 a, __m256 b) {
  const __m256 pixel_min = chroma ? _mm256_set1_ps(-0.5f) : _mm256_set1_ps(0.0f);
  return _mm256_max_ps(_mm256_sub_ps(a, b), pixel_min);
}

// for use case: diff = pixel1 - pixel2, saturates to zero
static RG_FORCEINLINE __m256 _mm256_subs_ps_for_diff(__m256 a, __m256 b) {
  const __m256 pixel_min = _mm256_set1_ps(0.0f);
  return _mm256_max_ps(_mm256_sub_ps(a, b), pixel_min);
}

template<bool chroma>
static RG_FORCEINLINE __m256 _mm256_adds_ps(__m256 a, __m256 b) {
  const __m256 pixel_max = chroma ? _mm256_set1_ps(0.5f) : _mm256_set1_ps(1.0f);
  return _mm256_min_ps(_mm256_add_ps(a, b), pixel_max);
}

static RG_FORCEINLINE __m256 _mm256_adds_ps_for_diff(__m256 a, __m256 b) {
  const __m256 pixel_max = _mm256_set1_ps(1.0f);
  return _mm256_min_ps(_mm256_add_ps(a, b), pixel_max);
}


// PF until I find out better
static RG_FORCEINLINE __m256 _mm256_avg_ps(__m256 a, __m256 b) {
  const __m256 div2 = _mm256_set1_ps(0.5f);
  return _mm256_mul_ps(_mm256_add_ps(a, b), div2);
}

static RG_FORCEINLINE __m256i select_on_equal(const __m256i &cmp1, const __m256i &cmp2, const __m256i &current, const __m256i &desired) {
  auto eq = _mm256_cmpeq_epi8(cmp1, cmp2);
  return blend(eq, desired, current);
}

static RG_FORCEINLINE __m256i select_on_equal_16(const __m256i &cmp1, const __m256i &cmp2, const __m256i &current, const __m256i &desired) {
  auto eq = _mm256_cmpeq_epi16(cmp1, cmp2);
  return blend_16(eq, desired, current);
}

static RG_FORCEINLINE __m256  _mm256_cmpeq_ps(__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }

static RG_FORCEINLINE __m256 select_on_equal_32(const __m256 &cmp1, const __m256 &cmp2, const __m256 &current, const __m256 &desired) {
  auto eq = _mm256_cmpeq_ps(cmp1, cmp2);
  return blend_32(eq, desired, current);
}

// sharpen, neighbourdiff: helpers for mode 25
// base on common.h sse routines

static RG_FORCEINLINE __m256i _MM256_SRLI_EPI8(const __m256i& v, int imm) {
  return _mm256_and_si256(_mm256_set1_epi8(0xFF >> imm), _mm256_srli_epi32(v, imm));
}

static RG_FORCEINLINE __m256i sharpen_avx2(const __m256i& center, const __m256i& minus, const __m256i& plus) {
  auto mp_diff = _mm256_subs_epu8(minus, plus);
  auto pm_diff = _mm256_subs_epu8(plus, minus);
  auto m_per2 = _MM256_SRLI_EPI8(minus, 1);
  auto p_per2 = _MM256_SRLI_EPI8(plus, 1);
  auto min_1 = _mm256_min_epu8(p_per2, mp_diff);
  auto min_2 = _mm256_min_epu8(m_per2, pm_diff);
  return _mm256_subs_epu8(_mm256_adds_epu8(center, min_1), min_2);
}

template<int bits_per_pixel>
static RG_FORCEINLINE __m256i sharpen_avx2_16(const __m256i& center, const __m256i& minus, const __m256i& plus) {
  auto mp_diff = _mm256_subs_epu16(minus, plus);
  auto pm_diff = _mm256_subs_epu16(plus, minus);
  auto m_per2 = _mm256_srli_epi16(minus, 1);
  auto p_per2 = _mm256_srli_epi16(plus, 1);
  auto min_1 = _mm256_min_epu16(p_per2, mp_diff);
  auto min_2 = _mm256_min_epu16(m_per2, pm_diff);
  auto result1 = _mm256_adds_epu16(center, min_1);
  if constexpr (bits_per_pixel < 16) {
    const auto max_pixel_value = _mm256_set1_epi16((1 << bits_per_pixel) - 1);
    result1 = _mm256_min_epi16(result1, max_pixel_value);
  }
  auto result = _mm256_subs_epu16(result1, min_2);
  return result;
}

template<bool chroma>
static RG_FORCEINLINE __m256 sharpen_avx2_32(const __m256& center, const __m256& minus, const __m256& plus) {
  auto mp_diff = _mm256_subs_ps_for_diff(minus, plus); // for diff: like luma (template false) 0..1
  auto pm_diff = _mm256_subs_ps_for_diff(plus, minus);
  const auto half = _mm256_set1_ps(0.5f);
  auto m_per2 = _mm256_mul_ps(minus, half);
  auto p_per2 = _mm256_mul_ps(plus, half);
  auto min_1 = _mm256_min_ps(p_per2, mp_diff);
  auto min_2 = _mm256_min_ps(m_per2, pm_diff);
  return  _mm256_subs_ps<chroma>(_mm256_adds_ps<chroma>(center, min_1), min_2);
}


// helper for mode 25
static RG_FORCEINLINE void neighbourdiff_avx2(__m256i& minus, __m256i& plus, __m256i center, __m256i neighbour, const __m256i& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  auto cn_diff = _mm256_subs_epu8(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm256_subs_epu8(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm256_cmpeq_epi8(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm256_cmpeq_epi8(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm256_and_si256(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  minus = _mm256_or_si256(cn_diff, cndiff_masked); // 5 FF FF FF
  plus = _mm256_or_si256(nc_diff, ncdiff_masked);  // FF 1  FF 254

  minus = _mm256_subs_epu8(minus, cn_equal); // 5 FF 00 FF 
  plus = _mm256_subs_epu8(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

static RG_FORCEINLINE void neighbourdiff_avx2_16(__m256i& minus, __m256i& plus, __m256i center, __m256i neighbour, const __m256i& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  // no bits_per_pixel template, mask FFFF is big enough for sharpen to subs saturate
  auto cn_diff = _mm256_subs_epu16(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm256_subs_epu16(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm256_cmpeq_epi16(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm256_cmpeq_epi16(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm256_and_si256(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  minus = _mm256_or_si256(cn_diff, cndiff_masked); // 5 FF FF FF
  plus = _mm256_or_si256(nc_diff, ncdiff_masked);  // FF 1  FF 254

  minus = _mm256_subs_epu16(minus, cn_equal); // 5 FF 00 FF 
  plus = _mm256_subs_epu16(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

static RG_FORCEINLINE void neighbourdiff_avx2_32(__m256& minus, __m256& plus, __m256 center, __m256 neighbour, const __m256& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  // FF in integer, max_mask=1.0 in float
  // for diffs: subs must clamp at 0
  auto cn_diff = _mm256_subs_ps_for_diff(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm256_subs_ps_for_diff(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm256_cmpeq_ps(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm256_cmpeq_ps(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm256_and_ps(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  // max_mask where <=
  // min_mask (0) where =
  const auto max_mask = _mm256_set1_ps(1.0f);
  // a, b, mask. If mask then b else a
  minus = _mm256_blendv_ps(cn_diff, max_mask, cndiff_masked); // 5 FF FF FF
  plus = _mm256_blendv_ps(nc_diff, max_mask, ncdiff_masked);  // FF 1  FF 254

  minus = _mm256_blendv_ps(minus, zero, cn_equal); // 5 FF 00 FF 
  plus = _mm256_blendv_ps(plus, zero, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

// end of mode 25 helpers

#define LOAD_SQUARE_AVX2_0(ptr, pitch, pixelsize, aligned) \
__m256i a1, a2, a3, a4, a5, a6, a7, a8, c; \
if constexpr(!aligned) {\
a1 = simd_loadu_si256((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loadu_si256((ptr) - (pitch)); \
a3 = simd_loadu_si256((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si256((ptr) - (pixelsize)); \
c  = simd_loadu_si256((ptr) ); \
a5 = simd_loadu_si256((ptr) + (pixelsize)); \
a6 = simd_loadu_si256((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loadu_si256((ptr) + (pitch)); \
a8 = simd_loadu_si256((ptr) + (pitch) + (pixelsize)); \
} else {\
a1 = simd_loadu_si256((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loada_si256((ptr) - (pitch)); \
a3 = simd_loadu_si256((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si256((ptr) - (pixelsize)); \
c  = simd_loada_si256((ptr) ); \
a5 = simd_loadu_si256((ptr) + (pixelsize)); \
a6 = simd_loadu_si256((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loada_si256((ptr) + (pitch)); \
a8 = simd_loadu_si256((ptr) + (pitch) + (pixelsize)); \
}

// 8 bit loads
// unaligned or aligned
#define LOAD_SQUARE_AVX2_UA(ptr, pitch, aligned) LOAD_SQUARE_AVX2_0(ptr, pitch, 1, aligned)

// 16 bit loads
// unaligned or aligned
#define LOAD_SQUARE_AVX2_16_UA(ptr, pitch, aligned) LOAD_SQUARE_AVX2_0(ptr, pitch, 2, aligned)

// 32 bit float loads
#define LOAD_SQUARE_AVX2_0_32(ptr, pitch, aligned) \
__m256 a1, a2, a3, a4, a5, a6, a7, a8, c; \
if constexpr(!aligned) {\
a1 = _mm256_loadu_ps((const float *)((ptr) - (pitch) - 4)); \
a2 = _mm256_loadu_ps((const float *)((ptr) - (pitch))); \
a3 = _mm256_loadu_ps((const float *)((ptr) - (pitch) + (4))); \
a4 = _mm256_loadu_ps((const float *)((ptr) - (4))); \
c  = _mm256_loadu_ps((const float *)((ptr) )); \
a5 = _mm256_loadu_ps((const float *)((ptr) + (4))); \
a6 = _mm256_loadu_ps((const float *)((ptr) + (pitch) - (4))); \
a7 = _mm256_loadu_ps((const float *)((ptr) + (pitch))); \
a8 = _mm256_loadu_ps((const float *)((ptr) + (pitch) + (4))); \
} else { \
a1 = _mm256_loadu_ps((const float *)((ptr) - (pitch) - 4)); \
a2 = _mm256_load_ps((const float *)((ptr) - (pitch))); \
a3 = _mm256_loadu_ps((const float *)((ptr) - (pitch) + (4))); \
a4 = _mm256_loadu_ps((const float *)((ptr) - (4))); \
c  = _mm256_load_ps((const float *)((ptr) )); \
a5 = _mm256_loadu_ps((const float *)((ptr) + (4))); \
a6 = _mm256_loadu_ps((const float *)((ptr) + (pitch) - (4))); \
a7 = _mm256_load_ps((const float *)((ptr) + (pitch))); \
a8 = _mm256_loadu_ps((const float *)((ptr) + (pitch) + (4))); \
}

// unaligned
#define LOAD_SQUARE_AVX2_32(ptr, pitch) LOAD_SQUARE_AVX2_0_32(ptr, pitch, false) 
// unaligned or aligned
#define LOAD_SQUARE_AVX2_32_UA(ptr, pitch, aligned) LOAD_SQUARE_AVX2_0_32(ptr, pitch, aligned)

#endif
