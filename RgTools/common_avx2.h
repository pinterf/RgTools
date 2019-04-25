#ifndef __COMMON_AVX2_H__
#define __COMMON_AVX2_H__

#include <algorithm>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include <immintrin.h>
#include "common.h"

typedef unsigned char Byte;

#if defined(CLANG)
#define RG_FORCEINLINE __attribute__((always_inline)) inline 
#else
#define RG_FORCEINLINE __forceinline
#endif

/*
template<typename T>
static RG_FORCEINLINE Byte clip(T val, T minimum, T maximum) {
    return std::max(std::min(val, maximum), minimum);
}

// avs+
template<typename T>
static RG_FORCEINLINE uint16_t clip_16(T val, T minimum, T maximum) {
  return std::max(std::min(val, maximum), minimum);
}

// avs+
template<typename T>
static RG_FORCEINLINE float clip_32(T val, T minimum, T maximum) {
  return std::max(std::min(val, maximum), minimum);
}
*/

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

// avs+
static RG_FORCEINLINE __m256 abs_diff_32(__m256 a, __m256 b) {
  // maybe not optimal, mask may be generated 
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(~(1<<31))); // 0x7FFFFFFF
  return _mm256_and_ps(_mm256_sub_ps(a, b), absmask);
}

// PF until I find out better
static RG_FORCEINLINE __m256 _mm256_subs_ps(__m256 a, __m256 b) {
#if 0
  const __m256 zero = _mm256_setzero_ps();
  return _mm256_max_ps(_mm256_sub_ps(a, b), zero);
#else
  // no float clamp
  return _mm256_sub_ps(a, b);
#endif
}

// PF until I find out better
static RG_FORCEINLINE __m256 _mm256_adds_ps(__m256 a, __m256 b) {
#if 0
  const __m256 one = _mm256_set1_ps(1.0f);
  return _mm256_min_ps(_mm256_add_ps(a, b), one);
#else
  // no float clamp
  return _mm256_add_ps(a, b);
#endif
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
