#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdint.h"
#include <algorithm>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4

typedef uint8_t Byte;

#if defined(CLANG)
#define RG_FORCEINLINE __attribute__((always_inline)) inline 
#else
#define RG_FORCEINLINE __forceinline
#endif


#define USE_MOVPS

template<typename T>
static RG_FORCEINLINE Byte clip(T val, T minimum, T maximum) {
    return std::max(std::min(val, maximum), minimum);
}

template<typename T>
static RG_FORCEINLINE uint16_t clip_16(T val, T minimum, T maximum) {
  return std::max(std::min(val, maximum), minimum);
}

template<typename T>
static RG_FORCEINLINE float clip_32(T val, T minimum, T maximum) {
  return std::max(std::min(val, maximum), minimum);
}

static RG_FORCEINLINE bool is_16byte_aligned(const void *ptr) {
    return (((uintptr_t)ptr) & 15) == 0;
}

static RG_FORCEINLINE __m128i simd_clip(const __m128i &val, const __m128i &minimum, const __m128i &maximum) {
  return _mm_max_epu8(_mm_min_epu8(val, maximum), minimum);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i simd_clip_16(const __m128i &val, const __m128i &minimum, const __m128i &maximum) {
  return _mm_max_epu16(_mm_min_epu16(val, maximum), minimum); // SSE4
}

static RG_FORCEINLINE __m128 simd_clip_32(const __m128 &val, const __m128 &minimum, const __m128 &maximum) {
  return _mm_max_ps(_mm_min_ps(val, maximum), minimum); 
}

static RG_FORCEINLINE void sort_pair(__m128i &a1, __m128i &a2)
{
  const __m128i tmp = _mm_min_epu8 (a1, a2);
  a2 = _mm_max_epu8 (a1, a2);
  a1 = tmp;
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE void sort_pair_16(__m128i &a1, __m128i &a2)
{
  const __m128i tmp = _mm_min_epu16 (a1, a2);
  a2 = _mm_max_epu16 (a1, a2);
  a1 = tmp;
}

static RG_FORCEINLINE void sort_pair_32(__m128 &a1, __m128 &a2)
{
  const __m128 tmp = _mm_min_ps (a1, a2);
  a2 = _mm_max_ps (a1, a2);
  a1 = tmp;
}

static RG_FORCEINLINE __m128i simd_loadu_si128(const Byte* ptr) {
#ifdef USE_MOVPS
  return _mm_castps_si128(_mm_loadu_ps(reinterpret_cast<const float*>(ptr)));
#else
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
}

// _mm_lddqu_si128: still faster on i7-3770 vs _mm_loadu_si128
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse3")))
#endif
static RG_FORCEINLINE __m128i simd_loadu_si128_sse3(const Byte* ptr) {
  return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(ptr));
}

static RG_FORCEINLINE __m128i simd_loada_si128(const Byte* ptr) {
#ifdef USE_MOVPS
    return _mm_castps_si128(_mm_load_ps(reinterpret_cast<const float*>(ptr)));
#else
    return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
}

//mask ? a : b
static RG_FORCEINLINE __m128i blend(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
  //return  _mm_blendv_epi8 (otherwise, desired, mask);
  auto andop = _mm_and_si128(mask , desired);
  auto andnop = _mm_andnot_si128(mask, otherwise);
  return _mm_or_si128(andop, andnop);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i blend_sse4(__m128i const& mask, __m128i const& desired, __m128i const& otherwise) {
  return  _mm_blendv_epi8(otherwise, desired, mask);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i blend_16(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
  return  _mm_blendv_epi8 (otherwise, desired, mask);
}

// sse4
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128 blend_32(__m128 const &mask, __m128 const &desired, __m128 const &otherwise) {
  return  _mm_blendv_ps (otherwise, desired, mask);
}

static RG_FORCEINLINE __m128i abs_diff(__m128i a, __m128i b) {
  auto positive = _mm_subs_epu8(a, b);
  auto negative = _mm_subs_epu8(b, a);
  return _mm_or_si128(positive, negative);
}

static RG_FORCEINLINE __m128i abs_diff_16(__m128i a, __m128i b) {
  auto positive = _mm_subs_epu16(a, b);
  auto negative = _mm_subs_epu16(b, a);
  return _mm_or_si128(positive, negative);
}

static RG_FORCEINLINE __m128 abs_diff_32(__m128 a, __m128 b) {
  // maybe not optimal, mask may be generated 
  const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1<<31))); // 0x7FFFFFFF
  return _mm_and_ps(_mm_sub_ps(a, b), absmask);
}

// PF until I find out better
// RemoveGraintT has artifacts w/o proper clamping
static RG_FORCEINLINE __m128 _mm_subs_ps(__m128 a, __m128 b) {
#if 0
const __m128 zero = _mm_setzero_ps();
return _mm_max_ps(_mm_sub_ps(a, b), zero);
#else
  // no float clamp
  return _mm_sub_ps(a, b);
#endif
}

// PF until I find out better
static RG_FORCEINLINE __m128 _mm_adds_ps(__m128 a, __m128 b) {
#if 0
  const __m128 one = _mm_set1_ps(1.0f);
  return _mm_min_ps(_mm_add_ps(a, b), one);
#else
  // no float clamp
  return _mm_add_ps(a, b);
#endif
}

// PF until I find out better
static RG_FORCEINLINE __m128 _mm_avg_ps(__m128 a, __m128 b) {
  const __m128 div2 = _mm_set1_ps(0.5f);
  return _mm_mul_ps(_mm_add_ps(a, b), div2);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i select_on_equal_sse4(const __m128i &cmp1, const __m128i &cmp2, const __m128i &current, const __m128i &desired) {
  auto eq = _mm_cmpeq_epi8(cmp1, cmp2);
  return blend_sse4(eq, desired, current);
}

static RG_FORCEINLINE __m128i select_on_equal(const __m128i& cmp1, const __m128i& cmp2, const __m128i& current, const __m128i& desired) {
  auto eq = _mm_cmpeq_epi8(cmp1, cmp2);
  return blend(eq, desired, current);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i select_on_equal_16(const __m128i &cmp1, const __m128i &cmp2, const __m128i &current, const __m128i &desired) {
  auto eq = _mm_cmpeq_epi16(cmp1, cmp2);
  return blend_16(eq, desired, current);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128 select_on_equal_32(const __m128 &cmp1, const __m128 &cmp2, const __m128 &current, const __m128 &desired) {
  auto eq = _mm_cmpeq_ps(cmp1, cmp2);
  return blend_32(eq, desired, current);
}

// center column as aligned
#define LOAD_SQUARE_SSE_0(ptr, pitch, pixelsize, aligned) \
__m128i a1, a2, a3, a4, a5, a6, a7, a8, c; \
if constexpr(!aligned) {\
a1 = simd_loadu_si128((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loadu_si128((ptr) - (pitch)); \
a3 = simd_loadu_si128((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128((ptr) - (pixelsize)); \
c  = simd_loadu_si128((ptr) ); \
a5 = simd_loadu_si128((ptr) + (pixelsize)); \
a6 = simd_loadu_si128((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loadu_si128((ptr) + (pitch)); \
a8 = simd_loadu_si128((ptr) + (pitch) + (pixelsize)); \
} else {\
a1 = simd_loadu_si128((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loada_si128((ptr) - (pitch)); \
a3 = simd_loadu_si128((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128((ptr) - (pixelsize)); \
c  = simd_loada_si128((ptr) ); \
a5 = simd_loadu_si128((ptr) + (pixelsize)); \
a6 = simd_loadu_si128((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loada_si128((ptr) + (pitch)); \
a8 = simd_loadu_si128((ptr) + (pitch) + (pixelsize)); \
}

#define LOAD_SQUARE_SSE3_0(ptr, pitch, pixelsize, aligned) \
__m128i a1, a2, a3, a4, a5, a6, a7, a8, c; \
if constexpr(!aligned) {\
a1 = simd_loadu_si128_sse3((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loadu_si128_sse3((ptr) - (pitch)); \
a3 = simd_loadu_si128_sse3((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128_sse3((ptr) - (pixelsize)); \
c  = simd_loadu_si128_sse3((ptr) ); \
a5 = simd_loadu_si128_sse3((ptr) + (pixelsize)); \
a6 = simd_loadu_si128_sse3((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loadu_si128_sse3((ptr) + (pitch)); \
a8 = simd_loadu_si128_sse3((ptr) + (pitch) + (pixelsize)); \
} else {\
a1 = simd_loadu_si128_sse3((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loada_si128((ptr) - (pitch)); \
a3 = simd_loadu_si128_sse3((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128_sse3((ptr) - (pixelsize)); \
c  = simd_loada_si128((ptr) ); \
a5 = simd_loadu_si128_sse3((ptr) + (pixelsize)); \
a6 = simd_loadu_si128_sse3((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loada_si128((ptr) + (pitch)); \
a8 = simd_loadu_si128_sse3((ptr) + (pitch) + (pixelsize)); \
}

// 8 bit loads
// unaligned or aligned
#define LOAD_SQUARE_SSE_UA(ptr, pitch, aligned) LOAD_SQUARE_SSE_0(ptr, pitch, 1, aligned)
#define LOAD_SQUARE_SSE3_UA(ptr, pitch, aligned) LOAD_SQUARE_SSE3_0(ptr, pitch, 1, aligned)

// 16 bit loads
// unaligned or aligned
#define LOAD_SQUARE_SSE_16_UA(ptr, pitch, aligned) LOAD_SQUARE_SSE3_0(ptr, pitch, 2, aligned)

// 32 bit float loads
#define LOAD_SQUARE_SSE_0_32(ptr, pitch, aligned) \
__m128 a1, a2, a3, a4, a5, a6, a7, a8, c; \
if constexpr(!aligned) {\
a1 = _mm_loadu_ps((const float *)((ptr) - (pitch) - 4)); \
a2 = _mm_loadu_ps((const float *)((ptr) - (pitch))); \
a3 = _mm_loadu_ps((const float *)((ptr) - (pitch) + (4))); \
a4 = _mm_loadu_ps((const float *)((ptr) - (4))); \
c  = _mm_loadu_ps((const float *)((ptr) )); \
a5 = _mm_loadu_ps((const float *)((ptr) + (4))); \
a6 = _mm_loadu_ps((const float *)((ptr) + (pitch) - (4))); \
a7 = _mm_loadu_ps((const float *)((ptr) + (pitch))); \
a8 = _mm_loadu_ps((const float *)((ptr) + (pitch) + (4))); \
} else { \
a1 = _mm_loadu_ps((const float *)((ptr) - (pitch) - 4)); \
a2 = _mm_load_ps((const float *)((ptr) - (pitch))); \
a3 = _mm_loadu_ps((const float *)((ptr) - (pitch) + (4))); \
a4 = _mm_loadu_ps((const float *)((ptr) - (4))); \
c  = _mm_load_ps((const float *)((ptr) )); \
a5 = _mm_loadu_ps((const float *)((ptr) + (4))); \
a6 = _mm_loadu_ps((const float *)((ptr) + (pitch) - (4))); \
a7 = _mm_load_ps((const float *)((ptr) + (pitch))); \
a8 = _mm_loadu_ps((const float *)((ptr) + (pitch) + (4))); \
}

// unaligned or aligned
#define LOAD_SQUARE_SSE_32_UA(ptr, pitch, aligned) LOAD_SQUARE_SSE_0_32(ptr, pitch, aligned)

// loaders for C routines
// pointers and pitch are byte-based
#define LOAD_SQUARE_CPP_0(pixel_t, ptr, pitch) \
    pixel_t a1 = *(pixel_t *)((ptr) - (pitch) - sizeof(pixel_t)); \
    pixel_t a2 = *(pixel_t *)((ptr) - (pitch)); \
    pixel_t a3 = *(pixel_t *)((ptr) - (pitch) + sizeof(pixel_t)); \
    pixel_t a4 = *(pixel_t *)((ptr) - sizeof(pixel_t)); \
    pixel_t c  = *(pixel_t *)((ptr) ); \
    pixel_t a5 = *(pixel_t *)((ptr) + sizeof(pixel_t)); \
    pixel_t a6 = *(pixel_t *)((ptr) + (pitch) - sizeof(pixel_t)); \
    pixel_t a7 = *(pixel_t *)((ptr) + (pitch)); \
    pixel_t a8 = *(pixel_t *)((ptr) + (pitch) + sizeof(pixel_t));

#define LOAD_SQUARE_CPP(ptr, pitch) LOAD_SQUARE_CPP_0(Byte, ptr, pitch);
#define LOAD_SQUARE_CPP_16(ptr, pitch) LOAD_SQUARE_CPP_0(uint16_t, ptr, pitch);
#define LOAD_SQUARE_CPP_32(ptr, pitch) LOAD_SQUARE_CPP_0(float, ptr, pitch);

#endif
