#ifndef __COMMON_H__
#define __COMMON_H__

#include <algorithm>
#define NOMINMAX
#include <Windows.h>
#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include <smmintrin.h>

typedef unsigned char Byte;

#define RG_FORCEINLINE __forceinline

#define USE_MOVPS

enum InstructionSet {
    SSE2,
    SSE3
};

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

static RG_FORCEINLINE bool is_16byte_aligned(const void *ptr) {
    return (((unsigned long)ptr) & 15) == 0;
}

static RG_FORCEINLINE __m128i simd_clip(const __m128i &val, const __m128i &minimum, const __m128i &maximum) {
  return _mm_max_epu8(_mm_min_epu8(val, maximum), minimum);
}

// SSE4!
// PF avs+
static RG_FORCEINLINE __m128i simd_clip_16(const __m128i &val, const __m128i &minimum, const __m128i &maximum) {
  return _mm_max_epu16(_mm_min_epu16(val, maximum), minimum); // SSE4
}

// PF avs+
static RG_FORCEINLINE __m128 simd_clip_32(const __m128 &val, const __m128 &minimum, const __m128 &maximum) {
  return _mm_max_ps(_mm_min_ps(val, maximum), minimum); 
}



static RG_FORCEINLINE void sort_pair(__m128i &a1, __m128i &a2)
{
  const __m128i tmp = _mm_min_epu8 (a1, a2);
  a2 = _mm_max_epu8 (a1, a2);
  a1 = tmp;
}

// SSE4
// PF avs+
static RG_FORCEINLINE void sort_pair_16(__m128i &a1, __m128i &a2)
{
  const __m128i tmp = _mm_min_epu16 (a1, a2);
  a2 = _mm_max_epu16 (a1, a2);
  a1 = tmp;
}

// PF avs+
static RG_FORCEINLINE void sort_pair_32(__m128 &a1, __m128 &a2)
{
  const __m128 tmp = _mm_min_ps (a1, a2);
  a2 = _mm_max_ps (a1, a2);
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

template<InstructionSet optLevel>
static RG_FORCEINLINE __m128i simd_loada_si128(const Byte* ptr) {
  if (optLevel == SSE2) {
#ifdef USE_MOVPS
    return _mm_castps_si128(_mm_load_ps(reinterpret_cast<const float*>(ptr)));
#else
    return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
  }
  return _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
}

//mask ? a : b
static RG_FORCEINLINE __m128i blend(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
  //return  _mm_blendv_epi8 (otherwise, desired, mask);
  auto andop = _mm_and_si128(mask , desired);
  auto andnop = _mm_andnot_si128(mask, otherwise);
  return _mm_or_si128(andop, andnop);
}

//avs+
//sse4
static RG_FORCEINLINE __m128i blend_16(__m128i const &mask, __m128i const &desired, __m128i const &otherwise) {
  return  _mm_blendv_epi8 (otherwise, desired, mask); // no need for epi 16 here
  //auto andop = _mm_and_si128(mask , desired);
  //auto andnop = _mm_andnot_si128(mask, otherwise);
  //return _mm_or_si128(andop, andnop);
}

// sse4
static RG_FORCEINLINE __m128 blend_32(__m128 const &mask, __m128 const &desired, __m128 const &otherwise) {
  return  _mm_blendv_ps (otherwise, desired, mask);
  //auto andop = _mm_and_si128(mask , desired);
  //auto andnop = _mm_andnot_si128(mask, otherwise);
  //return _mm_or_si128(andop, andnop);
}


static RG_FORCEINLINE __m128i abs_diff(__m128i a, __m128i b) {
  //return  _mm_blendv_epi8 (otherwise, desired, mask); // SSE4
  auto positive = _mm_subs_epu8(a, b);
  auto negative = _mm_subs_epu8(b, a);
  return _mm_or_si128(positive, negative);
}

// avs+ todo for sse4
static RG_FORCEINLINE __m128i abs_diff_16(__m128i a, __m128i b) {
  //return  _mm_blendv_epi16 (otherwise, desired, mask); // SSE4
  auto positive = _mm_subs_epu16(a, b);
  auto negative = _mm_subs_epu16(b, a);
  return _mm_or_si128(positive, negative);
}

// avs+
static RG_FORCEINLINE __m128 abs_diff_32(__m128 a, __m128 b) {
  // maybe not optimal, mask may be generated 
  const __m128 absmask = _mm_castsi128_ps(_mm_set1_epi32(~(1<<31))); // 0x7FFFFFFF
  return _mm_and_ps(_mm_sub_ps(a, b), absmask);
}

// PF until I find out better
static RG_FORCEINLINE __m128 _mm_subs_ps(__m128 a, __m128 b) {
const __m128 zero = _mm_setzero_ps();
return _mm_max_ps(_mm_sub_ps(a, b), zero);
}

// PF until I find out better
static RG_FORCEINLINE __m128 _mm_adds_ps(__m128 a, __m128 b) {
  const __m128 one = _mm_set1_ps(1.0f);
  return _mm_min_ps(_mm_add_ps(a, b), one);
}

// PF until I find out better
static RG_FORCEINLINE __m128 _mm_avg_ps(__m128 a, __m128 b) {
  const __m128 div2 = _mm_set1_ps(0.5f);
  return _mm_mul_ps(_mm_add_ps(a, b), div2);
}

static RG_FORCEINLINE __m128i select_on_equal(const __m128i &cmp1, const __m128i &cmp2, const __m128i &current, const __m128i &desired) {
  auto eq = _mm_cmpeq_epi8(cmp1, cmp2);
  return blend(eq, desired, current);
}

static RG_FORCEINLINE __m128i select_on_equal_16(const __m128i &cmp1, const __m128i &cmp2, const __m128i &current, const __m128i &desired) {
  auto eq = _mm_cmpeq_epi16(cmp1, cmp2);
  return blend_16(eq, desired, current);
}

static RG_FORCEINLINE __m128 select_on_equal_32(const __m128 &cmp1, const __m128 &cmp2, const __m128 &current, const __m128 &desired) {
  auto eq = _mm_cmpeq_ps(cmp1, cmp2);
  return blend_32(eq, desired, current);
}

/* Failed trial omitting the middle load for speed up
// 'or'-able extract byte/word/ps#1: shift left 14/12/8 bytes then right 15/14/12 bytes
__m128i a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
__m128i a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
__m128i a2 = _mm_or_si128(_mm_slli_si128(a1,pixelsize), \
_mm_srli_si128(_mm_slli_si128(a3,16-2*pixelsize),16-pixelsize));  \
__m128i a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
__m128i a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
__m128i c = _mm_or_si128(_mm_slli_si128(a4,pixelsize), \
_mm_srli_si128(_mm_slli_si128(a5,16-2*pixelsize),16-pixelsize));  \
__m128i a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); \
__m128i a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
__m128i a7 = _mm_or_si128(_mm_slli_si128(a6,pixelsize), \
_mm_srli_si128(_mm_slli_si128(a8,16-2*pixelsize),16-pixelsize));

__m128i a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
__m128i a2 = simd_loadu_si128<optLevel>((ptr) - (pitch)); \
__m128i a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
__m128i a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
__m128i c  = simd_loadu_si128<optLevel>((ptr) ); \
__m128i a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
__m128i a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
__m128i a7 = simd_loadu_si128<optLevel>((ptr) + (pitch)); \
__m128i a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize));

*/
/* middle pixels aligned load helps +10% speed,
   same VS2015 speed as ICC14 out-of-the box with full unalignde spec
   but checking is_16byte_aligned here has too much overhead, no gain
#define LOAD_SQUARE_SSE_0(optLevel, ptr, pitch, pixelsize) \
__m128i a1, a2, a3, a4, a5, a6, a7, a8, c; \
if(is_16byte_aligned(ptr)) { \
a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loada_si128<optLevel>((ptr) - (pitch)); \
a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
c  = simd_loada_si128<optLevel>((ptr) ); \
a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loada_si128<optLevel>((ptr) + (pitch)); \
a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); \
} else {\
a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loadu_si128<optLevel>((ptr) - (pitch)); \
a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
c  = simd_loadu_si128<optLevel>((ptr) ); \
a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loadu_si128<optLevel>((ptr) + (pitch)); \
a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); \

// worst attempt
__m128i a1, a2, a3, a4, a5, a6, a7, a8, c; \
__m128i a2 = simd_loada_si128<optLevel>((ptr) - (pitch)); \
__m128i a1 =  _mm_insert_epi8(_mm_srli_si128(a2,pixelsize), *((ptr) - (pitch) - (pixelsize)), 15); \
__m128i a3 =  _mm_insert_epi8(_mm_slli_si128(a2,1), *((ptr) - (pitch) + (16)), 0); \
__m128i c  = simd_loada_si128<optLevel>((ptr) ); \
__m128i a4 =  _mm_insert_epi8(_mm_srli_si128(c,pixelsize), *((ptr) - (pixelsize)), 15); \
__m128i a5 =  _mm_insert_epi8(_mm_slli_si128(c,1), *((ptr)  + (16)), 0); \
__m128i a7 = simd_loada_si128<optLevel>((ptr) + (pitch)); \
__m128i a6 =  _mm_insert_epi8(_mm_srli_si128(a7,pixelsize), *((ptr) + (pitch) - (pixelsize)), 15); \
__m128i a8 =  _mm_insert_epi8(_mm_slli_si128(a7,1), *((ptr) + (pitch) + (16)), 0);

if(!is_16byte_aligned(ptr)) { \
a2 = simd_loadu_si128<optLevel>((ptr) - (pitch)); \
c  = simd_loadu_si128<optLevel>((ptr) ); \
a7 = simd_loadu_si128<optLevel>((ptr) + (pitch)); \
} else {\
a2 = simd_loada_si128<optLevel>((ptr) - (pitch)); \
c  = simd_loada_si128<optLevel>((ptr) ); \
a7 = simd_loada_si128<optLevel>((ptr) + (pitch)); \
} \
a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); 

}
*/
#define LOAD_SQUARE_SSE_0(optLevel, ptr, pitch, pixelsize, aligned) \
__m128i a1, a2, a3, a4, a5, a6, a7, a8, c; \
if(!aligned) {\
a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loadu_si128<optLevel>((ptr) - (pitch)); \
a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
c  = simd_loadu_si128<optLevel>((ptr) ); \
a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loadu_si128<optLevel>((ptr) + (pitch)); \
a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); \
} else {\
a1 = simd_loadu_si128<optLevel>((ptr) - (pitch) - (pixelsize)); \
a2 = simd_loada_si128<optLevel>((ptr) - (pitch)); \
a3 = simd_loadu_si128<optLevel>((ptr) - (pitch) + (pixelsize)); \
a4 = simd_loadu_si128<optLevel>((ptr) - (pixelsize)); \
c  = simd_loada_si128<optLevel>((ptr) ); \
a5 = simd_loadu_si128<optLevel>((ptr) + (pixelsize)); \
a6 = simd_loadu_si128<optLevel>((ptr) + (pitch) - (pixelsize)); \
a7 = simd_loada_si128<optLevel>((ptr) + (pitch)); \
a8 = simd_loadu_si128<optLevel>((ptr) + (pitch) + (pixelsize)); \
}

#define LOAD_SQUARE_SSE(optLevel, ptr, pitch) LOAD_SQUARE_SSE_0(optLevel, ptr, pitch, 1, false)
// unaligned or aligned
#define LOAD_SQUARE_SSE_UA(optLevel, ptr, pitch, aligned) LOAD_SQUARE_SSE_0(optLevel, ptr, pitch, 1, aligned)
#define LOAD_SQUARE_SSE_16(ptr, pitch) LOAD_SQUARE_SSE_0(SSE3, ptr, pitch, 2, false)

#define LOAD_SQUARE_SSE_32(ptr, pitch) \
__m128 a1 = _mm_loadu_ps((const float *)((ptr) - (pitch) - 4)); \
__m128 a2 = _mm_loadu_ps((const float *)((ptr) - (pitch))); \
__m128 a3 = _mm_loadu_ps((const float *)((ptr) - (pitch) + (4))); \
__m128 a4 = _mm_loadu_ps((const float *)((ptr) - (4))); \
__m128 c  = _mm_loadu_ps((const float *)((ptr) )); \
__m128 a5 = _mm_loadu_ps((const float *)((ptr) + (4))); \
__m128 a6 = _mm_loadu_ps((const float *)((ptr) + (pitch) - (4))); \
__m128 a7 = _mm_loadu_ps((const float *)((ptr) + (pitch))); \
__m128 a8 = _mm_loadu_ps((const float *)((ptr) + (pitch) + (4)));

// pointers and pitch are byte-based
#define LOAD_SQUARE_CPP_T(pixel_t, ptr, pitch) \
    pixel_t a1 = *(pixel_t *)((ptr) - (pitch) - sizeof(pixel_t)); \
    pixel_t a2 = *(pixel_t *)((ptr) - (pitch)); \
    pixel_t a3 = *(pixel_t *)((ptr) - (pitch) + sizeof(pixel_t)); \
    pixel_t a4 = *(pixel_t *)((ptr) - sizeof(pixel_t)); \
    pixel_t c  = *(pixel_t *)((ptr) ); \
    pixel_t a5 = *(pixel_t *)((ptr) + sizeof(pixel_t)); \
    pixel_t a6 = *(pixel_t *)((ptr) + (pitch) - sizeof(pixel_t)); \
    pixel_t a7 = *(pixel_t *)((ptr) + (pitch)); \
    pixel_t a8 = *(pixel_t *)((ptr) + (pitch) + sizeof(pixel_t));

#define LOAD_SQUARE_CPP(ptr, pitch) LOAD_SQUARE_CPP_T(Byte, ptr, pitch);
#define LOAD_SQUARE_CPP_16(ptr, pitch) LOAD_SQUARE_CPP_T(uint16_t, ptr, pitch);
#define LOAD_SQUARE_CPP_32(ptr, pitch) LOAD_SQUARE_CPP_T(float, ptr, pitch);

/*
#define LOAD_SQUARE_CPP(ptr, pitch) \
    Byte a1 = *((ptr) - (pitch) - 1); \
    Byte a2 = *((ptr) - (pitch)); \
    Byte a3 = *((ptr) - (pitch) + 1); \
    Byte a4 = *((ptr) - 1); \
    Byte c  = *((ptr) ); \
    Byte a5 = *((ptr) + 1); \
    Byte a6 = *((ptr) + (pitch) - 1); \
    Byte a7 = *((ptr) + (pitch)); \
    Byte a8 = *((ptr) + (pitch) + 1);
*/
#endif
