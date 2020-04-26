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

static RG_FORCEINLINE int subs_c(int x, int y) {
  return std::max(0, x - y);
}

static RG_FORCEINLINE int subs_16_c(int x, int y) {
  return std::max(0, x - y);
}

template<bool chroma>
static RG_FORCEINLINE float subs_32_c(float x, float y) {
  constexpr float pixel_min = chroma ? -0.5f : 0.0f;
  return std::max(pixel_min, x - y);
}

static RG_FORCEINLINE float subs_32_c_for_diff(float x, float y) {
  constexpr float pixel_min = 0.0f;
  return std::max(pixel_min, x - y);
}

static RG_FORCEINLINE int adds_c(int x, int y) {
  constexpr int pixel_max = 255;
  return std::min(pixel_max, x + y);
}

template<int bits_per_pixel>
RG_FORCEINLINE int adds_16_c(int x, int y) {
  constexpr int pixel_max = (1 << bits_per_pixel) - 1;
  return std::min(pixel_max, x + y);
}

template<bool chroma>
RG_FORCEINLINE float adds_32_c(float x, float y) {
  constexpr float pixel_max = chroma ? 0.5f : 1.0f;
  return std::min(pixel_max, x + y);
}

RG_FORCEINLINE float adds_32_c_for_diff(float x, float y) {
  constexpr float pixel_max = 1.0f;
  return std::min(pixel_max, x + y);
}

template<bool chroma>
static RG_FORCEINLINE __m128 _mm_subs_ps(__m128 a, __m128 b) {
  const __m128 pixel_min = chroma ? _mm_set1_ps(-0.5f) : _mm_set1_ps(0.0f);
  return _mm_max_ps(_mm_sub_ps(a, b), pixel_min);
}

// for use case: diff = pixel1 - pixel2, saturates to zero
static RG_FORCEINLINE __m128 _mm_subs_ps_for_diff(__m128 a, __m128 b) {
  const __m128 pixel_min = _mm_set1_ps(0.0f);
  return _mm_max_ps(_mm_sub_ps(a, b), pixel_min);
}

template<bool chroma>
static RG_FORCEINLINE __m128 _mm_adds_ps(__m128 a, __m128 b) {
  const __m128 pixel_max = chroma ? _mm_set1_ps(0.5f) : _mm_set1_ps(1.0f);
  return _mm_min_ps(_mm_add_ps(a, b), pixel_max);
}

static RG_FORCEINLINE __m128 _mm_adds_ps_for_diff(__m128 a, __m128 b) {
  const __m128 pixel_max = _mm_set1_ps(1.0f);
  return _mm_min_ps(_mm_add_ps(a, b), pixel_max);
}


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

/*
in / out  in     in    tmp  tmp
#define	sharpen(center, minus, plus, reg1, reg2)\
__asm	SSE_RMOVE	reg1,				minus			\
__asm	SSE_RMOVE	reg2,				plus			\
__asm	psubusb		reg1,				plus			\
__asm	psubusb		reg2,				minus			\
__asm	psrlw		plus,				1				\
__asm	psrlw		minus,				1				\
__asm	pand		plus,				shift_mask		\
__asm	pand		minus,				shift_mask		\
__asm	pminub		plus,				reg1			\
__asm	pminub		minus,				reg2			\
__asm	paddusb		center,				plus			\
__asm	psubusb		center,				minus
*/

// sharpen, neighbourdiff: helpers for mode 25

// good for all integer
template<int bits_per_pixel>
static RG_FORCEINLINE int sharpen_c(const int& center, const int& minus, const int& plus) {
  auto mp_diff = subs_16_c(minus, plus);
  auto pm_diff = subs_16_c(plus, minus);
  auto m_per2 = minus >> 1;
  auto p_per2 = plus >> 1;
  auto min_1 = std::min(p_per2, mp_diff);
  auto min_2 = std::min(m_per2, pm_diff);
  return subs_16_c(adds_16_c<bits_per_pixel>(center, min_1), min_2);
}

static RG_FORCEINLINE __m128i _MM_SRLI_EPI8(const __m128i& v, int imm) {
  return _mm_and_si128(_mm_set1_epi8(0xFF >> imm), _mm_srli_epi32(v, imm));
}

static RG_FORCEINLINE __m128i sharpen(const __m128i& center, const __m128i& minus, const __m128i& plus) {
  auto mp_diff = _mm_subs_epu8(minus, plus);
  auto pm_diff = _mm_subs_epu8(plus, minus);
  auto m_per2 = _MM_SRLI_EPI8(minus, 1);
  auto p_per2 = _MM_SRLI_EPI8(plus, 1);
  auto min_1 = _mm_min_epu8(p_per2, mp_diff);
  auto min_2 = _mm_min_epu8(m_per2, pm_diff);
  return _mm_subs_epu8(_mm_adds_epu8(center, min_1), min_2);
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE __m128i sharpen_16(const __m128i& center, const __m128i& minus, const __m128i& plus) {
  auto mp_diff = _mm_subs_epu16(minus, plus);
  auto pm_diff = _mm_subs_epu16(plus, minus);
  auto m_per2 = _mm_srli_epi16(minus, 1);
  auto p_per2 = _mm_srli_epi16(plus, 1);
  auto min_1 = _mm_min_epu16(p_per2, mp_diff);
  auto min_2 = _mm_min_epu16(m_per2, pm_diff);
  auto result1 = _mm_adds_epu16(center, min_1);
  if constexpr (bits_per_pixel < 16) {
    const auto max_pixel_value = _mm_set1_epi16((1 << bits_per_pixel) - 1);
    result1 = _mm_min_epi16(result1, max_pixel_value);
  }
  auto result = _mm_subs_epu16(result1, min_2);
  return result;
}


template<bool chroma>
static RG_FORCEINLINE float sharpen_32_c(const float& center, const float& minus, const float& plus) {
  auto mp_diff = subs_32_c_for_diff(minus, plus);
  auto pm_diff = subs_32_c_for_diff(plus, minus);
  auto m_per2 = minus * 0.5f;
  auto p_per2 = plus * 0.5f;
  auto min_1 = std::min(p_per2, mp_diff);
  auto min_2 = std::min(m_per2, pm_diff);
  return subs_32_c<chroma>(adds_32_c<chroma>(center, min_1), min_2);
}

template<bool chroma>
static RG_FORCEINLINE __m128 sharpen_32(const __m128& center, const __m128& minus, const __m128& plus) {
  auto mp_diff = _mm_subs_ps_for_diff(minus, plus); // for diff: like luma (template false) 0..1
  auto pm_diff = _mm_subs_ps_for_diff(plus, minus);
  const auto half = _mm_set1_ps(0.5f);
  auto m_per2 = _mm_mul_ps(minus, half);
  auto p_per2 = _mm_mul_ps(plus, half);
  auto min_1 = _mm_min_ps(p_per2, mp_diff);
  auto min_2 = _mm_min_ps(m_per2, pm_diff);
  return  _mm_subs_ps<chroma>(_mm_adds_ps<chroma>(center, min_1), min_2);
}


/*
                       out    out     out    in        in         in
#define neighbourdiff(minus, plus, center1_as_centerNext, center2, neighbour, nullreg)	\
__asm	SSE_RMOVE	center1,			center2		\
__asm	psubusb		center2,			neighbour	\
__asm	psubusb		neighbour,			center1		\
__asm	SSE_RMOVE	minus,				center2		\
__asm	SSE_RMOVE	plus,				neighbour	\
__asm	pcmpeqb		center2,			nullreg		\
__asm	pcmpeqb		neighbour,			nullreg		\
__asm	por			minus,				center2		\
__asm	pand		center2,			neighbour	\
__asm	por			plus,				neighbour	\
__asm	psubusb		minus,				center2		\
__asm	psubusb		plus,				center2

  // c2 = 9 2 5 1
  // n  = 4 3 5 255
                                                c2            n            minus           plus
__asm	SSE_RMOVE	center1,			center2		\      9 2 5 1     4 3 5 255
__asm	psubusb		center2,			neighbour	\      5 0 0 0    
__asm	psubusb		neighbour,			center1		\                0 1 0 254
__asm	SSE_RMOVE	minus,				center2		\                                 5 0 0 0  
__asm	SSE_RMOVE	plus,				neighbour	\                                                  0 1 0 254 
__asm	pcmpeqb		center2,			nullreg		\      00 FF FF FF   
__asm	pcmpeqb		neighbour,			nullreg		\               FF 00 FF 00
__asm	por			minus,				center2		\                                   5 FF FF FF
__asm	pand		center2,			neighbour	\        00 00 FF 00
__asm	por			plus,				neighbour	\                                                    FF 1 FF 254
__asm	psubusb		minus,				center2		\                                 5 FF 00 FF
__asm	psubusb		plus,				center2                                                      FF 1 00 254

*/
// center1 and center2 are changing during consecutive calls
// helper for mode 25_mode24
static RG_FORCEINLINE void neighbourdiff_orig(__m128i& minus, __m128i& plus, __m128i& center_next, __m128i center2, __m128i neighbour, const __m128i& zero) {
  center_next = center2; // save
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  auto cn_diff = _mm_subs_epu8(center2, neighbour); // 5 0 0 0
  auto nc_diff = _mm_subs_epu8(neighbour, center2); // 0 1 0 254

  auto cndiff_masked = _mm_cmpeq_epi8(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm_cmpeq_epi8(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm_and_si128(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  minus = _mm_or_si128(cn_diff, cndiff_masked); // 5 FF FF FF
  plus = _mm_or_si128(nc_diff, ncdiff_masked);  // FF 1  FF 254

  minus = _mm_subs_epu8(minus, cn_equal); // 5 FF 00 FF 
  plus = _mm_subs_epu8(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

// helper for mode 25
template<int bits_per_pixel>
static RG_FORCEINLINE void neighbourdiff_c(int& minus, int& plus, int center, int neighbour) {
  bool n_ge_c = center <= neighbour;
  bool c_ge_n = neighbour <= center;
  bool equ = center == neighbour;

  constexpr int max_mask = (1 << bits_per_pixel) - 1;
  // an appropriately big number to use for testing max 
  // in sharpen

  if (equ) {
    minus = 0; // min_mask
    plus = 0; // min_mask
  }
  else {
    if (n_ge_c)
      minus = max_mask;
    else
      minus = center - neighbour;
    if (c_ge_n)
      plus = max_mask;
    else
      plus = neighbour - center;
  }

  /*
  // fixme: to less SIMD-like (it was reverse engineered from asm)
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  auto cn_diff = subs_c(center, neighbour); // 5 0 0 0
  auto nc_diff = subs_c(neighbour, center); // 0 1 0 254

  constexpr int max_mask = (1 << bits_per_pixel) - 1; // or just enough to use a very big common number? 
  // plus and minus is used in sharpen, see there

  auto cndiff_masked = cn_diff == 0 ? max_mask : 0; // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = nc_diff == 0 ? max_mask : 0;; // FF where n <= c     FF 00 FF 00
  auto cn_equal = cndiff_masked & ncdiff_masked; // FF where c == n   00 00 FF 00

  minus = cn_diff | cndiff_masked; // 5 FF FF FF
  plus =  nc_diff | ncdiff_masked;  // FF 1  FF 254

  minus = subs_c(minus, cn_equal); // 5 FF 00 FF 
  plus = subs_c(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
  */
}

// helper for mode 25
// differences are chroma or luma independent
static RG_FORCEINLINE void neighbourdiff_32_c(float& minus, float& plus, float center, float neighbour) {
  bool n_ge_c = center <= neighbour;
  bool c_ge_n = neighbour <= center;
  bool equ = center == neighbour;

  constexpr float max_mask = 1.0f;
  // an appropriately big number to use for testing max 
  // in sharpen

  if (equ) {
    minus = 0; // min_mask
    plus = 0; // min_mask
  }
  else {
    if (n_ge_c)
      minus = max_mask;
    else
      minus = center - neighbour;
    if (c_ge_n)
      plus = max_mask;
    else
      plus = neighbour - center;
  }

  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

// helper for mode 25
static RG_FORCEINLINE void neighbourdiff(__m128i& minus, __m128i& plus, __m128i center, __m128i neighbour, const __m128i& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  auto cn_diff = _mm_subs_epu8(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm_subs_epu8(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm_cmpeq_epi8(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm_cmpeq_epi8(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm_and_si128(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  minus = _mm_or_si128(cn_diff, cndiff_masked); // 5 FF FF FF
  plus = _mm_or_si128(nc_diff, ncdiff_masked);  // FF 1  FF 254

  minus = _mm_subs_epu8(minus, cn_equal); // 5 FF 00 FF 
  plus = _mm_subs_epu8(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE void neighbourdiff_16(__m128i& minus, __m128i& plus, __m128i center, __m128i neighbour, const __m128i& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  // no bits_per_pixel template, mask FFFF is big enough for sharpen to subs saturate
  auto cn_diff = _mm_subs_epu16(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm_subs_epu16(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm_cmpeq_epi16(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm_cmpeq_epi16(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm_and_si128(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  minus = _mm_or_si128(cn_diff, cndiff_masked); // 5 FF FF FF
  plus = _mm_or_si128(nc_diff, ncdiff_masked);  // FF 1  FF 254

  minus = _mm_subs_epu16(minus, cn_equal); // 5 FF 00 FF 
  plus = _mm_subs_epu16(plus, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static RG_FORCEINLINE void neighbourdiff_32(__m128& minus, __m128& plus, __m128 center, __m128 neighbour, const __m128& zero) {
  // c2 = 9 2 5 1
  // n  = 4 3 5 255
  // FF in integer, max_mask=1.0 in float
  // for diffs: subs must clamp at 0
  auto cn_diff = _mm_subs_ps_for_diff(center, neighbour); // 5 0 0 0
  auto nc_diff = _mm_subs_ps_for_diff(neighbour, center); // 0 1 0 254

  auto cndiff_masked = _mm_cmpeq_ps(cn_diff, zero); // FF where c <= n     00 FF FF FF
  auto ncdiff_masked = _mm_cmpeq_ps(nc_diff, zero); // FF where n <= c     FF 00 FF 00
  auto cn_equal = _mm_and_ps(cndiff_masked, ncdiff_masked); // FF where c == n   00 00 FF 00

  // max_mask where <=
  // min_mask (0) where =
  const auto max_mask = _mm_set1_ps(1.0f);
  // a, b, mask. If mask then b else a
  minus = _mm_blendv_ps(cn_diff, max_mask, cndiff_masked); // 5 FF FF FF
  plus = _mm_blendv_ps(nc_diff, max_mask, ncdiff_masked);  // FF 1  FF 254

  minus = _mm_blendv_ps(minus, zero, cn_equal); // 5 FF 00 FF 
  plus = _mm_blendv_ps(plus, zero, cn_equal);   // FF 1 00 254
  // When called for pixel pairs, minimum values of all minuses and all pluses are collected
  // min of cn_diff or 00 if there was any equality
  // min of nc_diff or 00 if there was any equality
  // Note: on equality both minus and plus will be zero, sharpen will do nothing
  // these values will be passed to the "sharpen"
}

// end of mode 25 helpers

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
