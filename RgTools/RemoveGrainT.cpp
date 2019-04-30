// TemporalRepair from RemoveGrainT 1.0 package"
//
// An Avisynth plugin for removing grain from progressive video
//
// 2007 By Rainer Wittmann <gorw@gmx.de>
// 2019 Additional work by Ferenc Pint�r
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// To get a copy of the GNU General Public License write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .

#include "RemoveGrainT.h"
#include "common.h"
#include <cassert>
#include "emmintrin.h"
#include "immintrin.h"

// called from GetFrame
typedef void (PlaneProcessor_t)(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height);

typedef void (temporal_repair_processor_simd)(uint8_t* dp, const uint8_t* sp1, const uint8_t* sp2, const uint8_t* pp, const uint8_t* np);
typedef int (temporal_repair_processor_c)(int src_1, int src_2, int src_prev, int src_next);
typedef float (temporal_repair_processor_32_c)(float src_1, float src_2, float src_prev, float src_next);

typedef void (PlaneProcessor_st)(BYTE* dp8, const BYTE* previous8, const BYTE* sp8, const BYTE* next8, intptr_t* pitches, int width, int height);

typedef __m128i (smooth_temporal_repair_processor_simd)(BYTE* dp, const BYTE* previous, const intptr_t pfpitch, const BYTE* sp, const intptr_t ofpitch, const BYTE* next, const intptr_t nfpitch);
typedef int (smooth_temporal_repair_processor_c)(BYTE* dp, const BYTE* previous, const intptr_t pfpitch, const BYTE* sp, const intptr_t ofpitch, const BYTE* next, const intptr_t nfpitch);

/*************************************

    TemporalRepair, Mode 0, Mode 4

*************************************/

RG_FORCEINLINE void RepairPixel_mode0_sse2(uint8_t *dest, const uint8_t *src1, const uint8_t *src2, const uint8_t *previous, const uint8_t *next)
{
	auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
	auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
	auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
	auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));

	auto min_np2 = _mm_min_epu8(_mm_min_epu8(src_next, src_prev), src_2);
	auto max_np2 = _mm_max_epu8(_mm_max_epu8(src_next, src_prev), src_2);
	auto result = simd_clip(src_1, min_np2, max_np2);

	_mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void RepairPixel_mode0_sse41_16(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next)
{
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
  auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));

  auto min_np2 = _mm_min_epu16(_mm_min_epu16(src_next, src_prev), src_2);
  auto max_np2 = _mm_max_epu16(_mm_max_epu16(src_next, src_prev), src_2);
  auto result = simd_clip_16(src_1, min_np2, max_np2);

  _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

RG_FORCEINLINE void RepairPixel_mode0_sse2_32(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next)
{
  auto src_next = _mm_loadu_ps(reinterpret_cast<const float*>(next));
  auto src_prev = _mm_loadu_ps(reinterpret_cast<const float*>(previous));
  auto src_1 = _mm_loadu_ps(reinterpret_cast<const float*>(src1));
  auto src_2 = _mm_loadu_ps(reinterpret_cast<const float*>(src2));

  auto min_np2 = _mm_min_ps(_mm_min_ps(src_next, src_prev), src_2);
  auto max_np2 = _mm_max_ps(_mm_max_ps(src_next, src_prev), src_2);
  auto result = simd_clip_32(src_1, min_np2, max_np2);

  _mm_storeu_ps(reinterpret_cast<float*>(dest), result);
}

RG_FORCEINLINE int RepairPixel_mode0_c(int src_1, int src_2, int src_prev, int src_next)
{
	auto min_np2 = std::min(std::min(src_next, src_prev), src_2);
	auto max_np2 = std::max(std::max(src_next, src_prev), src_2);
	return clip(src_1, min_np2, max_np2);
}

RG_FORCEINLINE int RepairPixel_mode0_16_c(int src_1, int src_2, int src_prev, int src_next)
{
  auto min_np2 = std::min(std::min(src_next, src_prev), src_2);
  auto max_np2 = std::max(std::max(src_next, src_prev), src_2);
  return clip_16(src_1, min_np2, max_np2);
}

RG_FORCEINLINE float RepairPixel_mode0_32_c(float src_1, float src_2, float src_prev, float src_next)
{
  auto min_np2 = std::min(std::min(src_next, src_prev), src_2);
  auto max_np2 = std::max(std::max(src_next, src_prev), src_2);
  return clip_32(src_1, min_np2, max_np2);
}


// SSE4.1 simulation for SSE2
// false: a, true: b
RG_FORCEINLINE __m128i _MM_BLENDV_EPI8(__m128i const& a, __m128i const& b, __m128i const& selector) {
  return _mm_or_si128(_mm_and_si128(selector, b), _mm_andnot_si128(selector, a));
}

RG_FORCEINLINE void BRepairPixel_mode4_sse2(uint8_t *dest, const uint8_t *src1, const uint8_t *src2, const uint8_t *previous, const uint8_t *next)
{
	__m128i reg3, reg5;
	auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
	auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
  auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));
	
  auto max_np = _mm_max_epu8(src_next, src_prev);
	auto min_np = _mm_min_epu8(src_next, src_prev);

	auto diff_src2_minnp = _mm_subs_epu8(src_2, min_np);
  reg5 = _mm_adds_epu8(_mm_adds_epu8(diff_src2_minnp, diff_src2_minnp) , min_np); // diff_src2_minnp * 2 - min_np
  
  auto diff_maxnp_src2 = _mm_subs_epu8(max_np, src_2);
  reg3 = _mm_subs_epu8(max_np, _mm_adds_epu8(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

	reg5 = _mm_min_epu8(reg5, max_np);
	reg3 = _mm_max_epu8(reg3, min_np);

	auto equ1 = _mm_cmpeq_epi8(min_np, reg5);
	auto equ2 = _mm_cmpeq_epi8(max_np, reg3);
  auto equ = _mm_or_si128(equ1, equ2);// _mm_max_epu8(equ1, equ2); // pracically or. mask set if min_np == reg5 or max_np == reg3

  reg5 = simd_clip(src_1, reg3, reg5);
  // FIXME: SSE4.1
  // real SSE4.1 _mm_blendv_epi8: 2830 vs 2960 FPS, worth using it
  auto result = _MM_BLENDV_EPI8(reg5, src_2, equ); // keep src2 where equal

	_mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

// real SSE4.1 _mm_blendv_epi8: 2830 vs 2960 FPS, worth using it
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void BRepairPixel_mode4_sse41(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next)
{
  __m128i reg3, reg5;
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
  auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));

  auto max_np = _mm_max_epu8(src_next, src_prev);
  auto min_np = _mm_min_epu8(src_next, src_prev);

  auto diff_src2_minnp = _mm_subs_epu8(src_2, min_np);
  reg5 = _mm_adds_epu8(_mm_adds_epu8(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = _mm_subs_epu8(max_np, src_2);
  reg3 = _mm_subs_epu8(max_np, _mm_adds_epu8(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = _mm_min_epu8(reg5, max_np);
  reg3 = _mm_max_epu8(reg3, min_np);

  auto equ1 = _mm_cmpeq_epi8(min_np, reg5);
  auto equ2 = _mm_cmpeq_epi8(max_np, reg3);
  auto equ = _mm_or_si128(equ1, equ2);// _mm_max_epu8(equ1, equ2); // pracically or. mask set if min_np == reg5 or max_np == reg3

  reg5 = simd_clip(src_1, reg3, reg5);
  auto result = _mm_blendv_epi8(reg5, src_2, equ); // keep src2 where equal

  _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void BRepairPixel_mode4_sse41_16(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next)
{
  __m128i reg3, reg5;
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
  auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));

  auto max_np = _mm_max_epu16(src_next, src_prev);
  auto min_np = _mm_min_epu16(src_next, src_prev);

  auto diff_src2_minnp = _mm_subs_epu16(src_2, min_np);
  reg5 = _mm_adds_epu16(_mm_adds_epu16(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = _mm_subs_epu16(max_np, src_2);
  reg3 = _mm_subs_epu16(max_np, _mm_adds_epu16(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = _mm_min_epu16(reg5, max_np);
  reg3 = _mm_max_epu16(reg3, min_np);

  auto equ1 = _mm_cmpeq_epi16(min_np, reg5);
  auto equ2 = _mm_cmpeq_epi16(max_np, reg3);
  auto equ = _mm_or_si128(equ1, equ2); // mask set if min_np == reg5 or max_np == reg3 FIXME

  reg5 = simd_clip_16(src_1, reg3, reg5);

  auto result = _mm_blendv_epi8(reg5, src_2, equ); // keep src2 where equal

  _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

template<bool chroma>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void BRepairPixel_mode4_sse41_32(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next)
{
  __m128 reg3, reg5;
  auto src_next = _mm_loadu_ps(reinterpret_cast<const float*>(next));
  auto src_prev = _mm_loadu_ps(reinterpret_cast<const float*>(previous));
  auto src_1 = _mm_loadu_ps(reinterpret_cast<const float*>(src1));
  auto src_2 = _mm_loadu_ps(reinterpret_cast<const float*>(src2));

  auto max_np = _mm_max_ps(src_next, src_prev);
  auto min_np = _mm_min_ps(src_next, src_prev);

  auto diff_src2_minnp = _mm_subs_ps<chroma>(src_2, min_np);
  reg5 = _mm_adds_ps<chroma>(_mm_adds_ps<chroma>(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = _mm_subs_ps<chroma>(max_np, src_2);
  reg3 = _mm_subs_ps<chroma>(max_np, _mm_adds_ps<chroma>(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = _mm_min_ps(reg5, max_np);
  reg3 = _mm_max_ps(reg3, min_np);

  auto equ1 = _mm_cmpeq_ps(min_np, reg5);
  auto equ2 = _mm_cmpeq_ps(max_np, reg3);
  auto equ = _mm_or_ps(equ1, equ2); // mask set if min_np == reg5 or max_np == reg3 FIXME

  reg5 = simd_clip_32(src_1, reg3, reg5);

  auto result = _mm_blendv_ps(reg5, src_2, equ); // keep src2 where equal

  _mm_storeu_ps(reinterpret_cast<float*>(dest), result);
}

// unaligned, aligned
template<typename pixel_t, temporal_repair_processor_simd processor, temporal_repair_processor_simd processor_a>
void temporal_repair_mode0and4_sse2(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height)
{
  dpitch /= sizeof(pixel_t);
  spitch1 /= sizeof(pixel_t);
  spitch2 /= sizeof(pixel_t);
  ppitch /= sizeof(pixel_t);
  npitch /= sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* sp1 = reinterpret_cast<const pixel_t*>(sp1_8);
  const pixel_t* sp2 = reinterpret_cast<const pixel_t*>(sp2_8);
  const pixel_t* pp = reinterpret_cast<const pixel_t*>(pp8);
  const pixel_t* np = reinterpret_cast<const pixel_t*>(np8);

  const int pixels_at_a_time = 16 / sizeof(pixel_t);

  int mod_width = width / pixels_at_a_time * pixels_at_a_time;

  for (int y = 0; y < height; y++) {
    // aligned
    for (int x = 0; x < mod_width; x += pixels_at_a_time) {
      processor_a(dp + x, sp1 + x, sp2 + x, pp + x, np + x);
    }

    if (mod_width != width) {
      const int xx = width - pixels_at_a_time;
      processor(dp + xx, sp1 + xx, sp2 + xx, pp + xx, np + xx);
    }

    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }

}

// unaligned, aligned
template<typename pixel_t, temporal_repair_processor_simd processor, temporal_repair_processor_simd processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void temporal_repair_mode0and4_sse41(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height)
{
  dpitch /= sizeof(pixel_t);
  spitch1 /= sizeof(pixel_t);
  spitch2 /= sizeof(pixel_t);
  ppitch /= sizeof(pixel_t);
  npitch /= sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* sp1 = reinterpret_cast<const pixel_t*>(sp1_8);
  const pixel_t* sp2 = reinterpret_cast<const pixel_t*>(sp2_8);
  const pixel_t* pp = reinterpret_cast<const pixel_t*>(pp8);
  const pixel_t* np = reinterpret_cast<const pixel_t*>(np8);

  const int pixels_at_a_time = 16 / sizeof(pixel_t);

  int mod_width = width / pixels_at_a_time * pixels_at_a_time;

  for (int y = 0; y < height; y++) {
    // aligned
    for (int x = 0; x < mod_width; x += pixels_at_a_time) {
      processor_a((uint8_t *)(dp + x), (uint8_t*)(sp1 + x), (uint8_t*)(sp2 + x), (uint8_t*)(pp + x), (uint8_t*)(np + x));
    }

    if (mod_width != width) {
      const int xx = width - pixels_at_a_time;
      processor((uint8_t*)(dp + xx), (uint8_t*)(sp1 + xx), (uint8_t*)(sp2 + xx), (uint8_t*)(pp + xx), (uint8_t*)(np + xx));
    }

    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }

}

template<typename pixel_t, temporal_repair_processor_c processor>
void temporal_repair_mode0and4_c(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height)
{
  dpitch /= sizeof(pixel_t);
  spitch1 /= sizeof(pixel_t);
  spitch2 /= sizeof(pixel_t);
  ppitch /= sizeof(pixel_t);
  npitch /= sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* sp1 = reinterpret_cast<const pixel_t*>(sp1_8);
  const pixel_t* sp2 = reinterpret_cast<const pixel_t*>(sp2_8);
  const pixel_t* pp = reinterpret_cast<const pixel_t*>(pp8);
  const pixel_t* np = reinterpret_cast<const pixel_t*>(np8);

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      dp[x] = processor(sp1[x], sp2[x], pp[x], np[x]);
    }
    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }
}

template<typename pixel_t, temporal_repair_processor_32_c processor>
void temporal_repair_mode0and4_32_c(BYTE* dp8, int dpitch, const BYTE* sp1_8, int spitch1, const BYTE* sp2_8, int spitch2, const BYTE* pp8, int ppitch, const BYTE* np8, int npitch, int width, int height)
{
  dpitch /= sizeof(pixel_t);
  spitch1 /= sizeof(pixel_t);
  spitch2 /= sizeof(pixel_t);
  ppitch /= sizeof(pixel_t);
  npitch /= sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* sp1 = reinterpret_cast<const pixel_t*>(sp1_8);
  const pixel_t* sp2 = reinterpret_cast<const pixel_t*>(sp2_8);
  const pixel_t* pp = reinterpret_cast<const pixel_t*>(pp8);
  const pixel_t* np = reinterpret_cast<const pixel_t*>(np8);

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      dp[x] = processor(sp1[x], sp2[x], pp[x], np[x]);
    }
    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }
}

RG_FORCEINLINE int subs_c(int x, int y) {
  return std::max(0, x - y);
}

RG_FORCEINLINE int subs_16_c(int x, int y) {
  return std::max(0, x - y);
}

template<bool chroma>
RG_FORCEINLINE float subs_32_c(float x, float y) {
#if 1
  constexpr float pixel_min = chroma ? -0.5f : 0.0f;
  return std::max(pixel_min, x - y);
#else
  // no clamp, heavy artifacts
  return x - y; // float: no clamp
#endif
}

RG_FORCEINLINE int adds_c(int x, int y) {
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
#if 1
  constexpr float pixel_max = chroma ? 0.5f : 1.0f;
  return std::min(pixel_max, x + y);
#else
  // no clamp, heavy artifacts
  return x + y; // float: no clamp
#endif
}

template<bool chroma>
static RG_FORCEINLINE __m128 _mm_subs_ps(__m128 a, __m128 b) {
#if 1
  const __m128 pixel_min = chroma ? _mm_set1_ps(-0.5f) : _mm_set1_ps(0.0f);
  return _mm_max_ps(_mm_sub_ps(a, b), pixel_min);
#else
  // no clamp, heavy artifacts
  return _mm_sub_ps(a, b);
#endif
}

template<bool chroma>
static RG_FORCEINLINE __m128 _mm_adds_ps(__m128 a, __m128 b) {
#if 1
  const __m128 pixel_max = chroma ? _mm_set1_ps(0.5f) : _mm_set1_ps(1.0f);
  return _mm_min_ps(_mm_add_ps(a, b), pixel_max);
#else
  // no clamp, heavy artifacts
  return _mm_add_ps(a, b);
#endif
}

RG_FORCEINLINE int BRepairPixel_mode4_c(int src_1, int src_2, int src_prev, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);

  auto diff_src2_minnp = subs_c(src_2, min_np);
  auto reg5 = adds_c(adds_c(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_c(max_np, src_2);
  auto reg3 = subs_c(max_np, adds_c(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = std::min(reg5, max_np);
  reg3 = std::max(reg3, min_np);

  auto equ = min_np == reg5 || max_np == reg3;

  reg5 = clip(src_1, reg3, reg5);
  return equ ? src_2 : reg5;
}

template<int bits_per_pixel>
RG_FORCEINLINE int BRepairPixel_mode4_16_c(int src_1, int src_2, int src_prev, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);

  auto diff_src2_minnp = subs_16_c(src_2, min_np);
  auto reg5 = adds_16_c<bits_per_pixel>(adds_16_c<bits_per_pixel>(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_16_c(max_np, src_2);
  auto reg3 = subs_16_c(max_np, adds_16_c<bits_per_pixel>(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = std::min(reg5, max_np);
  reg3 = std::max(reg3, min_np);

  auto equ = min_np == reg5 || max_np == reg3;

  reg5 = clip_16(src_1, reg3, reg5);
  return equ ? src_2 : reg5;
}

template<bool chroma>
RG_FORCEINLINE float BRepairPixel_mode4_32_c(float src_1, float src_2, float src_prev, float src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);

  auto diff_src2_minnp = subs_32_c<chroma>(src_2, min_np);
  auto reg5 = adds_32_c<chroma>(adds_32_c<chroma>(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_32_c<chroma>(max_np, src_2);
  auto reg3 = subs_32_c<chroma>(max_np, adds_32_c<chroma>(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = std::min(reg5, max_np);
  reg3 = std::max(reg3, min_np);

  auto equ = min_np == reg5 || max_np == reg3;

  reg5 = clip_32(src_1, reg3, reg5);
  return equ ? src_2 : reg5;
}

static PlaneProcessor_t* t_c_functions[] = {
    temporal_repair_mode0and4_c<uint8_t, RepairPixel_mode0_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_c<uint8_t, BRepairPixel_mode4_c>
};

static PlaneProcessor_t* t_c_functions_16[] = {
    temporal_repair_mode0and4_c<uint16_t, RepairPixel_mode0_16_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_c<uint16_t, BRepairPixel_mode4_16_c<16>>
};

static PlaneProcessor_t* t_c_functions_32[] = {
    temporal_repair_mode0and4_32_c<float, RepairPixel_mode0_32_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_32_c<float, BRepairPixel_mode4_32_c<false>> // false: not chroma
};

static PlaneProcessor_t* t_c_functions_chroma_32[] = {
    temporal_repair_mode0and4_32_c<float, RepairPixel_mode0_32_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_32_c<float, BRepairPixel_mode4_32_c<true>> // false: not chroma
};

static PlaneProcessor_t* t_sse2_functions[] = {
    temporal_repair_mode0and4_sse2<uint8_t, RepairPixel_mode0_sse2, RepairPixel_mode0_sse2>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse2<uint8_t, BRepairPixel_mode4_sse2, BRepairPixel_mode4_sse2>
};

static PlaneProcessor_t* t_sse41_functions[] = {
    temporal_repair_mode0and4_sse41<uint8_t, RepairPixel_mode0_sse2, RepairPixel_mode0_sse2>, // no 4.1 version
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<uint8_t, BRepairPixel_mode4_sse41, BRepairPixel_mode4_sse41>
};

static PlaneProcessor_t* t_sse41_functions_16[] = {
    temporal_repair_mode0and4_sse41<uint16_t, RepairPixel_mode0_sse41_16, RepairPixel_mode0_sse41_16>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<uint16_t, BRepairPixel_mode4_sse41_16<16>, BRepairPixel_mode4_sse41_16<16>>
};

static PlaneProcessor_t* t_sse41_functions_32[] = {
    temporal_repair_mode0and4_sse41<float, RepairPixel_mode0_sse2_32, RepairPixel_mode0_sse2_32>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<float, BRepairPixel_mode4_sse41_32<false>, BRepairPixel_mode4_sse41_32<false>>
};

static PlaneProcessor_t* t_sse41_functions_chroma_32[] = {
    temporal_repair_mode0and4_sse41<float, RepairPixel_mode0_sse2_32, RepairPixel_mode0_sse2_32>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<float, BRepairPixel_mode4_sse41_32<true>, BRepairPixel_mode4_sse41_32<true>>
};

static void CompareVideoInfo(VideoInfo& vi1, const VideoInfo& vi2, const char* progname, IScriptEnvironment* env)
{
  if ((vi1.width != vi2.width) || (vi1.height != vi2.height) || !vi1.IsSameColorspace(vi2))
  {
    env->ThrowError("%s: clips must be of equal type", progname);
  }
  if (vi1.num_frames > vi2.num_frames) 
    vi1.num_frames = vi2.num_frames;
}

static void copy_plane(PVideoFrame& destf, PVideoFrame& currf, int plane, IScriptEnvironment* env) {
  const uint8_t* srcp = currf->GetReadPtr(plane);
  int src_pitch = currf->GetPitch(plane);
  int height = currf->GetHeight(plane);
  int row_size = currf->GetRowSize(plane);
  uint8_t* destp = destf->GetWritePtr(plane);
  int dst_pitch = destf->GetPitch(plane);
  env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

class	TemporalRepair : public GenericVideoFilter
{
  PlaneProcessor_t* processor_t_repair;
  PlaneProcessor_t* processor_t_repair_chroma;
  PlaneProcessor_t* processor_t_repair_c;
  PlaneProcessor_t* processor_t_repair_chroma_c;
  int last_frame;
  PClip orig;
  bool grey;

  // MT mode Registration for Avisynth+
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    if (n <= 0 || n - 1 >= last_frame)
      return child->GetFrame(n, env);
    PVideoFrame pf = orig->GetFrame(n - 1, env);
    PVideoFrame sf = orig->GetFrame(n, env);
    PVideoFrame nf = orig->GetFrame(n + 1, env);
    PVideoFrame cf = child->GetFrame(n, env);
    PVideoFrame df = env->NewVideoFrame(vi);

    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

    const int planecount = grey ? 1 : std::min(vi.NumComponents(),3); // no Alpha plane processing
    for (int p = 0; p < planecount; ++p) {
      const int plane = planes[p];

      const bool chroma = plane == PLANAR_U || plane == PLANAR_V;

      PlaneProcessor_t* actual_processor;

      if (sf->GetRowSize(plane) < 16) {
        if (chroma)
          actual_processor = processor_t_repair_chroma_c;
        else
          actual_processor = processor_t_repair_c;
      }
      else {
        if (chroma)
          actual_processor = processor_t_repair_chroma;
        else
          actual_processor = processor_t_repair;
      }
      actual_processor(
        df->GetWritePtr(plane), df->GetPitch(plane),
        cf->GetReadPtr(plane), cf->GetPitch(plane),
        sf->GetReadPtr(plane), sf->GetPitch(plane),
        pf->GetReadPtr(plane), pf->GetPitch(plane),
        nf->GetReadPtr(plane), nf->GetPitch(plane),
        vi.width >> vi.GetPlaneWidthSubsampling(plane), vi.height >> vi.GetPlaneHeightSubsampling(plane));
    }

    if (vi.NumComponents() == 4)
      copy_plane(df, sf, PLANAR_A, env);
 
    return df;
  }

public:
  TemporalRepair(PClip clip, PClip oclip, int mode, bool grey, bool planar, int opt, IScriptEnvironment* env) :
    GenericVideoFilter(clip),
    orig(oclip), grey(grey)
  {
    if (!planar && !vi.IsPlanar())
      env->ThrowError("TemporalRepair: only planar color spaces are supported");

    CompareVideoInfo(vi, orig->GetVideoInfo(), "TemporalRepair", env);

    // only for modes 0 and 4
    assert(mode == 0 || mode == 4);
    switch (vi.BitsPerComponent()) {
    case 8:
      processor_t_repair_c = t_c_functions[mode];
      processor_t_repair_chroma_c = processor_t_repair_c;
      break;
    case 10: case 12: case 14: case 16:
      processor_t_repair_c = t_c_functions_16[mode];
      processor_t_repair_chroma_c = processor_t_repair_c;
      break;
    case 32:
      processor_t_repair_c = t_c_functions_32[mode];
      processor_t_repair_chroma_c = t_c_functions_chroma_32[mode]; // float: special
      break;
    }

    bool sse2 = (env->GetCPUFlags() & CPUF_SSE2) == CPUF_SSE2;
    bool sse41 = (env->GetCPUFlags() & CPUF_SSE4_1) == CPUF_SSE4_1;

    // disable by opt
    if (opt >= 0) {
      if (opt < 1) sse2 = false;
      if (opt < 2) sse41 = false;
    }

    if (opt == 0) { // 0: C
      processor_t_repair = processor_t_repair_c;
      processor_t_repair_chroma = processor_t_repair_chroma_c;
    }
    else {
      switch (vi.BitsPerComponent()) {
      case 8:
        if (sse41)
          processor_t_repair = t_sse41_functions[mode];
        else if (sse2)
          processor_t_repair = t_sse2_functions[mode];
        else
          processor_t_repair = processor_t_repair_c;
        processor_t_repair_chroma = processor_t_repair;
        break;
      case 10: case 12: case 14: case 16:
        if (sse41)
          processor_t_repair = t_sse41_functions_16[mode];
        else
          processor_t_repair = processor_t_repair_c;
        break;
        processor_t_repair_chroma = processor_t_repair;
      case 32:
        if (sse41) {
          processor_t_repair = t_sse41_functions_32[mode];
          processor_t_repair_chroma = t_sse41_functions_chroma_32[mode];
        }
        else {
          processor_t_repair = processor_t_repair_c;
          processor_t_repair_chroma = processor_t_repair_chroma_c;
        }
        break;
      }
    }
    last_frame = vi.num_frames >= 2 ? vi.num_frames - 2 : 0;
  }
};

/*****************************************

    SmoothTemporalRepair, Mode 1, 2, 3

*****************************************/

RG_FORCEINLINE void get_lu(__m128i &lower, __m128i &upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i *>(previous));
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current));
  auto max_np = _mm_max_epu8(src_next, src_prev);
  auto min_np = _mm_min_epu8(src_next, src_prev);
  upper = _mm_subs_epu8(max_np, src_curr);
  lower = _mm_subs_epu8(src_curr, min_np);
}

RG_FORCEINLINE void get_lu_c(int& lower, int& upper, int src_prev, int src_curr, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);
  upper = subs_c(max_np, src_curr);
  lower = subs_c(src_curr, min_np);
}

RG_FORCEINLINE __m128i SmoothTRepair1(uint8_t *dest, __m128i &lower, __m128i &upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  auto tmp_u = _mm_adds_epu8(upper, src_curr);
  auto tmp_l = _mm_subs_epu8(src_curr, lower);

  auto tmp_max = _mm_max_epu8(_mm_max_epu8(tmp_u, src_prev), src_next);
  auto tmp_min = _mm_min_epu8(_mm_min_epu8(tmp_l, src_prev), src_next);
  
  auto result = simd_clip(src_dest, tmp_min, tmp_max);
  
  return result;
}

RG_FORCEINLINE int SmoothTRepair1_c(int dest, int lower, int upper, const int src_prev, const int src_curr, const int src_next)
{
  auto src_dest = dest;

  auto tmp_u = adds_c(upper, src_curr);
  auto tmp_l = subs_c(src_curr, lower);

  auto tmp_max = std::max(std::max(tmp_u, src_prev), src_next);
  auto tmp_min = std::min(std::min(tmp_l, src_prev), src_next);

  auto result = clip((int)src_dest, tmp_min, tmp_max);

  return result;
}

RG_FORCEINLINE __m128i SmoothTRepair2(uint8_t *dest, __m128i lower, __m128i upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i lower1 = _mm_undefined_si128();
  __m128i upper1 = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  get_lu(lower1, upper1, previous, current, next);
  auto uppermax = _mm_max_epu8(upper, upper1);
  auto lowermax = _mm_max_epu8(lower, lower1);
  auto upperlowermax = _mm_max_epu8(uppermax, lowermax);

  auto tmp_max = _mm_adds_epu8(src_curr, upperlowermax);
  auto tmp_min = _mm_subs_epu8(src_curr, upperlowermax);

  auto result = simd_clip(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE int SmoothTRepair2_c(int dest, int lower, int upper, int src_prev, int src_curr, int src_next)
{
  int lower1;
  int upper1;

  auto src_dest = dest;

  get_lu_c(lower1, upper1, src_prev, src_curr, src_next);
  auto uppermax = std::max(upper, upper1);
  auto lowermax = std::max(lower, lower1);
  auto upperlowermax = std::max(uppermax, lowermax);

  auto tmp_max = adds_c(src_curr, upperlowermax);
  auto tmp_min = subs_c(src_curr, upperlowermax);

  auto result = clip((int)src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE void get2diff(__m128i &pdiff, __m128i &ndiff, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i *>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(next));

  pdiff = abs_diff(src_curr, src_prev); // abs(c-p)
  ndiff = abs_diff(src_curr, src_next); // abs(c-n)
}

RG_FORCEINLINE void get2diff_c(int& pdiff, int& ndiff, int src_prev, int src_curr, int src_next)
{
  pdiff = abs(src_curr - src_prev); 
  ndiff = abs(src_curr - src_next);
}

RG_FORCEINLINE __m128i SmoothTRepair3(uint8_t *dest, __m128i &pmax, __m128i &nmax, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));

  get2diff(pdiff, ndiff, previous, current, next);
  pmax = _mm_max_epu8(pmax, pdiff);
  nmax = _mm_max_epu8(nmax, ndiff);

  pmax = _mm_min_epu8(pmax, nmax);
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dest));
  auto tmp_max = _mm_adds_epu8(src_curr, pmax);
  auto tmp_min = _mm_subs_epu8(src_curr, pmax);
  auto result = simd_clip(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE int SmoothTRepair3_c(int dest, int pmax, int nmax, int src_prev, int src_curr, int src_next)
{
  int pdiff;
  int ndiff;

  get2diff_c(pdiff, ndiff, src_prev, src_curr, src_next);
  pmax = std::max(pmax, pdiff);
  nmax = std::max(nmax, ndiff);

  pmax = std::min(pmax, nmax);
  auto src_dest = dest;
  auto tmp_max = adds_c(src_curr, pmax);
  auto tmp_min = subs_c(src_curr, pmax);
  auto result = clip((int)src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode1_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  __m128i lowermax = _mm_undefined_si128();
  __m128i uppermax = _mm_undefined_si128();
  __m128i lower = _mm_undefined_si128();
  __m128i upper = _mm_undefined_si128();
  get_lu(lowermax, uppermax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get_lu(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  return SmoothTRepair1(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0);
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode2_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  __m128i lowermax = _mm_undefined_si128();
  __m128i uppermax = _mm_undefined_si128();
  __m128i lower = _mm_undefined_si128();
  __m128i upper = _mm_undefined_si128();
  get_lu(lowermax, uppermax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get_lu(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu(lower, upper, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  return SmoothTRepair2(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode3_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
  )
{
  __m128i pdiffmax = _mm_undefined_si128();
  __m128i ndiffmax = _mm_undefined_si128();
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();
  get2diff(pdiffmax, ndiffmax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get2diff(pdiff, ndiff, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff(pdiff, ndiff, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  return SmoothTRepair3(dp, pdiffmax, ndiffmax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

// unaligned, aligned
// FIXME: no difference at the moment
template<typename pixel_t, smooth_temporal_repair_processor_simd processor, smooth_temporal_repair_processor_simd processor_a>
void smooth_temporal_repair_mode1to3_sse2(BYTE* dp8, const BYTE* previous8, const BYTE* sp8, const BYTE* next8, intptr_t* pitches, int width, int height)
{
  // #   X    X
  // X new_dp X
  // X   X    X

  // memo: const intptr_t pitches[4] = { dppitch, pfpitch, ofpitch, nfpitch };

  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0] / sizeof(pixel_t);
  const intptr_t pfpitch = pitches[1] / sizeof(pixel_t);
  const intptr_t ofpitch = pitches[2] / sizeof(pixel_t);
  const intptr_t nfpitch = pitches[3] / sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* previous = reinterpret_cast<const pixel_t*>(previous8);
  const pixel_t* sp = reinterpret_cast<const pixel_t*>(sp8);
  const pixel_t* next = reinterpret_cast<const pixel_t*>(next8);

  const int pixels_at_a_time = 16 / sizeof(pixel_t);

  dp += dppitch;
  previous += pfpitch;
  sp += ofpitch;
  next += nfpitch;

  int mod_width = width / pixels_at_a_time * pixels_at_a_time;

  // top line copy: done by full frame copy 

  for (int y = 1; y < height - 1; ++y) {
    // pDst[0] = pSrc[0]; done by full frame copy

    // unaligned first 16 bytes, last pixel overlaps with the next aligned loop
    __m128i result = processor((uint8_t*)(dp + 1), (uint8_t*)(previous + 1), pfpitch, (uint8_t*)(sp + 1), ofpitch, (uint8_t*)(next + 1), nfpitch);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 1), result);

    // aligned
    for (int x = pixels_at_a_time; x < mod_width - 1; x += pixels_at_a_time) {
      __m128i result = processor_a((uint8_t*)(dp + x), (uint8_t*)(previous + x), pfpitch, (uint8_t*)(sp + x), ofpitch, (uint8_t*)(next + x), nfpitch);
      _mm_store_si128(reinterpret_cast<__m128i*>(dp + x), result);
    }

    if (mod_width != width) {
      const int xx = width - 1 - pixels_at_a_time;
      __m128i result = processor((uint8_t*)(dp + xx), (uint8_t*)(previous + xx), pfpitch, (uint8_t*)(sp + xx), ofpitch, (uint8_t*)(next + xx), nfpitch);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + xx), result);
    }

    // pDst[width - 1] = pSrc[width - 1]; done by full frame copy

    dp += dppitch;
    previous += pfpitch;
    sp += ofpitch;
    next += nfpitch;
  }
  // bottom line copy: done by full frame copy 

}

RG_FORCEINLINE int temporal_repair_processor_mode1_c(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  int lowermax, uppermax;
  int lower, upper;
  get_lu_c(lowermax, uppermax, previous[-1 * pfpitch - 1], sp[-1 * ofpitch - 1], next[-1 * nfpitch - 1]);
  get_lu_c(lower, upper, previous[-1 * pfpitch + 0], sp[-1 * ofpitch + 0], next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[-1 * pfpitch + 1], sp[-1 * ofpitch + 1], next[-1 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch - 1], sp[1 * ofpitch - 1], next[1 * nfpitch - 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch + 0], sp[1 * ofpitch + 0], next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch + 1], sp[1 * ofpitch + 1], next[1 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[0 * pfpitch - 1], sp[0 * ofpitch - 1], next[0 * nfpitch - 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[0 * pfpitch + 1], sp[0 * ofpitch + 1], next[0 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair1_c(dp[0], lowermax, uppermax, previous[0 * pfpitch + 0], sp[0 * ofpitch + 0], next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode2_c(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  int lowermax, uppermax;
  int lower, upper;
  get_lu_c(lowermax, uppermax, previous[-1 * pfpitch - 1], sp[-1 * ofpitch - 1], next[-1 * nfpitch - 1]);
  get_lu_c(lower, upper, previous[-1 * pfpitch + 0], sp[-1 * ofpitch + 0], next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[-1 * pfpitch + 1], sp[-1 * ofpitch + 1], next[-1 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch - 1], sp[1 * ofpitch - 1], next[1 * nfpitch - 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch + 0], sp[1 * ofpitch + 0], next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[1 * pfpitch + 1], sp[1 * ofpitch + 1], next[1 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[0 * pfpitch - 1], sp[0 * ofpitch - 1], next[0 * nfpitch - 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_c(lower, upper, previous[0 * pfpitch + 1], sp[0 * ofpitch + 1], next[0 * nfpitch + 1]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair2_c(dp[0], lowermax, uppermax, previous[0 * pfpitch + 0], sp[0 * ofpitch + 0], next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode3_c(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  int pdiffmax;
  int ndiffmax;
  int pdiff;
  int ndiff;
  get2diff_c(pdiffmax, ndiffmax, previous[-1 * pfpitch - 1], sp[-1 * ofpitch - 1], next[-1 * nfpitch - 1]);
  get2diff_c(pdiff, ndiff, previous[-1 * pfpitch + 0], sp[-1 * ofpitch + 0], next[-1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[-1 * pfpitch + 1], sp[-1 * ofpitch + 1], next[-1 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[1 * pfpitch - 1], sp[1 * ofpitch - 1], next[1 * nfpitch - 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[1 * pfpitch + 0], sp[1 * ofpitch + 0], next[1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[1 * pfpitch + 1], sp[1 * ofpitch + 1], next[1 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[0 * pfpitch - 1], sp[0 * ofpitch - 1], next[0 * nfpitch - 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_c(pdiff, ndiff, previous[0 * pfpitch + 1], sp[0 * ofpitch + 1], next[0 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  return SmoothTRepair3_c(dp[0], pdiffmax, ndiffmax, previous[0 * pfpitch + 0], sp[0 * ofpitch + 0], next[0 * nfpitch + 0]);
}


template<typename pixel_t, smooth_temporal_repair_processor_c processor>
void smooth_temporal_repair_mode1to3_c(BYTE* dp8, const BYTE* previous8, const BYTE* sp8, const BYTE* next8, intptr_t* pitches, int width, int height)
{
  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0] / sizeof(pixel_t);
  const intptr_t pfpitch = pitches[1] / sizeof(pixel_t);
  const intptr_t ofpitch = pitches[2] / sizeof(pixel_t);
  const intptr_t nfpitch = pitches[3] / sizeof(pixel_t);

  pixel_t* dp = reinterpret_cast<pixel_t*>(dp8);
  const pixel_t* previous = reinterpret_cast<const pixel_t*>(previous8);
  const pixel_t* sp = reinterpret_cast<const pixel_t*>(sp8);
  const pixel_t* next = reinterpret_cast<const pixel_t*>(next8);

  dp += dppitch;
  previous += pfpitch;
  sp += ofpitch;
  next += nfpitch;

  for (int y = 1; y < height - 1; y++)
  {
    for (int x = 1; x < width - 1; x++)
    {
      dp[x] = processor(&dp[x], &previous[x], pfpitch, &sp[x], ofpitch, &next[x], nfpitch);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

static PlaneProcessor_st* st_sse2_functions[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode1_sse2, temporal_repair_processor_mode1_sse2>,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode2_sse2, temporal_repair_processor_mode2_sse2>,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode3_sse2, temporal_repair_processor_mode3_sse2>
};

static PlaneProcessor_st* st_c_functions[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode1_c>,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode2_c>,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode3_c>
};

class SmoothTemporalRepair : public GenericVideoFilter
{
  PClip oclip;

  PlaneProcessor_st* processor_st_repair;
  PlaneProcessor_st* processor_st_repair_c;

  int last_frame;
  bool grey;
  
  // MT mode Registration for Avisynth+
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env)
  {
    if (n <= 0 || n - 1 >= last_frame)
      return child->GetFrame(n, env);
    PVideoFrame sf = child->GetFrame(n, env);
    PVideoFrame pf = oclip->GetFrame(n - 1, env);
    PVideoFrame of = oclip->GetFrame(n, env);
    PVideoFrame nf = oclip->GetFrame(n + 1, env);
    PVideoFrame df = env->NewVideoFrame(vi);

    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

    const int planecount = grey ? 1 : std::min(vi.NumComponents(), 3); // no Alpha plane processing
    for (int p = 0; p < planecount; ++p) {
      const int plane = planes[p];

      BYTE* dp = df->GetWritePtr(plane);
      int dppitch = df->GetPitch(plane);
      int pfpitch = pf->GetPitch(plane);
      int ofpitch = of->GetPitch(plane);
      int nfpitch = nf->GetPitch(plane);

      // copy the plane from sp to dp
      copy_plane(df, sf, plane, env);

      intptr_t pitches[4] = { dppitch, pfpitch, ofpitch, nfpitch };

      if (df->GetRowSize(plane) < 18)
        processor_st_repair_c(dp, pf->GetReadPtr(plane), of->GetReadPtr(plane), nf->GetReadPtr(plane),
          pitches, // pitch array
          df->GetRowSize(plane) / vi.ComponentSize(),
          df->GetHeight(plane));
      else
        // Spatial: Edge rows/columns are unhandled: -2 pixels less
        processor_st_repair(dp, pf->GetReadPtr(plane), of->GetReadPtr(plane), nf->GetReadPtr(plane),
          pitches, // pitch array
          df->GetRowSize(plane) / vi.ComponentSize(),
          df->GetHeight(plane));
    }

    if (vi.NumComponents() == 4)
      copy_plane(df, sf, PLANAR_A, env);

    return df;
  }
public:
  SmoothTemporalRepair(PClip clip, PClip _oclip, int mode, bool grey, bool planar, int opt, IScriptEnvironment* env) : 
    GenericVideoFilter(clip), 
    oclip(_oclip),
    grey(grey)
  {
    if (!planar && !vi.IsPlanar())
      env->ThrowError("TemporalRepair: only planar color spaces are supported");

    CompareVideoInfo(vi, _oclip->GetVideoInfo(), "TemporalRepair", env);

    if (vi.BitsPerComponent() > 8)
      env->ThrowError("TemporalRepair: only 8 bit color spaces are supported for this mode");

    if (vi.IsY())
      grey = true;

    // only mode 1, 2 and 3
    processor_st_repair_c = st_c_functions[mode];

    if (opt == 0)
      processor_st_repair = processor_st_repair_c;
    else
      processor_st_repair = st_sse2_functions[mode];

    last_frame = vi.num_frames >= 2 ? vi.num_frames - 2 : 0;

    /*FIXME: check min dimensions 3x3
      env->ThrowError("TemporalRepair: the width or height of the clip is too small");*/
  }
};

AVSValue __cdecl Create_TemporalRepair(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  constexpr int MAXTMODE = 4;

  enum ARGS { CLIP, OCLIP, MODE, SMOOTH, GREY, PLANAR, OPT };
  PClip clip = args[CLIP].AsClip();
  PClip oclip = args[OCLIP].AsClip();
  bool grey = args[GREY].AsBool(false);
  // mode and smooth are the same, probably for historical reasons.
  int mode = args[MODE].AsInt(args[SMOOTH].AsInt(0));
  if ((unsigned)mode > MAXTMODE) 
    env->ThrowError("TemporalRepair: illegal mode %i", mode);
  bool planar = args[PLANAR].AsBool(false);
  int opt = args[OPT].AsInt(-1);
  /*
  if (clip->GetVideoInfo().BitsPerComponent() > 8)
    env->ThrowError("TemporalRepair: only 8 bit colorspaces are supported");
    */
  bool spatial[MAXTMODE + 1] = { false, true, true, true, false };

  return spatial[mode] ? (AVSValue) new SmoothTemporalRepair(clip, oclip, mode, grey, planar, opt, env)
    : (AVSValue) new TemporalRepair(clip, oclip, mode, grey, planar, opt, env);

};
