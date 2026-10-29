// TemporalRepair: x86 SSE2 / SSE4.1 kernels

#include "RemoveGrainT.h"
#include "common.h"

#ifdef INTEL_INTRINSICS

#include "common_sse.h"
#include "emmintrin.h"
#include "immintrin.h"

namespace {

typedef void (temporal_repair_processor_simd)(uint8_t* dp, const uint8_t* sp1, const uint8_t* sp2, const uint8_t* pp, const uint8_t* np);
typedef __m128i (smooth_temporal_repair_processor_simd)(BYTE* dp, const BYTE* previous, const intptr_t pfpitch, const BYTE* sp, const intptr_t ofpitch, const BYTE* next, const intptr_t nfpitch);

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

  auto diff_src2_minnp = _mm_subs_ps_for_diff(src_2, min_np); // not really subsat, ordinary difference, clamped to zero
  reg5 = _mm_adds_ps<chroma>(
    _mm_adds_ps_for_diff(diff_src2_minnp, diff_src2_minnp), // 2x
    min_np
    ); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = _mm_subs_ps_for_diff(max_np, src_2); // not really subsat, ordinary difference, clamped to zero
  reg3 = _mm_subs_ps<chroma>(max_np, 
    _mm_adds_ps_for_diff(diff_maxnp_src2, diff_maxnp_src2)  // saturated 2x
    ); // max_np - 2 * diff_maxnp_src2

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

RG_FORCEINLINE void get_lu_8_sse2(__m128i &lower, __m128i &upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i *>(previous));
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current));
  auto max_np = _mm_max_epu8(src_next, src_prev);
  auto min_np = _mm_min_epu8(src_next, src_prev);
  upper = _mm_subs_epu8(max_np, src_curr);
  lower = _mm_subs_epu8(src_curr, min_np);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE void get_lu_16_sse41(__m128i& lower, __m128i& upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto max_np = _mm_max_epu16(src_next, src_prev);
  auto min_np = _mm_min_epu16(src_next, src_prev);
  upper = _mm_subs_epu16(max_np, src_curr);
  lower = _mm_subs_epu16(src_curr, min_np);
}

RG_FORCEINLINE void get_lu_32_sse2(__m128& lower, __m128& upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_next = _mm_loadu_ps(reinterpret_cast<const float*>(next));
  auto src_prev = _mm_loadu_ps(reinterpret_cast<const float*>(previous));
  auto src_curr = _mm_loadu_ps(reinterpret_cast<const float*>(current));
  auto max_np = _mm_max_ps(src_next, src_prev);
  auto min_np = _mm_min_ps(src_next, src_prev);
  upper = _mm_subs_ps_for_diff(max_np, src_curr); // not really subsat, ordinary difference, clamped to zero
  lower = _mm_subs_ps_for_diff(src_curr, min_np);
}

RG_FORCEINLINE __m128i SmoothTRepair1_8_sse2(uint8_t *dest, __m128i &lower, __m128i &upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
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

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i SmoothTRepair1_16_sse41(uint8_t* dest, __m128i& lower, __m128i& upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  auto tmp_u = _mm_adds_epu16(upper, src_curr); // FIXME: use bits_per_pixel for adds
  auto tmp_l = _mm_subs_epu16(src_curr, lower);

  auto tmp_max = _mm_max_epu16(_mm_max_epu16(tmp_u, src_prev), src_next);
  auto tmp_min = _mm_min_epu16(_mm_min_epu16(tmp_l, src_prev), src_next);

  auto result = simd_clip_16(src_dest, tmp_min, tmp_max);

  return result;
}

template<bool chroma>
RG_FORCEINLINE __m128 SmoothTRepair1_32_sse2(uint8_t* dest, __m128& lower, __m128& upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_curr = _mm_loadu_ps(reinterpret_cast<const float*>(current));
  auto src_prev = _mm_loadu_ps(reinterpret_cast<const float*>(previous));
  auto src_next = _mm_loadu_ps(reinterpret_cast<const float*>(next));
  auto src_dest = _mm_loadu_ps(reinterpret_cast<const float*>(dest));

  auto tmp_u = _mm_adds_ps<chroma>(upper, src_curr); // really addsat
  auto tmp_l = _mm_subs_ps<chroma>(src_curr, lower); // really subsat

  auto tmp_max = _mm_max_ps(_mm_max_ps(tmp_u, src_prev), src_next);
  auto tmp_min = _mm_min_ps(_mm_min_ps(tmp_l, src_prev), src_next);

  auto result = simd_clip_32(src_dest, tmp_min, tmp_max);

  return result;
}

RG_FORCEINLINE __m128i SmoothTRepair2_8_sse2(uint8_t *dest, __m128i lower, __m128i upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i lower1 = _mm_undefined_si128();
  __m128i upper1 = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  get_lu_8_sse2(lower1, upper1, previous, current, next);
  auto uppermax = _mm_max_epu8(upper, upper1);
  auto lowermax = _mm_max_epu8(lower, lower1);
  auto upperlowermax = _mm_max_epu8(uppermax, lowermax);

  auto tmp_max = _mm_adds_epu8(src_curr, upperlowermax);
  auto tmp_min = _mm_subs_epu8(src_curr, upperlowermax);

  auto result = simd_clip(src_dest, tmp_min, tmp_max);
  return result;
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i SmoothTRepair2_16_sse41(uint8_t* dest, __m128i lower, __m128i upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  __m128i lower1 = _mm_undefined_si128();
  __m128i upper1 = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  get_lu_16_sse41(lower1, upper1, previous, current, next);
  auto uppermax = _mm_max_epu16(upper, upper1);
  auto lowermax = _mm_max_epu16(lower, lower1);
  auto upperlowermax = _mm_max_epu16(uppermax, lowermax);

  auto tmp_max = _mm_adds_epu16(src_curr, upperlowermax); // FIXME: use bits_per_pixel for adds
  auto tmp_min = _mm_subs_epu16(src_curr, upperlowermax);

  auto result = simd_clip_16(src_dest, tmp_min, tmp_max);
  return result;
}

template<bool chroma>
RG_FORCEINLINE __m128 SmoothTRepair2_32_sse2(uint8_t* dest, __m128 lower, __m128 upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  __m128 lower1 = _mm_undefined_ps();
  __m128 upper1 = _mm_undefined_ps();

  auto src_curr = _mm_loadu_ps(reinterpret_cast<const float*>(current));
  auto src_dest = _mm_loadu_ps(reinterpret_cast<const float*>(dest));

  get_lu_32_sse2(lower1, upper1, previous, current, next);
  auto uppermax = _mm_max_ps(upper, upper1);
  auto lowermax = _mm_max_ps(lower, lower1);
  auto upperlowermax = _mm_max_ps(uppermax, lowermax);

  auto tmp_max = _mm_adds_ps<chroma>(src_curr, upperlowermax);
  auto tmp_min = _mm_subs_ps<chroma>(src_curr, upperlowermax);

  auto result = simd_clip_32(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE void get2diff_8_sse2(__m128i &pdiff, __m128i &ndiff, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i *>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(next));

  pdiff = abs_diff(src_curr, src_prev); // abs(c-p)
  ndiff = abs_diff(src_curr, src_next); // abs(c-n)
}

RG_FORCEINLINE void get2diff_16_sse2(__m128i& pdiff, __m128i& ndiff, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));

  pdiff = abs_diff_16(src_curr, src_prev); // abs(c-p)
  ndiff = abs_diff_16(src_curr, src_next); // abs(c-n)
}

RG_FORCEINLINE void get2diff_32_sse2(__m128& pdiff, __m128& ndiff, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  auto src_curr = _mm_loadu_ps(reinterpret_cast<const float*>(current));
  auto src_prev = _mm_loadu_ps(reinterpret_cast<const float*>(previous));
  auto src_next = _mm_loadu_ps(reinterpret_cast<const float*>(next));

  pdiff = abs_diff_32(src_curr, src_prev); // abs(c-p)
  ndiff = abs_diff_32(src_curr, src_next); // abs(c-n)
}

RG_FORCEINLINE __m128i SmoothTRepair3_8_sse2(uint8_t *dest, __m128i &pmax, __m128i &nmax, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));

  get2diff_8_sse2(pdiff, ndiff, previous, current, next);
  pmax = _mm_max_epu8(pmax, pdiff);
  nmax = _mm_max_epu8(nmax, ndiff);

  pmax = _mm_min_epu8(pmax, nmax);
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dest));
  auto tmp_max = _mm_adds_epu8(src_curr, pmax);
  auto tmp_min = _mm_subs_epu8(src_curr, pmax);
  auto result = simd_clip(src_dest, tmp_min, tmp_max);
  return result;
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i SmoothTRepair3_16_sse41(uint8_t* dest, __m128i& pmax, __m128i& nmax, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));

  get2diff_16_sse2(pdiff, ndiff, previous, current, next);
  pmax = _mm_max_epu16(pmax, pdiff);
  nmax = _mm_max_epu16(nmax, ndiff);

  pmax = _mm_min_epu16(pmax, nmax);
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));
  auto tmp_max = _mm_adds_epu16(src_curr, pmax); // FIXME: use bits_per_pixel for adds
  auto tmp_min = _mm_subs_epu16(src_curr, pmax);
  auto result = simd_clip_16(src_dest, tmp_min, tmp_max);
  return result;
}

template<bool chroma>
RG_FORCEINLINE __m128 SmoothTRepair3_32_sse2(uint8_t* dest, __m128& pmax, __m128& nmax, const uint8_t* previous, const uint8_t* current, const uint8_t* next)
{
  __m128 pdiff = _mm_undefined_ps();
  __m128 ndiff = _mm_undefined_ps();

  auto src_curr = _mm_loadu_ps(reinterpret_cast<const float*>(current));

  get2diff_32_sse2(pdiff, ndiff, previous, current, next);
  pmax = _mm_max_ps(pmax, pdiff);
  nmax = _mm_max_ps(nmax, ndiff);

  pmax = _mm_min_ps(pmax, nmax);
  auto src_dest = _mm_loadu_ps(reinterpret_cast<const float*>(dest));
  auto tmp_max = _mm_adds_ps<chroma>(src_curr, pmax);
  auto tmp_min = _mm_subs_ps<chroma>(src_curr, pmax);
  auto result = simd_clip_32(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode1_8_sse2(
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
  get_lu_8_sse2(lowermax, uppermax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get_lu_8_sse2(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  return SmoothTRepair1_8_sse2(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0);
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i temporal_repair_processor_mode1_16_sse41(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(uint16_t);

  __m128i lowermax = _mm_undefined_si128();
  __m128i uppermax = _mm_undefined_si128();
  __m128i lower = _mm_undefined_si128();
  __m128i upper = _mm_undefined_si128();
  get_lu_16_sse41(lowermax, uppermax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get_lu_16_sse41(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  return SmoothTRepair1_16_sse41<bits_per_pixel>(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0);
}

template<bool chroma>
RG_FORCEINLINE __m128i temporal_repair_processor_mode1_32_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(float);

  __m128 lowermax = _mm_undefined_ps();
  __m128 uppermax = _mm_undefined_ps();
  __m128 lower = _mm_undefined_ps();
  __m128 upper = _mm_undefined_ps();
  get_lu_32_sse2(lowermax, uppermax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get_lu_32_sse2(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  return _mm_castps_si128(SmoothTRepair1_32_sse2<chroma>(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0));
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode2_8_sse2(
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
  get_lu_8_sse2(lowermax, uppermax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get_lu_8_sse2(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  get_lu_8_sse2(lower, upper, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  uppermax = _mm_max_epu8(uppermax, upper);
  lowermax = _mm_max_epu8(lowermax, lower);
  return SmoothTRepair2_8_sse2(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i temporal_repair_processor_mode2_16_sse41(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(uint16_t);

  __m128i lowermax = _mm_undefined_si128();
  __m128i uppermax = _mm_undefined_si128();
  __m128i lower = _mm_undefined_si128();
  __m128i upper = _mm_undefined_si128();
  get_lu_16_sse41(lowermax, uppermax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get_lu_16_sse41(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  get_lu_16_sse41(lower, upper, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  uppermax = _mm_max_epu16(uppermax, upper);
  lowermax = _mm_max_epu16(lowermax, lower);
  return SmoothTRepair2_16_sse41<bits_per_pixel>(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

template<bool chroma>
RG_FORCEINLINE __m128i temporal_repair_processor_mode2_32_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(float);

  __m128 lowermax = _mm_undefined_ps();
  __m128 uppermax = _mm_undefined_ps();
  __m128 lower = _mm_undefined_ps();
  __m128 upper = _mm_undefined_ps();
  get_lu_32_sse2(lowermax, uppermax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get_lu_32_sse2(lower, upper, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  get_lu_32_sse2(lower, upper, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  uppermax = _mm_max_ps(uppermax, upper);
  lowermax = _mm_max_ps(lowermax, lower);
  return _mm_castps_si128(SmoothTRepair2_32_sse2<chroma>(dp, lowermax, uppermax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0)); // dp points to the center already
}

RG_FORCEINLINE __m128i temporal_repair_processor_mode3_8_sse2(
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
  get2diff_8_sse2(pdiffmax, ndiffmax, previous - 1 * pfpitch - 1, sp - 1 * ofpitch - 1, next - 1 * nfpitch - 1);
  get2diff_8_sse2(pdiff, ndiff, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous - 1 * pfpitch + 1, sp - 1 * ofpitch + 1, next - 1 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous + 1 * pfpitch - 1, sp + 1 * ofpitch - 1, next + 1 * nfpitch - 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous + 1 * pfpitch + 1, sp + 1 * ofpitch + 1, next + 1 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous + 0 * pfpitch - 1, sp + 0 * ofpitch - 1, next + 0 * nfpitch - 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  get2diff_8_sse2(pdiff, ndiff, previous + 0 * pfpitch + 1, sp + 0 * ofpitch + 1, next + 0 * nfpitch + 1);
  pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
  return SmoothTRepair3_8_sse2(dp, pdiffmax, ndiffmax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
RG_FORCEINLINE __m128i temporal_repair_processor_mode3_16_sse41(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(uint16_t);

  __m128i pdiffmax = _mm_undefined_si128();
  __m128i ndiffmax = _mm_undefined_si128();
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();
  get2diff_16_sse2(pdiffmax, ndiffmax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get2diff_16_sse2(pdiff, ndiff, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  get2diff_16_sse2(pdiff, ndiff, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  pdiffmax = _mm_max_epu16(pdiffmax, pdiff);
  ndiffmax = _mm_max_epu16(ndiffmax, ndiff);
  return SmoothTRepair3_16_sse41<bits_per_pixel>(dp, pdiffmax, ndiffmax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0); // dp points to the center already
}

template<bool chroma>
RG_FORCEINLINE __m128i temporal_repair_processor_mode3_32_sse2(
  BYTE* dp,
  const BYTE* previous, const intptr_t pfpitch,
  const BYTE* sp, const intptr_t ofpitch,
  const BYTE* next, const intptr_t nfpitch
)
{
  constexpr int pixelsize = sizeof(float);

  __m128 pdiffmax = _mm_undefined_ps();
  __m128 ndiffmax = _mm_undefined_ps();
  __m128 pdiff = _mm_undefined_ps();
  __m128 ndiff = _mm_undefined_ps();
  get2diff_32_sse2(pdiffmax, ndiffmax, previous - 1 * pfpitch - pixelsize, sp - 1 * ofpitch - pixelsize, next - 1 * nfpitch - pixelsize);
  get2diff_32_sse2(pdiff, ndiff, previous - 1 * pfpitch + 0, sp - 1 * ofpitch + 0, next - 1 * nfpitch + 0);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous - 1 * pfpitch + pixelsize, sp - 1 * ofpitch + pixelsize, next - 1 * nfpitch + pixelsize);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous + 1 * pfpitch - pixelsize, sp + 1 * ofpitch - pixelsize, next + 1 * nfpitch - pixelsize);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous + 1 * pfpitch + 0, sp + 1 * ofpitch + 0, next + 1 * nfpitch + 0);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous + 1 * pfpitch + pixelsize, sp + 1 * ofpitch + pixelsize, next + 1 * nfpitch + pixelsize);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous + 0 * pfpitch - pixelsize, sp + 0 * ofpitch - pixelsize, next + 0 * nfpitch - pixelsize);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  get2diff_32_sse2(pdiff, ndiff, previous + 0 * pfpitch + pixelsize, sp + 0 * ofpitch + pixelsize, next + 0 * nfpitch + pixelsize);
  pdiffmax = _mm_max_ps(pdiffmax, pdiff);
  ndiffmax = _mm_max_ps(ndiffmax, ndiff);
  return _mm_castps_si128(SmoothTRepair3_32_sse2<chroma>(dp, pdiffmax, ndiffmax, previous + 0 * pfpitch + 0, sp + 0 * ofpitch + 0, next + 0 * nfpitch + 0)); // dp points to the center already
}

// unaligned, aligned
// FIXME: no difference at the moment
template<typename pixel_t, smooth_temporal_repair_processor_simd processor, smooth_temporal_repair_processor_simd processor_a>
void smooth_temporal_repair_mode1to3_sse2(BYTE* dp, const BYTE* previous, const BYTE* sp, const BYTE* next, intptr_t* pitches, int width, int height)
{
  // #   X    X
  // X new_dp X
  // X   X    X

  // memo: const intptr_t pitches[4] = { dppitch, pfpitch, ofpitch, nfpitch };

  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0];
  const intptr_t pfpitch = pitches[1];
  const intptr_t ofpitch = pitches[2];
  const intptr_t nfpitch = pitches[3];

  constexpr int pixelsize = sizeof(pixel_t);
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
    __m128i result = processor(dp + 1 * pixelsize, previous + 1 * pixelsize, pfpitch, sp + 1 * pixelsize, ofpitch, next + 1 * pixelsize, nfpitch);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 1 * pixelsize), result);

    // aligned
    for (int x = pixels_at_a_time; x < mod_width - 1; x += pixels_at_a_time) {
      __m128i result = processor_a(dp + x * pixelsize, previous + x * pixelsize, pfpitch, sp + x * pixelsize, ofpitch, next + x * pixelsize, nfpitch);
      _mm_store_si128(reinterpret_cast<__m128i*>(dp + x * pixelsize), result);
    }

    if (mod_width != width) {
      const int xx = width - 1 - pixels_at_a_time;
      __m128i result = processor(dp + xx * pixelsize, previous + xx * pixelsize, pfpitch, sp + xx * pixelsize, ofpitch, next + xx * pixelsize, nfpitch);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + xx * pixelsize), result);
    }

    // pDst[width - 1] = pSrc[width - 1]; done by full frame copy

    dp += dppitch;
    previous += pfpitch;
    sp += ofpitch;
    next += nfpitch;
  }
  // bottom line copy: done by full frame copy 

}

// same as SSE2. Different because of target attribute
// unaligned, aligned
// FIXME: no difference at the moment
template<typename pixel_t, smooth_temporal_repair_processor_simd processor, smooth_temporal_repair_processor_simd processor_a>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void smooth_temporal_repair_mode1to3_sse41(BYTE* dp, const BYTE* previous, const BYTE* sp, const BYTE* next, intptr_t* pitches, int width, int height)
{
  // #   X    X
  // X new_dp X
  // X   X    X

  // memo: const intptr_t pitches[4] = { dppitch, pfpitch, ofpitch, nfpitch };

  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0];
  const intptr_t pfpitch = pitches[1];
  const intptr_t ofpitch = pitches[2];
  const intptr_t nfpitch = pitches[3];

  constexpr int pixelsize = sizeof(pixel_t);
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
    __m128i result = processor(dp + 1 * pixelsize, previous + 1 * pixelsize, pfpitch, sp + 1 * pixelsize, ofpitch, next + 1 * pixelsize, nfpitch);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + 1 * pixelsize), result);

    // aligned
    for (int x = pixels_at_a_time; x < mod_width - 1; x += pixels_at_a_time) {
      __m128i result = processor_a(dp + x * pixelsize, previous + x * pixelsize, pfpitch, sp + x * pixelsize, ofpitch, next + x * pixelsize, nfpitch);
      _mm_store_si128(reinterpret_cast<__m128i*>(dp + x * pixelsize), result);
    }

    if (mod_width != width) {
      const int xx = width - 1 - pixels_at_a_time;
      __m128i result = processor(dp + xx * pixelsize, previous + xx * pixelsize, pfpitch, sp + xx * pixelsize, ofpitch, next + xx * pixelsize, nfpitch);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dp + xx * pixelsize), result);
    }

    // pDst[width - 1] = pSrc[width - 1]; done by full frame copy

    dp += dppitch;
    previous += pfpitch;
    sp += ofpitch;
    next += nfpitch;
  }
  // bottom line copy: done by full frame copy 

}

} // namespace

// 32 bit float chroma
PlaneProcessor_t* t_sse2_functions[] = {
    temporal_repair_mode0and4_sse2<uint8_t, RepairPixel_mode0_sse2, RepairPixel_mode0_sse2>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse2<uint8_t, BRepairPixel_mode4_sse2, BRepairPixel_mode4_sse2>
};

PlaneProcessor_t* t_sse41_functions[] = {
    temporal_repair_mode0and4_sse41<uint8_t, RepairPixel_mode0_sse2, RepairPixel_mode0_sse2>, // no 4.1 version
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<uint8_t, BRepairPixel_mode4_sse41, BRepairPixel_mode4_sse41>
};

PlaneProcessor_t* t_sse41_functions_16[] = {
    temporal_repair_mode0and4_sse41<uint16_t, RepairPixel_mode0_sse41_16, RepairPixel_mode0_sse41_16>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<uint16_t, BRepairPixel_mode4_sse41_16<16>, BRepairPixel_mode4_sse41_16<16>>
};

PlaneProcessor_t* t_sse41_functions_32[] = {
    temporal_repair_mode0and4_sse41<float, RepairPixel_mode0_sse2_32, RepairPixel_mode0_sse2_32>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<float, BRepairPixel_mode4_sse41_32<false>, BRepairPixel_mode4_sse41_32<false>>
};

PlaneProcessor_t* t_sse41_functions_chroma_32[] = {
    temporal_repair_mode0and4_sse41<float, RepairPixel_mode0_sse2_32, RepairPixel_mode0_sse2_32>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_sse41<float, BRepairPixel_mode4_sse41_32<true>, BRepairPixel_mode4_sse41_32<true>>
};

PlaneProcessor_st* st_sse2_functions[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode1_8_sse2, temporal_repair_processor_mode1_8_sse2>,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode2_8_sse2, temporal_repair_processor_mode2_8_sse2>,
    smooth_temporal_repair_mode1to3_sse2<uint8_t, temporal_repair_processor_mode3_8_sse2, temporal_repair_processor_mode3_8_sse2>
};

PlaneProcessor_st* st_sse41_functions_16[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_sse41<uint16_t, temporal_repair_processor_mode1_16_sse41<16>, temporal_repair_processor_mode1_16_sse41<16>>,
    smooth_temporal_repair_mode1to3_sse41<uint16_t, temporal_repair_processor_mode2_16_sse41<16>, temporal_repair_processor_mode2_16_sse41<16>>,
    smooth_temporal_repair_mode1to3_sse41<uint16_t, temporal_repair_processor_mode3_16_sse41<16>, temporal_repair_processor_mode3_16_sse41<16>>
};

PlaneProcessor_st* st_sse2_functions_32[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode1_32_sse2<false>, temporal_repair_processor_mode1_32_sse2<false>>,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode2_32_sse2<false>, temporal_repair_processor_mode2_32_sse2<false>>,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode3_32_sse2<false>, temporal_repair_processor_mode3_32_sse2<false>>
};

PlaneProcessor_st* st_sse2_functions_chroma_32[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode1_32_sse2<true>, temporal_repair_processor_mode1_32_sse2<true>>,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode2_32_sse2<true>, temporal_repair_processor_mode2_32_sse2<true>>,
    smooth_temporal_repair_mode1to3_sse2<float, temporal_repair_processor_mode3_32_sse2<true>, temporal_repair_processor_mode3_32_sse2<true>>
};

#endif // INTEL_INTRINSICS
