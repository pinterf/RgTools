#ifndef __RG_FUNCTIONS_NEON_H__
#define __RG_FUNCTIONS_NEON_H__

// RemoveGrain aarch64 NEON kernels: one template per mode over the pixel traits of common_neon.h (8 bit, 9-16 bit,
// float). They follow the SSE4.1 kernels (rg_functions_sse.h) operation by operation, so the result is the same as
// SSE4.1 / AVX2 and C. pSrc: the center pixels, srcPitch: bytes.

#include "common_neon.h"

#ifdef NEON_INTRINSICS

#include <type_traits>

namespace {

template<class P> using NV = typename P::V;
template<class P> using NCtx = typename P::Ctx;

template<class P>
RG_FORCEINLINE NV<P> rg_mode1_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto mi = P::min(P::min(P::min(a1, a2), P::min(a3, a4)), P::min(P::min(a5, a6), P::min(a7, a8)));
  auto ma = P::max(P::max(P::max(a1, a2), P::max(a3, a4)), P::max(P::max(a5, a6), P::max(a7, a8)));
  return neon_clip<P>(c, mi, ma);
}

// modes 2-4: the common first part of the sorting network
#define RG_NEON_SORT_8 \
  auto b1 = a1, b2 = a2, b3 = a3, b4 = a4, b5 = a5, b6 = a6, b7 = a7, b8 = a8; \
  neon_sort_pair<P>(b1, b2); \
  neon_sort_pair<P>(b3, b4); \
  neon_sort_pair<P>(b5, b6); \
  neon_sort_pair<P>(b7, b8); \
  neon_sort_pair<P>(b1, b3); \
  neon_sort_pair<P>(b2, b4); \
  neon_sort_pair<P>(b5, b7); \
  neon_sort_pair<P>(b6, b8); \
  neon_sort_pair<P>(b2, b3); \
  neon_sort_pair<P>(b6, b7)

template<class P>
RG_FORCEINLINE NV<P> rg_mode2_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_SORT_8;
  b5 = P::max(b1, b5);
  neon_sort_pair<P>(b2, b6);
  neon_sort_pair<P>(b3, b7);
  b4 = P::min(b4, b8);
  b3 = P::min(b3, b5);
  b6 = P::max(b4, b6);
  b2 = P::min(b2, b3);
  b7 = P::max(b6, b7);
  return neon_clip<P>(c, b2, b7);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode3_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_SORT_8;
  b5 = P::max(b1, b5);
  neon_sort_pair<P>(b2, b6);
  neon_sort_pair<P>(b3, b7);
  b4 = P::min(b4, b8);
  b3 = P::min(b3, b5);
  b6 = P::max(b4, b6);
  b3 = P::max(b2, b3);
  b6 = P::min(b6, b7);
  return neon_clip<P>(c, b3, b6);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode4_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_SORT_8;
  b5 = P::max(b1, b5);
  b6 = P::max(b2, b6);
  b3 = P::min(b3, b7);
  b4 = P::min(b4, b8);
  b5 = P::max(b3, b5);
  b4 = P::min(b4, b6);
  neon_sort_pair<P>(b4, b5);
  return neon_clip<P>(c, b4, b5);
}

#undef RG_NEON_SORT_8

// the four lines through the center: max / min of the pairs (a1, a8), (a2, a7), (a3, a6), (a4, a5)
#define RG_NEON_LINES \
  auto mal1 = P::max(a1, a8); \
  auto mil1 = P::min(a1, a8); \
  auto mal2 = P::max(a2, a7); \
  auto mil2 = P::min(a2, a7); \
  auto mal3 = P::max(a3, a6); \
  auto mil3 = P::min(a3, a6); \
  auto mal4 = P::max(a4, a5); \
  auto mil4 = P::min(a4, a5)

template<class P>
RG_FORCEINLINE NV<P> rg_mode5_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_LINES;
  auto clipped1 = neon_clip<P>(c, mil1, mal1);
  auto clipped2 = neon_clip<P>(c, mil2, mal2);
  auto clipped3 = neon_clip<P>(c, mil3, mal3);
  auto clipped4 = neon_clip<P>(c, mil4, mal4);
  auto c1 = P::absdiff(c, clipped1);
  auto c2 = P::absdiff(c, clipped2);
  auto c3 = P::absdiff(c, clipped3);
  auto c4 = P::absdiff(c, clipped4);
  auto mindiff = P::min(P::min(P::min(c1, c2), c3), c4);
  auto result = P::sel_eq(mindiff, c1, c, clipped1);
  result = P::sel_eq(mindiff, c3, result, clipped3);
  result = P::sel_eq(mindiff, c2, result, clipped2);
  return P::sel_eq(mindiff, c4, result, clipped4);
}

// modes 6, 7, 8: the same frame, only the cost differs
// 6: 2 * |c - clipped| + range, 7: |c - clipped| + range, 8: |c - clipped| + 2 * range
template<class P, int mode>
RG_FORCEINLINE NV<P> rg_mode678_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_LINES;
  auto d1 = P::subs(mal1, mil1);
  auto d2 = P::subs(mal2, mil2);
  auto d3 = P::subs(mal3, mil3);
  auto d4 = P::subs(mal4, mil4);
  auto clipped1 = neon_clip<P>(c, mil1, mal1);
  auto clipped2 = neon_clip<P>(c, mil2, mal2);
  auto clipped3 = neon_clip<P>(c, mil3, mal3);
  auto clipped4 = neon_clip<P>(c, mil4, mal4);
  auto ad1 = P::absdiff(c, clipped1);
  auto ad2 = P::absdiff(c, clipped2);
  auto ad3 = P::absdiff(c, clipped3);
  auto ad4 = P::absdiff(c, clipped4);
  NV<P> c1, c2, c3, c4;
  if constexpr (mode == 6) {
    if constexpr (P::is_float) { // clamped to 1.0 like the saturating integer versions
      c1 = P::adds_diff(P::adds(ad1, ad1), d1);
      c2 = P::adds_diff(P::adds(ad2, ad2), d2);
      c3 = P::adds_diff(P::adds(ad3, ad3), d3);
      c4 = P::adds_diff(P::adds(ad4, ad4), d4);
    }
    else {
      c1 = P::clamp_max(P::adds(P::adds(ad1, ad1), d1), ctx);
      c2 = P::clamp_max(P::adds(P::adds(ad2, ad2), d2), ctx);
      c3 = P::clamp_max(P::adds(P::adds(ad3, ad3), d3), ctx);
      c4 = P::clamp_max(P::adds(P::adds(ad4, ad4), d4), ctx);
    }
  }
  else if constexpr (mode == 7) { // no clamp at the max pixel value (as C, SSE)
    c1 = P::adds(ad1, d1);
    c2 = P::adds(ad2, d2);
    c3 = P::adds(ad3, d3);
    c4 = P::adds(ad4, d4);
  }
  else { // 8
    c1 = P::clamp_max(P::adds(ad1, P::adds(d1, d1)), ctx);
    c2 = P::clamp_max(P::adds(ad2, P::adds(d2, d2)), ctx);
    c3 = P::clamp_max(P::adds(ad3, P::adds(d3, d3)), ctx);
    c4 = P::clamp_max(P::adds(ad4, P::adds(d4, d4)), ctx);
  }
  auto mindiff = P::min(P::min(P::min(c1, c2), c3), c4);
  auto result = P::sel_eq(mindiff, c1, c, clipped1);
  result = P::sel_eq(mindiff, c3, result, clipped3);
  result = P::sel_eq(mindiff, c2, result, clipped2);
  return P::sel_eq(mindiff, c4, result, clipped4);
}

template<class P> RG_FORCEINLINE NV<P> rg_mode6_neon(const uint8_t* p, int pitch, const NCtx<P>& ctx) { return rg_mode678_neon<P, 6>(p, pitch, ctx); }
template<class P> RG_FORCEINLINE NV<P> rg_mode7_neon(const uint8_t* p, int pitch, const NCtx<P>& ctx) { return rg_mode678_neon<P, 7>(p, pitch, ctx); }
template<class P> RG_FORCEINLINE NV<P> rg_mode8_neon(const uint8_t* p, int pitch, const NCtx<P>& ctx) { return rg_mode678_neon<P, 8>(p, pitch, ctx); }

template<class P>
RG_FORCEINLINE NV<P> rg_mode9_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_LINES;
  auto d1 = P::subs(mal1, mil1);
  auto d2 = P::subs(mal2, mil2);
  auto d3 = P::subs(mal3, mil3);
  auto d4 = P::subs(mal4, mil4);
  auto mindiff = P::min(P::min(P::min(d1, d2), d3), d4);
  auto result = P::sel_eq(mindiff, d1, c, neon_clip<P>(c, mil1, mal1));
  result = P::sel_eq(mindiff, d3, result, neon_clip<P>(c, mil3, mal3));
  result = P::sel_eq(mindiff, d2, result, neon_clip<P>(c, mil2, mal2));
  return P::sel_eq(mindiff, d4, result, neon_clip<P>(c, mil4, mal4));
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode10_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto d1 = P::absdiff(c, a1);
  auto d2 = P::absdiff(c, a2);
  auto d3 = P::absdiff(c, a3);
  auto d4 = P::absdiff(c, a4);
  auto d5 = P::absdiff(c, a5);
  auto d6 = P::absdiff(c, a6);
  auto d7 = P::absdiff(c, a7);
  auto d8 = P::absdiff(c, a8);
  auto mindiff = P::min(d1, d2);
  mindiff = P::min(mindiff, d3);
  mindiff = P::min(mindiff, d4);
  mindiff = P::min(mindiff, d5);
  mindiff = P::min(mindiff, d6);
  mindiff = P::min(mindiff, d7);
  mindiff = P::min(mindiff, d8);
  auto result = P::sel_eq(mindiff, d4, c, a4);
  result = P::sel_eq(mindiff, d5, result, a5);
  result = P::sel_eq(mindiff, d1, result, a1);
  result = P::sel_eq(mindiff, d3, result, a3);
  result = P::sel_eq(mindiff, d2, result, a2);
  result = P::sel_eq(mindiff, d6, result, a6);
  result = P::sel_eq(mindiff, d8, result, a8);
  return P::sel_eq(mindiff, d7, result, a7);
}

// mode 11: exact (4*c + 2*(a2+a4+a5+a7) + a1+a3+a6+a8 + 8) >> 4; 8 bit in 16 bit lanes, 9-16 bit in 32 bit lanes,
// float in the C operation order
static RG_FORCEINLINE uint16x8_t rg_mode11_sum_u16(uint16x8_t c, uint16x8_t a1, uint16x8_t a2, uint16x8_t a3, uint16x8_t a4,
  uint16x8_t a5, uint16x8_t a6, uint16x8_t a7, uint16x8_t a8) {
  auto edges = vaddq_u16(vaddq_u16(a2, a4), vaddq_u16(a5, a7));
  auto corners = vaddq_u16(vaddq_u16(a1, a3), vaddq_u16(a6, a8));
  auto sum = vaddq_u16(vshlq_n_u16(vaddq_u16(vshlq_n_u16(c, 1), edges), 1), corners);
  return vshrq_n_u16(vaddq_u16(sum, vdupq_n_u16(8)), 4);
}

static RG_FORCEINLINE uint32x4_t rg_mode11_sum_u32(uint32x4_t c, uint32x4_t a1, uint32x4_t a2, uint32x4_t a3, uint32x4_t a4,
  uint32x4_t a5, uint32x4_t a6, uint32x4_t a7, uint32x4_t a8) {
  auto edges = vaddq_u32(vaddq_u32(a2, a4), vaddq_u32(a5, a7));
  auto corners = vaddq_u32(vaddq_u32(a1, a3), vaddq_u32(a6, a8));
  auto sum = vaddq_u32(vshlq_n_u32(vaddq_u32(vshlq_n_u32(c, 1), edges), 1), corners);
  return vshrq_n_u32(vaddq_u32(sum, vdupq_n_u32(8)), 4);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode11_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  if constexpr (P::is_float) {
    auto edges = vaddq_f32(vaddq_f32(vaddq_f32(a2, a4), a5), a7);
    auto sum = vaddq_f32(vmulq_n_f32(c, 4.0f), vmulq_n_f32(edges, 2.0f));
    sum = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(sum, a1), a3), a6), a8);
    return vmulq_n_f32(sum, 1.0f / 16.0f);
  }
  else if constexpr (sizeof(typename P::T) == 1) {
    auto lo = rg_mode11_sum_u16(vmovl_u8(vget_low_u8(c)), vmovl_u8(vget_low_u8(a1)), vmovl_u8(vget_low_u8(a2)), vmovl_u8(vget_low_u8(a3)),
      vmovl_u8(vget_low_u8(a4)), vmovl_u8(vget_low_u8(a5)), vmovl_u8(vget_low_u8(a6)), vmovl_u8(vget_low_u8(a7)), vmovl_u8(vget_low_u8(a8)));
    auto hi = rg_mode11_sum_u16(vmovl_high_u8(c), vmovl_high_u8(a1), vmovl_high_u8(a2), vmovl_high_u8(a3),
      vmovl_high_u8(a4), vmovl_high_u8(a5), vmovl_high_u8(a6), vmovl_high_u8(a7), vmovl_high_u8(a8));
    return vcombine_u8(vqmovn_u16(lo), vqmovn_u16(hi));
  }
  else {
    auto lo = rg_mode11_sum_u32(vmovl_u16(vget_low_u16(c)), vmovl_u16(vget_low_u16(a1)), vmovl_u16(vget_low_u16(a2)), vmovl_u16(vget_low_u16(a3)),
      vmovl_u16(vget_low_u16(a4)), vmovl_u16(vget_low_u16(a5)), vmovl_u16(vget_low_u16(a6)), vmovl_u16(vget_low_u16(a7)), vmovl_u16(vget_low_u16(a8)));
    auto hi = rg_mode11_sum_u32(vmovl_high_u16(c), vmovl_high_u16(a1), vmovl_high_u16(a2), vmovl_high_u16(a3),
      vmovl_high_u16(a4), vmovl_high_u16(a5), vmovl_high_u16(a6), vmovl_high_u16(a7), vmovl_high_u16(a8));
    return vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
  }
}

// x - 1 (saturating) for the integer types: the rounding correction of the cascaded averages; float: none
template<class P> static RG_FORCEINLINE NV<P> rg_minus1(NV<P> x) {
  if constexpr (P::is_float) return x;
  else return P::subs(x, P::dup(1));
}

// mode 12: faster approximation of mode 11 (difference <= 1) with cascaded averages
template<class P>
RG_FORCEINLINE NV<P> rg_mode12_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto a13 = P::avg(a1, a3);
  auto a123 = P::avg(a2, a13);
  auto a68 = P::avg(a6, a8);
  auto a678 = P::avg(a7, a68);
  auto a45 = P::avg(a4, a5);
  auto a4c5 = P::avg(c, a45);
  auto a123678 = P::avg(a123, a678);
  return P::avg(a4c5, rg_minus1<P>(a123678));
}

// modes 13, 14 (field interpolation, the rows of the other field)
template<class P>
RG_FORCEINLINE NV<P> rg_mode13_and14_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto d1 = P::absdiff(a1, a8);
  auto d2 = P::absdiff(a2, a7);
  auto d3 = P::absdiff(a3, a6);
  auto mindiff = P::min(P::min(d1, d2), d3);
  auto result = P::sel_eq(mindiff, d1, c, P::avg(a1, a8));
  result = P::sel_eq(mindiff, d3, result, P::avg(a3, a6));
  return P::sel_eq(mindiff, d2, result, P::avg(a2, a7));
}

// modes 15, 16
template<class P>
RG_FORCEINLINE NV<P> rg_mode15_and16_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto max18 = P::max(a1, a8);
  auto min18 = P::min(a1, a8);
  auto max27 = P::max(a2, a7);
  auto min27 = P::min(a2, a7);
  auto max36 = P::max(a3, a6);
  auto min36 = P::min(a3, a6);
  auto d1 = P::subs(max18, min18);
  auto d2 = P::subs(max27, min27);
  auto d3 = P::subs(max36, min36);
  auto mindiff = P::min(P::min(d1, d2), d3);
  auto avg12 = P::avg(a1, a2);
  auto avg23 = P::avg(a2, a3);
  auto avg67 = P::avg(a6, a7);
  auto avg78 = P::avg(a7, a8);
  auto avg1223 = P::avg(avg12, avg23);
  auto avg6778 = P::avg(avg67, avg78);
  auto avg = P::avg(avg1223, rg_minus1<P>(avg6778));
  auto result = P::sel_eq(mindiff, d1, c, neon_clip<P>(avg, min18, max18));
  result = P::sel_eq(mindiff, d3, result, neon_clip<P>(avg, min36, max36));
  return P::sel_eq(mindiff, d2, result, neon_clip<P>(avg, min27, max27));
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode17_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_LINES;
  auto lower = P::max(P::max(P::max(mil1, mil2), mil3), mil4);
  auto upper = P::min(P::min(P::min(mal1, mal2), mal3), mal4);
  auto real_upper = P::max(upper, lower);
  auto real_lower = P::min(upper, lower);
  return neon_clip<P>(c, real_lower, real_upper);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode18_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto d1 = P::max(P::absdiff(c, a1), P::absdiff(c, a8));
  auto d2 = P::max(P::absdiff(c, a2), P::absdiff(c, a7));
  auto d3 = P::max(P::absdiff(c, a3), P::absdiff(c, a6));
  auto d4 = P::max(P::absdiff(c, a4), P::absdiff(c, a5));
  auto mindiff = P::min(P::min(P::min(d1, d2), d3), d4);
  auto c1 = neon_clip<P>(c, P::min(a1, a8), P::max(a1, a8));
  auto c2 = neon_clip<P>(c, P::min(a2, a7), P::max(a2, a7));
  auto c3 = neon_clip<P>(c, P::min(a3, a6), P::max(a3, a6));
  auto c4 = neon_clip<P>(c, P::min(a4, a5), P::max(a4, a5));
  auto result = P::sel_eq(mindiff, d1, c, c1);
  result = P::sel_eq(mindiff, d3, result, c3);
  result = P::sel_eq(mindiff, d2, result, c2);
  return P::sel_eq(mindiff, d4, result, c4);
}

// mode 19: (a1 + ... + a8) / 8 with cascaded averages
template<class P>
RG_FORCEINLINE NV<P> rg_mode19_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto a1368 = P::avg(P::avg(a1, a3), P::avg(a6, a8));
  auto a2457 = P::avg(P::avg(a2, a5), P::avg(a4, a7));
  return P::avg(rg_minus1<P>(a1368), a2457);
}

// mode 20: (sum + 4) / 9; 8 bit: 16 bit lanes, multiply-high with ((1 << 16) + 4) / 9 (as SSE _mm_mulhi_epu16);
// 9-16 bit: 32 bit lanes, ((sum + 4) * (((1 << 15) + 4) / 9)) >> 15, saturated to 16 bit; float: sum order as SSE, * 1/9
static RG_FORCEINLINE uint16x8_t rg_mode20_div9_u16(uint16x8_t sum) {
  const uint16x4_t onenineth = vdup_n_u16((uint16_t)(((1u << 16) + 4) / 9));
  auto lo = vshrn_n_u32(vmull_u16(vget_low_u16(sum), onenineth), 16);
  auto hi = vshrn_n_u32(vmull_high_u16(sum, vcombine_u16(onenineth, onenineth)), 16);
  return vcombine_u16(lo, hi);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode20_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  if constexpr (P::is_float) {
    auto a1234 = vaddq_f32(vaddq_f32(a1, a2), vaddq_f32(a3, a4));
    auto a5678 = vaddq_f32(vaddq_f32(a5, a6), vaddq_f32(a7, a8));
    auto val = vaddq_f32(vaddq_f32(a1234, a5678), c);
    return vmulq_n_f32(val, 1 / 9.0f);
  }
  else if constexpr (sizeof(typename P::T) == 1) {
    const uint16x8_t bias = vdupq_n_u16(4);
    auto lo = vaddq_u16(vaddq_u16(vaddq_u16(vaddl_u8(vget_low_u8(a1), vget_low_u8(a2)), vaddl_u8(vget_low_u8(a3), vget_low_u8(a4))),
      vaddq_u16(vaddl_u8(vget_low_u8(a5), vget_low_u8(a6)), vaddl_u8(vget_low_u8(a7), vget_low_u8(a8)))), vaddl_u8(vget_low_u8(c), vdup_n_u8(4)));
    auto hi = vaddq_u16(vaddq_u16(vaddq_u16(vaddl_high_u8(a1, a2), vaddl_high_u8(a3, a4)),
      vaddq_u16(vaddl_high_u8(a5, a6), vaddl_high_u8(a7, a8))), vaddq_u16(vmovl_high_u8(c), bias));
    return vcombine_u8(vqmovn_u16(rg_mode20_div9_u16(lo)), vqmovn_u16(rg_mode20_div9_u16(hi)));
  }
  else {
    constexpr uint32_t onenineth = ((1u << 15) + 4) / 9;
    const uint32x4_t bias = vdupq_n_u32(4);
    auto lo = vaddq_u32(vaddq_u32(vaddq_u32(vaddl_u16(vget_low_u16(a1), vget_low_u16(a2)), vaddl_u16(vget_low_u16(a3), vget_low_u16(a4))),
      vaddq_u32(vaddl_u16(vget_low_u16(a5), vget_low_u16(a6)), vaddl_u16(vget_low_u16(a7), vget_low_u16(a8)))), vaddq_u32(vmovl_u16(vget_low_u16(c)), bias));
    auto hi = vaddq_u32(vaddq_u32(vaddq_u32(vaddl_high_u16(a1, a2), vaddl_high_u16(a3, a4)),
      vaddq_u32(vaddl_high_u16(a5, a6), vaddl_high_u16(a7, a8))), vaddq_u32(vmovl_high_u16(c), bias));
    lo = vshrq_n_u32(vmulq_n_u32(lo, onenineth), 15);
    hi = vshrq_n_u32(vmulq_n_u32(hi, onenineth), 15);
    return vcombine_u16(vqmovn_u32(lo), vqmovn_u32(hi));
  }
}

// mode 22: the center clipped to the range of the line averages
template<class P>
RG_FORCEINLINE NV<P> rg_mode22_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto l1 = P::avg(a1, a8);
  auto l2 = P::avg(a2, a7);
  auto ma = P::max(l1, l2);
  auto mi = P::min(l1, l2);
  auto l3 = P::avg(a3, a6);
  ma = P::max(ma, l3);
  mi = P::min(mi, l3);
  auto l4 = P::avg(a4, a5);
  ma = P::max(ma, l4);
  mi = P::min(mi, l4);
  return neon_clip<P>(c, mi, ma);
}

// mode 21: as 22, with the rounded averages for the upper and the truncated ones for the lower limit; float: mode 22
template<class P>
RG_FORCEINLINE NV<P> rg_mode21_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>& ctx) {
  if constexpr (P::is_float) {
    return rg_mode22_neon<P>(pSrc, srcPitch, ctx);
  }
  else {
    LOAD_SQUARE_NEON(P, pSrc, srcPitch);
    auto l1a = P::avg_floor(a1, a8);
    auto l2a = P::avg_floor(a2, a7);
    auto l3a = P::avg_floor(a3, a6);
    auto l4a = P::avg_floor(a4, a5);
    auto l1b = P::avg(a1, a8);
    auto l2b = P::avg(a2, a7);
    auto l3b = P::avg(a3, a6);
    auto l4b = P::avg(a4, a5);
    auto ma = P::max(P::max(P::max(l1b, l2b), l3b), l4b);
    auto mi = P::min(P::min(P::min(l1a, l2a), l3a), l4a);
    return neon_clip<P>(c, mi, ma);
  }
}

// c - u + d: float within the pixel range of the plane (chroma: -0.5..0.5)
template<class P> static RG_FORCEINLINE NV<P> rg_sub_add(NV<P> c, NV<P> u, NV<P> d) {
  if constexpr (P::is_float) return P::adds_px(P::subs_px(c, u), d);
  else return P::adds(P::subs(c, u), d);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode23_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RG_NEON_LINES;
  auto linediff1 = P::subs(mal1, mil1);
  auto linediff2 = P::subs(mal2, mil2);
  auto linediff3 = P::subs(mal3, mil3);
  auto linediff4 = P::subs(mal4, mil4);
  auto u1 = P::min(P::subs(c, mal1), linediff1);
  auto u2 = P::min(P::subs(c, mal2), linediff2);
  auto u3 = P::min(P::subs(c, mal3), linediff3);
  auto u4 = P::min(P::subs(c, mal4), linediff4);
  auto u = P::max(P::max(P::max(u1, u2), u3), u4);
  auto d1 = P::min(P::subs(mil1, c), linediff1);
  auto d2 = P::min(P::subs(mil2, c), linediff2);
  auto d3 = P::min(P::subs(mil3, c), linediff3);
  auto d4 = P::min(P::subs(mil4, c), linediff4);
  auto d = P::max(P::max(P::max(d1, d2), d3), d4);
  return rg_sub_add<P>(c, u, d);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode24_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto line = [&](NV<P> x, NV<P> y, NV<P>& u, NV<P>& d) {
    auto mal = P::max(x, y);
    auto mil = P::min(x, y);
    auto diff = P::subs(mal, mil);
    auto temp = P::subs(c, mal);
    u = P::min(temp, P::subs(diff, temp));
    temp = P::subs(mil, c);
    d = P::min(temp, P::subs(diff, temp));
  };
  NV<P> u1, d1, u2, d2, u3, d3, u4, d4;
  line(a1, a8, u1, d1);
  line(a2, a7, u2, d2);
  auto d = P::max(d1, d2);
  auto u = P::max(u1, u2);
  line(a3, a6, u3, d3);
  d = P::max(d, d3);
  u = P::max(u, u3);
  line(a4, a5, u4, d4);
  d = P::max(d, d4);
  u = P::max(u, u4);
  return rg_sub_add<P>(c, u, d);
}

// mode 25 helpers. minus / plus: the difference to a smaller / larger neighbour; "no neighbour in this direction":
// all bits set (integer: 0xFF / 0xFFFF; float: 1.0); an equal neighbour: both 0 (no sharpening)
template<class P>
static RG_FORCEINLINE void rg_neighbourdiff_neon(NV<P>& minus, NV<P>& plus, NV<P> center, NV<P> neighbour) {
  if constexpr (P::is_float) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    auto cn_diff = P::subs(center, neighbour);
    auto nc_diff = P::subs(neighbour, center);
    auto cndiff_masked = vceqq_f32(cn_diff, zero);
    auto ncdiff_masked = vceqq_f32(nc_diff, zero);
    auto cn_equal = vandq_u32(cndiff_masked, ncdiff_masked);
    const float32x4_t max_mask = vdupq_n_f32(1.0f);
    minus = vbslq_f32(cndiff_masked, max_mask, cn_diff);
    plus = vbslq_f32(ncdiff_masked, max_mask, nc_diff);
    minus = vbslq_f32(cn_equal, zero, minus);
    plus = vbslq_f32(cn_equal, zero, plus);
  }
  else {
    const NV<P> zero = P::dup(0);
    auto cn_diff = P::subs(center, neighbour);
    auto nc_diff = P::subs(neighbour, center);
    auto cndiff_masked = P::eq_mask(cn_diff, zero);
    auto ncdiff_masked = P::eq_mask(nc_diff, zero);
    auto cn_equal = P::and_(cndiff_masked, ncdiff_masked);
    minus = P::subs(P::or_(cn_diff, cndiff_masked), cn_equal);
    plus = P::subs(P::or_(nc_diff, ncdiff_masked), cn_equal);
  }
}

template<class P>
static RG_FORCEINLINE NV<P> rg_sharpen_neon(NV<P> center, NV<P> minus, NV<P> plus, const NCtx<P>& ctx) {
  auto mp_diff = P::subs(minus, plus);
  auto pm_diff = P::subs(plus, minus);
  auto m_per2 = P::half(minus);
  auto p_per2 = P::half(plus);
  auto min_1 = P::min(p_per2, mp_diff);
  auto min_2 = P::min(m_per2, pm_diff);
  if constexpr (P::is_float)
    return P::subs_px(P::adds_px(center, min_1), min_2);
  else
    return P::subs(P::clamp_max(P::adds(center, min_1), ctx), min_2);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode25_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  NV<P> minus, plus, m, p;
  rg_neighbourdiff_neon<P>(minus, plus, c, a4);
  rg_neighbourdiff_neon<P>(m, p, c, a5); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a1); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a2); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a3); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a6); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a7); minus = P::min(minus, m); plus = P::min(plus, p);
  rg_neighbourdiff_neon<P>(m, p, c, a8); minus = P::min(minus, m); plus = P::min(plus, p);
  return rg_sharpen_neon<P>(c, minus, plus, ctx);
}

// modes 26-28: the center clipped to the range of neighbour pairs; acc: the pairs in the order of the SSE code
template<class P> struct RgPairRange {
  NV<P> lower, upper;
  RG_FORCEINLINE RgPairRange(NV<P> x, NV<P> y, NV<P> v, NV<P> w) : lower(P::max(P::min(x, y), P::min(v, w))), upper(P::min(P::max(x, y), P::max(v, w))) {}
  RG_FORCEINLINE void add(NV<P> x, NV<P> y) { lower = P::max(lower, P::min(x, y)); upper = P::min(upper, P::max(x, y)); }
  RG_FORCEINLINE NV<P> clip(NV<P> c) const { return neon_clip<P>(c, P::min(lower, upper), P::max(lower, upper)); }
};

template<class P>
RG_FORCEINLINE NV<P> rg_mode26_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RgPairRange<P> r(a1, a2, a2, a3);
  r.add(a3, a5);
  r.add(a5, a8);
  r.add(a7, a8);
  r.add(a6, a7);
  r.add(a4, a6);
  r.add(a1, a4);
  return r.clip(c);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode27_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RgPairRange<P> r(a1, a8, a1, a2);
  r.add(a7, a8);
  r.add(a2, a7);
  r.add(a2, a3);
  r.add(a6, a7);
  r.add(a3, a6);
  r.add(a3, a5);
  r.add(a4, a6);
  r.add(a4, a5);
  r.add(a5, a8);
  r.add(a1, a4);
  return r.clip(c);
}

template<class P>
RG_FORCEINLINE NV<P> rg_mode28_neon(const uint8_t* pSrc, int srcPitch, const NCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RgPairRange<P> r(a1, a2, a2, a3);
  r.add(a3, a5);
  r.add(a5, a8);
  r.add(a7, a8);
  r.add(a6, a7);
  r.add(a4, a6);
  r.add(a1, a4);
  r.add(a1, a8);
  r.add(a3, a6);
  r.add(a2, a7);
  r.add(a4, a5);
  return r.clip(c);
}

#undef RG_NEON_LINES

} // namespace

#endif // NEON_INTRINSICS

#endif // __RG_FUNCTIONS_NEON_H__
