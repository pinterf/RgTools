#ifndef __REPAIR_FUNCTIONS_NEON_H__
#define __REPAIR_FUNCTIONS_NEON_H__

// Repair aarch64 NEON kernels: one template per mode over the pixel traits of common_neon.h (8 bit, 9-16 bit,
// float), following the SSE4.1 kernels (repair_functions_sse.h) operation by operation: the same result as SSE4.1
// and C. pSrc: the center pixels of the reference clip, val: the pixels of the clip to repair, srcPitch: bytes.

#include "common_neon.h"

#ifdef NEON_INTRINSICS

namespace {

template<class P> using RV = typename P::V;
template<class P> using RCtx = typename P::Ctx;

// c -/+ x within the pixel range: integer saturating and limited to the max pixel value, float within the range
// of the plane (chroma: -0.5..0.5)
template<class P> static RG_FORCEINLINE RV<P> rp_lower(RV<P> c, RV<P> x) {
  if constexpr (P::is_float) return P::subs_px(c, x);
  else return P::subs(c, x);
}
template<class P> static RG_FORCEINLINE RV<P> rp_upper(RV<P> c, RV<P> x, const RCtx<P>& ctx) {
  if constexpr (P::is_float) return P::adds_px(c, x);
  else return P::clamp_max(P::adds(c, x), ctx);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode1_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto mi = P::min(P::min(P::min(P::min(a1, a2), P::min(a3, a4)), P::min(P::min(a5, a6), P::min(a7, a8))), c);
  auto ma = P::max(P::max(P::max(P::max(a1, a2), P::max(a3, a4)), P::max(P::max(a5, a6), P::max(a7, a8))), c);
  return neon_clip<P>(val, mi, ma);
}

// modes 2-4: the common first part of the sorting network of the 9 pixels (the center too)
#define RP_NEON_SORT_9 \
  auto b1 = a1, b2 = a2, b3 = a3, b4 = a4, b5 = a5, b6 = a6, b7 = a7, b8 = a8, bc = c; \
  neon_sort_pair<P>(b1, b8); \
  neon_sort_pair<P>(b1, bc); \
  neon_sort_pair<P>(b2, b5); \
  neon_sort_pair<P>(b3, b6); \
  neon_sort_pair<P>(b4, b7); \
  neon_sort_pair<P>(bc, b8); \
  neon_sort_pair<P>(b1, b3); \
  neon_sort_pair<P>(bc, b6); \
  neon_sort_pair<P>(b2, b4); \
  neon_sort_pair<P>(b5, b7); \
  neon_sort_pair<P>(b3, b8); \
  neon_sort_pair<P>(b3, bc); \
  neon_sort_pair<P>(b6, b8); \
  neon_sort_pair<P>(b4, b5)

template<class P>
RG_FORCEINLINE RV<P> repair_mode2_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_9;
  b2 = P::max(b1, b2);
  b3 = P::min(b3, b4);
  neon_sort_pair<P>(bc, b5);
  b7 = P::max(b6, b7);
  neon_sort_pair<P>(b2, b8);
  b2 = P::min(b2, bc);
  b8 = P::max(b5, b8);
  b2 = P::min(b2, b3);
  b7 = P::min(b7, b8);
  return neon_clip<P>(val, b2, b7);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode3_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_9;
  b2 = P::max(b1, b2);
  neon_sort_pair<P>(b3, b4);
  neon_sort_pair<P>(bc, b5);
  b6 = P::min(b6, b7);
  neon_sort_pair<P>(b2, b8);
  b2 = P::min(b2, bc);
  b6 = P::max(b4, b6);
  b5 = P::min(b5, b8);
  b3 = P::max(b2, b3);
  b6 = P::max(b5, b6);
  return neon_clip<P>(val, b3, b6);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode4_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_9;
  b2 = P::max(b1, b2);
  b4 = P::max(b3, b4);
  neon_sort_pair<P>(bc, b5);
  b6 = P::min(b6, b7);
  neon_sort_pair<P>(b2, b8);
  bc = P::max(b2, bc);
  neon_sort_pair<P>(b4, b6);
  b5 = P::min(b5, b8);
  b4 = P::min(b4, bc);
  b5 = P::min(b5, b6);
  return neon_clip<P>(val, b4, b5);
}

#undef RP_NEON_SORT_9

// the four lines through the center, with the center: max / min
#define RP_NEON_LINES_C \
  auto mal1 = P::max(P::max(a1, a8), c); \
  auto mil1 = P::min(P::min(a1, a8), c); \
  auto mal2 = P::max(P::max(a2, a7), c); \
  auto mil2 = P::min(P::min(a2, a7), c); \
  auto mal3 = P::max(P::max(a3, a6), c); \
  auto mil3 = P::min(P::min(a3, a6), c); \
  auto mal4 = P::max(P::max(a4, a5), c); \
  auto mil4 = P::min(P::min(a4, a5), c)

// the four lines without the center
#define RP_NEON_LINES \
  auto mal1 = P::max(a1, a8); \
  auto mil1 = P::min(a1, a8); \
  auto mal2 = P::max(a2, a7); \
  auto mil2 = P::min(a2, a7); \
  auto mal3 = P::max(a3, a6); \
  auto mil3 = P::min(a3, a6); \
  auto mal4 = P::max(a4, a5); \
  auto mil4 = P::min(a4, a5)

template<class P>
RG_FORCEINLINE RV<P> repair_mode5_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_LINES_C;
  auto clipped1 = neon_clip<P>(val, mil1, mal1);
  auto clipped2 = neon_clip<P>(val, mil2, mal2);
  auto clipped3 = neon_clip<P>(val, mil3, mal3);
  auto clipped4 = neon_clip<P>(val, mil4, mal4);
  auto c1 = P::absdiff(val, clipped1);
  auto c2 = P::absdiff(val, clipped2);
  auto c3 = P::absdiff(val, clipped3);
  auto c4 = P::absdiff(val, clipped4);
  auto mindiff = P::min(P::min(P::min(c1, c2), c3), c4);
  auto result = P::sel_eq(mindiff, c1, val, clipped1);
  result = P::sel_eq(mindiff, c3, result, clipped3);
  result = P::sel_eq(mindiff, c2, result, clipped2);
  return P::sel_eq(mindiff, c4, result, clipped4);
}

// modes 6, 7, 8: 6: 2 * |val - clipped| + range, 7: |val - clipped| + range, 8: |val - clipped| + 2 * range;
// float: plain adds (as SSE, C)
template<class P, int mode>
RG_FORCEINLINE RV<P> repair_mode678_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_LINES_C;
  auto d1 = P::subs(mal1, mil1);
  auto d2 = P::subs(mal2, mil2);
  auto d3 = P::subs(mal3, mil3);
  auto d4 = P::subs(mal4, mil4);
  auto clipped1 = neon_clip<P>(val, mil1, mal1);
  auto clipped2 = neon_clip<P>(val, mil2, mal2);
  auto clipped3 = neon_clip<P>(val, mil3, mal3);
  auto clipped4 = neon_clip<P>(val, mil4, mal4);
  auto ad1 = P::absdiff(val, clipped1);
  auto ad2 = P::absdiff(val, clipped2);
  auto ad3 = P::absdiff(val, clipped3);
  auto ad4 = P::absdiff(val, clipped4);
  RV<P> c1, c2, c3, c4;
  if constexpr (mode == 6) {
    c1 = P::clamp_max(P::adds(P::adds(ad1, ad1), d1), ctx);
    c2 = P::clamp_max(P::adds(P::adds(ad2, ad2), d2), ctx);
    c3 = P::clamp_max(P::adds(P::adds(ad3, ad3), d3), ctx);
    c4 = P::clamp_max(P::adds(P::adds(ad4, ad4), d4), ctx);
  }
  else if constexpr (mode == 7) {
    c1 = P::adds(ad1, d1);
    c2 = P::adds(ad2, d2);
    c3 = P::adds(ad3, d3);
    c4 = P::adds(ad4, d4);
  }
  else {
    c1 = P::clamp_max(P::adds(ad1, P::adds(d1, d1)), ctx);
    c2 = P::clamp_max(P::adds(ad2, P::adds(d2, d2)), ctx);
    c3 = P::clamp_max(P::adds(ad3, P::adds(d3, d3)), ctx);
    c4 = P::clamp_max(P::adds(ad4, P::adds(d4, d4)), ctx);
  }
  auto mindiff = P::min(P::min(P::min(c1, c2), c3), c4);
  auto result = P::sel_eq(mindiff, c1, val, clipped1);
  result = P::sel_eq(mindiff, c3, result, clipped3);
  result = P::sel_eq(mindiff, c2, result, clipped2);
  return P::sel_eq(mindiff, c4, result, clipped4);
}

template<class P> RG_FORCEINLINE RV<P> repair_mode6_neon(const uint8_t* p, RV<P> v, int pitch, const RCtx<P>& ctx) { return repair_mode678_neon<P, 6>(p, v, pitch, ctx); }
template<class P> RG_FORCEINLINE RV<P> repair_mode7_neon(const uint8_t* p, RV<P> v, int pitch, const RCtx<P>& ctx) { return repair_mode678_neon<P, 7>(p, v, pitch, ctx); }
template<class P> RG_FORCEINLINE RV<P> repair_mode8_neon(const uint8_t* p, RV<P> v, int pitch, const RCtx<P>& ctx) { return repair_mode678_neon<P, 8>(p, v, pitch, ctx); }

template<class P>
RG_FORCEINLINE RV<P> repair_mode9_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_LINES_C;
  auto d1 = P::subs(mal1, mil1);
  auto d2 = P::subs(mal2, mil2);
  auto d3 = P::subs(mal3, mil3);
  auto d4 = P::subs(mal4, mil4);
  auto mindiff = P::min(P::min(P::min(d1, d2), d3), d4);
  auto result = P::sel_eq(mindiff, d1, val, neon_clip<P>(val, mil1, mal1));
  result = P::sel_eq(mindiff, d3, result, neon_clip<P>(val, mil3, mal3));
  result = P::sel_eq(mindiff, d2, result, neon_clip<P>(val, mil2, mal2));
  return P::sel_eq(mindiff, d4, result, neon_clip<P>(val, mil4, mal4));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode10_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto d1 = P::absdiff(val, a1);
  auto d2 = P::absdiff(val, a2);
  auto d3 = P::absdiff(val, a3);
  auto d4 = P::absdiff(val, a4);
  auto d5 = P::absdiff(val, a5);
  auto d6 = P::absdiff(val, a6);
  auto d7 = P::absdiff(val, a7);
  auto d8 = P::absdiff(val, a8);
  auto dc = P::absdiff(val, c);
  auto mindiff = P::min(d1, d2);
  mindiff = P::min(mindiff, d3);
  mindiff = P::min(mindiff, d4);
  mindiff = P::min(mindiff, d5);
  mindiff = P::min(mindiff, d6);
  mindiff = P::min(mindiff, d7);
  mindiff = P::min(mindiff, d8);
  mindiff = P::min(mindiff, dc);
  auto result = P::sel_eq(mindiff, d4, c, a4);
  result = P::sel_eq(mindiff, dc, result, c);
  result = P::sel_eq(mindiff, d5, result, a5);
  result = P::sel_eq(mindiff, d1, result, a1);
  result = P::sel_eq(mindiff, d3, result, a3);
  result = P::sel_eq(mindiff, d2, result, a2);
  result = P::sel_eq(mindiff, d6, result, a6);
  result = P::sel_eq(mindiff, d8, result, a8);
  return P::sel_eq(mindiff, d7, result, a7);
}

// modes 12-14: the sorting network of the 8 neighbours (as RemoveGrain 2-4), the range extended to the center
#define RP_NEON_SORT_8 \
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
RG_FORCEINLINE RV<P> repair_mode12_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_8;
  b5 = P::max(b1, b5);
  neon_sort_pair<P>(b2, b6);
  neon_sort_pair<P>(b3, b7);
  b4 = P::min(b4, b8);
  b3 = P::min(b3, b5);
  b6 = P::max(b4, b6);
  b2 = P::min(b2, b3);
  b7 = P::max(b6, b7);
  return neon_clip<P>(val, P::min(c, b2), P::max(c, b7));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode13_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_8;
  b5 = P::max(b1, b5);
  neon_sort_pair<P>(b2, b6);
  neon_sort_pair<P>(b3, b7);
  b4 = P::min(b4, b8);
  b3 = P::min(b3, b5);
  b6 = P::max(b4, b6);
  b3 = P::max(b2, b3);
  b6 = P::min(b6, b7);
  return neon_clip<P>(val, P::min(c, b3), P::max(c, b6));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode14_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_SORT_8;
  b5 = P::max(b1, b5);
  b6 = P::max(b2, b6);
  b3 = P::min(b3, b7);
  b4 = P::min(b4, b8);
  b5 = P::max(b3, b5);
  b4 = P::min(b4, b6);
  neon_sort_pair<P>(b4, b5);
  return neon_clip<P>(val, P::min(c, b4), P::max(c, b5));
}

#undef RP_NEON_SORT_8

// modes 15, 16: the line is chosen by the center (as RemoveGrain 5, 6), val clipped to the line range with the center
template<class P, int mode>
RG_FORCEINLINE RV<P> repair_mode15_16_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_LINES;
  auto cma1 = P::max(c, mal1);
  auto cma2 = P::max(c, mal2);
  auto cma3 = P::max(c, mal3);
  auto cma4 = P::max(c, mal4);
  auto cmi1 = P::min(c, mil1);
  auto cmi2 = P::min(c, mil2);
  auto cmi3 = P::min(c, mil3);
  auto cmi4 = P::min(c, mil4);
  auto clipped1 = neon_clip<P>(c, mil1, mal1);
  auto clipped2 = neon_clip<P>(c, mil2, mal2);
  auto clipped3 = neon_clip<P>(c, mil3, mal3);
  auto clipped4 = neon_clip<P>(c, mil4, mal4);
  RV<P> c1, c2, c3, c4;
  if constexpr (mode == 15) {
    c1 = P::absdiff(c, clipped1);
    c2 = P::absdiff(c, clipped2);
    c3 = P::absdiff(c, clipped3);
    c4 = P::absdiff(c, clipped4);
  }
  else {
    auto d1 = P::subs(mal1, mil1);
    auto d2 = P::subs(mal2, mil2);
    auto d3 = P::subs(mal3, mil3);
    auto d4 = P::subs(mal4, mil4);
    auto ad1 = P::absdiff(c, clipped1);
    auto ad2 = P::absdiff(c, clipped2);
    auto ad3 = P::absdiff(c, clipped3);
    auto ad4 = P::absdiff(c, clipped4);
    c1 = P::clamp_max(P::adds(P::adds(ad1, ad1), d1), ctx);
    c2 = P::clamp_max(P::adds(P::adds(ad2, ad2), d2), ctx);
    c3 = P::clamp_max(P::adds(P::adds(ad3, ad3), d3), ctx);
    c4 = P::clamp_max(P::adds(P::adds(ad4, ad4), d4), ctx);
  }
  auto mindiff = P::min(P::min(P::min(c1, c2), c3), c4);
  auto result = P::sel_eq(mindiff, c1, val, neon_clip<P>(val, cmi1, cma1));
  result = P::sel_eq(mindiff, c3, result, neon_clip<P>(val, cmi3, cma3));
  result = P::sel_eq(mindiff, c2, result, neon_clip<P>(val, cmi2, cma2));
  return P::sel_eq(mindiff, c4, result, neon_clip<P>(val, cmi4, cma4));
}

template<class P> RG_FORCEINLINE RV<P> repair_mode15_neon(const uint8_t* p, RV<P> v, int pitch, const RCtx<P>& ctx) { return repair_mode15_16_neon<P, 15>(p, v, pitch, ctx); }
template<class P> RG_FORCEINLINE RV<P> repair_mode16_neon(const uint8_t* p, RV<P> v, int pitch, const RCtx<P>& ctx) { return repair_mode15_16_neon<P, 16>(p, v, pitch, ctx); }

template<class P>
RG_FORCEINLINE RV<P> repair_mode17_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RP_NEON_LINES;
  auto lower = P::max(P::max(P::max(mil1, mil2), mil3), mil4);
  auto upper = P::min(P::min(P::min(mal1, mal2), mal3), mal4);
  auto real_upper = P::max(P::max(upper, lower), c);
  auto real_lower = P::min(P::min(upper, lower), c);
  return neon_clip<P>(val, real_lower, real_upper);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode18_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto d1 = P::max(P::absdiff(c, a1), P::absdiff(c, a8));
  auto d2 = P::max(P::absdiff(c, a2), P::absdiff(c, a7));
  auto d3 = P::max(P::absdiff(c, a3), P::absdiff(c, a6));
  auto d4 = P::max(P::absdiff(c, a4), P::absdiff(c, a5));
  auto mindiff = P::min(P::min(P::min(d1, d2), d3), d4);
  auto c1 = neon_clip<P>(val, P::min(c, P::min(a1, a8)), P::max(c, P::max(a1, a8)));
  auto c2 = neon_clip<P>(val, P::min(c, P::min(a2, a7)), P::max(c, P::max(a2, a7)));
  auto c3 = neon_clip<P>(val, P::min(c, P::min(a3, a6)), P::max(c, P::max(a3, a6)));
  auto c4 = neon_clip<P>(val, P::min(c, P::min(a4, a5)), P::max(c, P::max(a4, a5)));
  auto result = P::sel_eq(mindiff, d1, val, c1);
  result = P::sel_eq(mindiff, d3, result, c3);
  result = P::sel_eq(mindiff, d2, result, c2);
  return P::sel_eq(mindiff, d4, result, c4);
}

// the smallest / the second smallest of the 8 differences to x (modes 19-20, 22-23)
template<class P> static RG_FORCEINLINE RV<P> rp_min8(RV<P> x, RV<P> a1, RV<P> a2, RV<P> a3, RV<P> a4, RV<P> a5, RV<P> a6, RV<P> a7, RV<P> a8) {
  auto mindiff = P::min(P::absdiff(x, a1), P::absdiff(x, a2));
  mindiff = P::min(mindiff, P::absdiff(x, a3));
  mindiff = P::min(mindiff, P::absdiff(x, a4));
  mindiff = P::min(mindiff, P::absdiff(x, a5));
  mindiff = P::min(mindiff, P::absdiff(x, a6));
  mindiff = P::min(mindiff, P::absdiff(x, a7));
  return P::min(mindiff, P::absdiff(x, a8));
}

template<class P> static RG_FORCEINLINE RV<P> rp_second8(RV<P> x, RV<P> a1, RV<P> a2, RV<P> a3, RV<P> a4, RV<P> a5, RV<P> a6, RV<P> a7, RV<P> a8) {
  auto d1 = P::absdiff(x, a1);
  auto d2 = P::absdiff(x, a2);
  auto mindiff = P::min(d1, d2);
  auto maxdiff = P::max(d1, d2);
  const RV<P> d[6] = { P::absdiff(x, a3), P::absdiff(x, a4), P::absdiff(x, a5), P::absdiff(x, a6), P::absdiff(x, a7), P::absdiff(x, a8) };
  for (int i = 0; i < 5; ++i) {
    maxdiff = neon_clip<P>(maxdiff, mindiff, d[i]);
    mindiff = P::min(mindiff, d[i]);
  }
  return neon_clip<P>(maxdiff, mindiff, d[5]);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode19_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto mindiff = rp_min8<P>(c, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(val, rp_lower<P>(c, mindiff), rp_upper<P>(c, mindiff, ctx));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode20_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto maxdiff = rp_second8<P>(c, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(val, rp_lower<P>(c, maxdiff), rp_upper<P>(c, maxdiff, ctx));
}

// the smallest of the line ranges around x (modes 21, 24)
template<class P> static RG_FORCEINLINE RV<P> rp_line_u(RV<P> x, RV<P> a1, RV<P> a2, RV<P> a3, RV<P> a4, RV<P> a5, RV<P> a6, RV<P> a7, RV<P> a8) {
  RP_NEON_LINES;
  auto u1 = P::max(P::subs(mal1, x), P::subs(x, mil1));
  auto u2 = P::max(P::subs(mal2, x), P::subs(x, mil2));
  auto u3 = P::max(P::subs(mal3, x), P::subs(x, mil3));
  auto u4 = P::max(P::subs(mal4, x), P::subs(x, mil4));
  return P::min(P::min(P::min(u1, u2), u3), u4);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode21_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto u = rp_line_u<P>(c, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(val, rp_lower<P>(c, u), rp_upper<P>(c, u, ctx));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode22_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto mindiff = rp_min8<P>(val, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(c, rp_lower<P>(val, mindiff), rp_upper<P>(val, mindiff, ctx));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode23_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto maxdiff = rp_second8<P>(val, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(c, rp_lower<P>(val, maxdiff), rp_upper<P>(val, maxdiff, ctx));
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode24_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>& ctx) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  auto u = rp_line_u<P>(val, a1, a2, a3, a4, a5, a6, a7, a8);
  return neon_clip<P>(c, rp_lower<P>(val, u), rp_upper<P>(val, u, ctx));
}

// modes 26-28: val clipped to the range of neighbour pairs extended to the center
template<class P> struct RpPairRange {
  RV<P> lower, upper;
  RG_FORCEINLINE RpPairRange(RV<P> x, RV<P> y, RV<P> v, RV<P> w) : lower(P::max(P::min(x, y), P::min(v, w))), upper(P::min(P::max(x, y), P::max(v, w))) {}
  RG_FORCEINLINE void add(RV<P> x, RV<P> y) { lower = P::max(lower, P::min(x, y)); upper = P::min(upper, P::max(x, y)); }
  RG_FORCEINLINE RV<P> clip(RV<P> val, RV<P> c) const { return neon_clip<P>(val, P::min(P::min(upper, lower), c), P::max(P::max(upper, lower), c)); }
};

template<class P>
RG_FORCEINLINE RV<P> repair_mode26_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RpPairRange<P> r(a1, a2, a2, a3);
  r.add(a3, a5);
  r.add(a5, a8);
  r.add(a7, a8);
  r.add(a6, a7);
  r.add(a4, a6);
  r.add(a1, a4);
  return r.clip(val, c);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode27_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RpPairRange<P> r(a1, a8, a1, a2);
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
  return r.clip(val, c);
}

template<class P>
RG_FORCEINLINE RV<P> repair_mode28_neon(const uint8_t* pSrc, RV<P> val, int srcPitch, const RCtx<P>&) {
  LOAD_SQUARE_NEON(P, pSrc, srcPitch);
  RpPairRange<P> r(a1, a2, a2, a3);
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
  return r.clip(val, c);
}

#undef RP_NEON_LINES
#undef RP_NEON_LINES_C

} // namespace

#endif // NEON_INTRINSICS

#endif // __REPAIR_FUNCTIONS_NEON_H__
