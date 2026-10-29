// TemporalRepair: aarch64 NEON kernels, plane loops and function tables

#include "RemoveGrainT_core.h"
#include "common.h"

#ifdef NEON_INTRINSICS

#include "common_neon.h"

namespace {

// a + b / a - b for the clip bounds: integer saturating at the type limit (see RemoveGrainT_core.cpp: the bounds above
// the max pixel value give the same result), float within the pixel range of the plane (chroma: -0.5..0.5)
template<class P> static RG_FORCEINLINE typename P::V tr_up(typename P::V a, typename P::V b) {
  if constexpr (P::is_float) return P::adds_px(a, b);
  else return P::adds(a, b);
}
template<class P> static RG_FORCEINLINE typename P::V tr_dn(typename P::V a, typename P::V b) {
  if constexpr (P::is_float) return P::subs_px(a, b);
  else return P::subs(a, b);
}

/*************************************
  TemporalRepair, modes 0 and 4: per pixel
*************************************/

template<class P>
static RG_FORCEINLINE void tr_mode0_neon(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next) {
  auto src_next = P::load(next);
  auto src_prev = P::load(previous);
  auto src_2 = P::load(src2);
  auto min_np2 = P::min(P::min(src_next, src_prev), src_2);
  auto max_np2 = P::max(P::max(src_next, src_prev), src_2);
  P::store(dest, neon_clip<P>(P::load(src1), min_np2, max_np2));
}

template<class P>
static RG_FORCEINLINE void tr_mode4_neon(uint8_t* dest, const uint8_t* src1, const uint8_t* src2, const uint8_t* previous, const uint8_t* next) {
  auto src_next = P::load(next);
  auto src_prev = P::load(previous);
  auto src_1 = P::load(src1);
  auto src_2 = P::load(src2);
  auto max_np = P::max(src_next, src_prev);
  auto min_np = P::min(src_next, src_prev);
  auto diff_src2_minnp = P::subs(src_2, min_np);
  auto diff_maxnp_src2 = P::subs(max_np, src_2);
  typename P::V reg5, reg3;
  if constexpr (P::is_float) {
    reg5 = P::adds_px(P::adds_diff(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 + min_np
    reg3 = P::subs_px(max_np, P::adds_diff(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2
  }
  else {
    reg5 = P::adds(P::adds(diff_src2_minnp, diff_src2_minnp), min_np);
    reg3 = P::subs(max_np, P::adds(diff_maxnp_src2, diff_maxnp_src2));
  }
  reg5 = P::min(reg5, max_np);
  reg3 = P::max(reg3, min_np);
  // src_2 where min_np == reg5 or max_np == reg3
  auto result = P::sel_eq(min_np, reg5, neon_clip<P>(src_1, reg3, reg5), src_2);
  P::store(dest, P::sel_eq(max_np, reg3, result, src_2));
}

template<class P> using TrKernel = void(*)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*);

// all pixels, the last vector of a row overlaps the previous one (the minimum width is one vector)
template<class P, TrKernel<P> kernel>
static void temporal_repair_mode0and4_neon(uint8_t* dp, int dpitch, const uint8_t* sp1, int spitch1, const uint8_t* sp2, int spitch2, const uint8_t* pp, int ppitch, const uint8_t* np, int npitch, int width, int height) {
  constexpr int ps = (int)sizeof(typename P::T);
  constexpr int N = P::N;
  const int mod_width = width / N * N;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < mod_width; x += N)
      kernel(dp + x * ps, sp1 + x * ps, sp2 + x * ps, pp + x * ps, np + x * ps);
    if (mod_width != width) {
      const int x = width - N;
      kernel(dp + x * ps, sp1 + x * ps, sp2 + x * ps, pp + x * ps, np + x * ps);
    }
    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }
}

/*************************************
  SmoothTemporalRepair, modes 1, 2, 3: 3x3 neighbourhood of the previous, current and next frame of the original
*************************************/

// the distance of the current pixel to the range of previous and next
template<class P>
static RG_FORCEINLINE void tr_get_lu(typename P::V& lower, typename P::V& upper, const uint8_t* previous, const uint8_t* current, const uint8_t* next) {
  auto src_next = P::load(next);
  auto src_prev = P::load(previous);
  auto src_curr = P::load(current);
  upper = P::subs(P::max(src_next, src_prev), src_curr);
  lower = P::subs(src_curr, P::min(src_next, src_prev));
}

template<class P>
static RG_FORCEINLINE void tr_get2diff(typename P::V& pdiff, typename P::V& ndiff, const uint8_t* previous, const uint8_t* current, const uint8_t* next) {
  auto src_curr = P::load(current);
  pdiff = P::absdiff(src_curr, P::load(previous));
  ndiff = P::absdiff(src_curr, P::load(next));
}

// the 8 neighbours, in the order of the SSE code: (-1,-1) (-1,0) (-1,1) (1,-1) (1,0) (1,1) (0,-1) (0,1)
#define TR_NEON_NEIGHBOURS(F, A, B) \
  F(A, B, previous - pfpitch - ps, sp - ofpitch - ps, next - nfpitch - ps); \
  F(t1, t2, previous - pfpitch, sp - ofpitch, next - nfpitch); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous - pfpitch + ps, sp - ofpitch + ps, next - nfpitch + ps); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous + pfpitch - ps, sp + ofpitch - ps, next + nfpitch - ps); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous + pfpitch, sp + ofpitch, next + nfpitch); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous + pfpitch + ps, sp + ofpitch + ps, next + nfpitch + ps); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous - ps, sp - ps, next - ps); A = P::max(A, t1); B = P::max(B, t2); \
  F(t1, t2, previous + ps, sp + ps, next + ps); A = P::max(A, t1); B = P::max(B, t2)

template<class P, int mode>
static RG_FORCEINLINE typename P::V tr_smooth_neon(const uint8_t* dp, const uint8_t* previous, intptr_t pfpitch, const uint8_t* sp, intptr_t ofpitch, const uint8_t* next, intptr_t nfpitch) {
  constexpr int ps = (int)sizeof(typename P::T);
  typename P::V a, b, t1, t2;
  auto src_curr = P::load(sp);
  auto src_dest = P::load(dp);
  if constexpr (mode == 1) {
    TR_NEON_NEIGHBOURS(tr_get_lu<P>, a, b); // a: lower, b: upper
    auto tmp_max = P::max(P::max(tr_up<P>(b, src_curr), P::load(previous)), P::load(next));
    auto tmp_min = P::min(P::min(tr_dn<P>(src_curr, a), P::load(previous)), P::load(next));
    return neon_clip<P>(src_dest, tmp_min, tmp_max);
  }
  else if constexpr (mode == 2) {
    TR_NEON_NEIGHBOURS(tr_get_lu<P>, a, b);
    typename P::V lower1, upper1;
    tr_get_lu<P>(lower1, upper1, previous, sp, next);
    auto upperlowermax = P::max(P::max(b, upper1), P::max(a, lower1));
    return neon_clip<P>(src_dest, tr_dn<P>(src_curr, upperlowermax), tr_up<P>(src_curr, upperlowermax));
  }
  else {
    TR_NEON_NEIGHBOURS(tr_get2diff<P>, a, b); // a: pdiff max, b: ndiff max
    typename P::V pdiff, ndiff;
    tr_get2diff<P>(pdiff, ndiff, previous, sp, next);
    auto pmax = P::min(P::max(a, pdiff), P::max(b, ndiff));
    return neon_clip<P>(src_dest, tr_dn<P>(src_curr, pmax), tr_up<P>(src_curr, pmax));
  }
}

#undef TR_NEON_NEIGHBOURS

// rows 1..height-2, columns 1..width-2 in place on the copied plane (the border keeps the clip); the overlapping
// vectors clip already clipped pixels to the same bounds, the bounds do not depend on dp
template<class P, int mode>
static void smooth_temporal_repair_mode1to3_neon(uint8_t* dp, const uint8_t* previous, const uint8_t* sp, const uint8_t* next, intptr_t* pitches, int width, int height) {
  const intptr_t dppitch = pitches[0];
  const intptr_t pfpitch = pitches[1];
  const intptr_t ofpitch = pitches[2];
  const intptr_t nfpitch = pitches[3];
  constexpr int ps = (int)sizeof(typename P::T);
  constexpr int N = P::N;

  dp += dppitch;
  previous += pfpitch;
  sp += ofpitch;
  next += nfpitch;

  for (int y = 1; y < height - 1; ++y) {
    P::store(dp + ps, tr_smooth_neon<P, mode>(dp + ps, previous + ps, pfpitch, sp + ps, ofpitch, next + ps, nfpitch));
    // the vector and its right neighbours stay inside the row
    for (int x = N; x < width - N; x += N)
      P::store(dp + x * ps, tr_smooth_neon<P, mode>(dp + x * ps, previous + x * ps, pfpitch, sp + x * ps, ofpitch, next + x * ps, nfpitch));
    { // the last pixels before the right border
      const int x = width - 1 - N;
      P::store(dp + x * ps, tr_smooth_neon<P, mode>(dp + x * ps, previous + x * ps, pfpitch, sp + x * ps, ofpitch, next + x * ps, nfpitch));
    }
    dp += dppitch;
    previous += pfpitch;
    sp += ofpitch;
    next += nfpitch;
  }
}

} // namespace

// indexed by mode
#define TR_NEON_T_TABLE(P) { \
  temporal_repair_mode0and4_neon<P, tr_mode0_neon<P>>, nullptr, nullptr, nullptr, temporal_repair_mode0and4_neon<P, tr_mode4_neon<P>> }
#define TR_NEON_ST_TABLE(P) { \
  nullptr, smooth_temporal_repair_mode1to3_neon<P, 1>, smooth_temporal_repair_mode1to3_neon<P, 2>, smooth_temporal_repair_mode1to3_neon<P, 3> }

PlaneProcessor_t* t_neon_functions[] = TR_NEON_T_TABLE(NeonU8);
PlaneProcessor_t* t_neon_functions_16[] = TR_NEON_T_TABLE(NeonU16);
PlaneProcessor_t* t_neon_functions_32[] = TR_NEON_T_TABLE(NeonF32<false>);
PlaneProcessor_t* t_neon_functions_chroma_32[] = TR_NEON_T_TABLE(NeonF32<true>);
PlaneProcessor_st* st_neon_functions[] = TR_NEON_ST_TABLE(NeonU8);
PlaneProcessor_st* st_neon_functions_16[] = TR_NEON_ST_TABLE(NeonU16);
PlaneProcessor_st* st_neon_functions_32[] = TR_NEON_ST_TABLE(NeonF32<false>);
PlaneProcessor_st* st_neon_functions_chroma_32[] = TR_NEON_ST_TABLE(NeonF32<true>);

#undef TR_NEON_T_TABLE
#undef TR_NEON_ST_TABLE

#endif // NEON_INTRINSICS
