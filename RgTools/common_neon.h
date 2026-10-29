#ifndef __COMMON_NEON_H__
#define __COMMON_NEON_H__

// aarch64 NEON helpers: pixel type traits for the generic kernels (one template per mode for 8 bit, 9-16 bit and
// float). Each operation has the semantics of the SSE helper of the same role, so the NEON kernels give the same
// result as the SSE4.1 / AVX2 kernels and the C. No FMA: the float average is (a + b) * 0.5 as in SSE.

#include "common.h"

#ifdef NEON_INTRINSICS

#include <arm_neon.h>

namespace {

// 8 bit: the saturating operations stop at 255, the max pixel value
struct NeonU8 {
  using T = uint8_t;
  using V = uint8x16_t;
  static constexpr int N = 16;  // pixels per vector
  static constexpr bool is_float = false;
  struct Ctx {};
  static Ctx make_ctx(int /*bits_per_pixel*/) { return {}; }

  static RG_FORCEINLINE V load(const uint8_t* p) { return vld1q_u8(p); }
  static RG_FORCEINLINE void store(uint8_t* p, V v) { vst1q_u8(p, v); }
  static RG_FORCEINLINE V dup(int x) { return vdupq_n_u8((uint8_t)x); }
  static RG_FORCEINLINE V min(V a, V b) { return vminq_u8(a, b); }
  static RG_FORCEINLINE V max(V a, V b) { return vmaxq_u8(a, b); }
  static RG_FORCEINLINE V adds(V a, V b) { return vqaddq_u8(a, b); }      // _mm_adds_epu8
  static RG_FORCEINLINE V subs(V a, V b) { return vqsubq_u8(a, b); }      // _mm_subs_epu8
  static RG_FORCEINLINE V avg(V a, V b) { return vrhaddq_u8(a, b); }      // _mm_avg_epu8: (a + b + 1) >> 1
  static RG_FORCEINLINE V avg_floor(V a, V b) { return vhaddq_u8(a, b); } // not_rounded_average: (a + b) >> 1
  static RG_FORCEINLINE V absdiff(V a, V b) { return vabdq_u8(a, b); }
  static RG_FORCEINLINE V half(V a) { return vshrq_n_u8(a, 1); }
  static RG_FORCEINLINE V clamp_max(V a, const Ctx&) { return a; }       // the saturation is at the max already
  // cmp1 == cmp2 ? desired : current
  static RG_FORCEINLINE V sel_eq(V cmp1, V cmp2, V current, V desired) { return vbslq_u8(vceqq_u8(cmp1, cmp2), desired, current); }
  static RG_FORCEINLINE V eq_mask(V a, V b) { return vceqq_u8(a, b); }
  static RG_FORCEINLINE V or_(V a, V b) { return vorrq_u8(a, b); }
  static RG_FORCEINLINE V and_(V a, V b) { return vandq_u8(a, b); }
};

// 9-16 bit: the saturating operations stop at 0xFFFF, clamp_max limits to the max pixel value (set per plane)
struct NeonU16 {
  using T = uint16_t;
  using V = uint16x8_t;
  static constexpr int N = 8;
  static constexpr bool is_float = false;
  struct Ctx { uint16x8_t pixel_max; };
  static Ctx make_ctx(int bits_per_pixel) { return { vdupq_n_u16((uint16_t)((1u << bits_per_pixel) - 1)) }; }

  static RG_FORCEINLINE V load(const uint8_t* p) { return vld1q_u16(reinterpret_cast<const uint16_t*>(p)); }
  static RG_FORCEINLINE void store(uint8_t* p, V v) { vst1q_u16(reinterpret_cast<uint16_t*>(p), v); }
  static RG_FORCEINLINE V dup(int x) { return vdupq_n_u16((uint16_t)x); }
  static RG_FORCEINLINE V min(V a, V b) { return vminq_u16(a, b); }
  static RG_FORCEINLINE V max(V a, V b) { return vmaxq_u16(a, b); }
  static RG_FORCEINLINE V adds(V a, V b) { return vqaddq_u16(a, b); }
  static RG_FORCEINLINE V subs(V a, V b) { return vqsubq_u16(a, b); }
  static RG_FORCEINLINE V avg(V a, V b) { return vrhaddq_u16(a, b); }
  static RG_FORCEINLINE V avg_floor(V a, V b) { return vhaddq_u16(a, b); }
  static RG_FORCEINLINE V absdiff(V a, V b) { return vabdq_u16(a, b); }
  static RG_FORCEINLINE V half(V a) { return vshrq_n_u16(a, 1); }
  static RG_FORCEINLINE V clamp_max(V a, const Ctx& ctx) { return vminq_u16(a, ctx.pixel_max); }
  static RG_FORCEINLINE V sel_eq(V cmp1, V cmp2, V current, V desired) { return vbslq_u16(vceqq_u16(cmp1, cmp2), desired, current); }
  static RG_FORCEINLINE V eq_mask(V a, V b) { return vceqq_u16(a, b); }
  static RG_FORCEINLINE V or_(V a, V b) { return vorrq_u16(a, b); }
  static RG_FORCEINLINE V and_(V a, V b) { return vandq_u16(a, b); }
};

// 32 bit float. chroma: the pixel range of adds_px / subs_px is -0.5..0.5 instead of 0..1.
// adds / subs are the "for diff" variants of the SSE code: a plain add, and a subtraction limited to 0.
template<bool chroma>
struct NeonF32 {
  using T = float;
  using V = float32x4_t;
  static constexpr int N = 4;
  static constexpr bool is_float = true;
  struct Ctx {};
  static Ctx make_ctx(int /*bits_per_pixel*/) { return {}; }

  static RG_FORCEINLINE V load(const uint8_t* p) { return vld1q_f32(reinterpret_cast<const float*>(p)); }
  static RG_FORCEINLINE void store(uint8_t* p, V v) { vst1q_f32(reinterpret_cast<float*>(p), v); }
  static RG_FORCEINLINE V dupf(float x) { return vdupq_n_f32(x); }
  static RG_FORCEINLINE V min(V a, V b) { return vminq_f32(a, b); }
  static RG_FORCEINLINE V max(V a, V b) { return vmaxq_f32(a, b); }
  static RG_FORCEINLINE V adds(V a, V b) { return vaddq_f32(a, b); } // where the integer kernels saturate: a plain add
  static RG_FORCEINLINE V subs(V a, V b) { return vmaxq_f32(vsubq_f32(a, b), vdupq_n_f32(0.0f)); } // _mm_subs_ps_for_diff
  static RG_FORCEINLINE V adds_diff(V a, V b) { return vminq_f32(vaddq_f32(a, b), vdupq_n_f32(1.0f)); } // _mm_adds_ps_for_diff
  static RG_FORCEINLINE V subs_px(V a, V b) { return vmaxq_f32(vsubq_f32(a, b), vdupq_n_f32(chroma ? -0.5f : 0.0f)); } // _mm_subs_ps<chroma>
  static RG_FORCEINLINE V adds_px(V a, V b) { return vminq_f32(vaddq_f32(a, b), vdupq_n_f32(chroma ? 0.5f : 1.0f)); }  // _mm_adds_ps<chroma>
  static RG_FORCEINLINE V avg(V a, V b) { return vmulq_f32(vaddq_f32(a, b), vdupq_n_f32(0.5f)); } // _mm_avg_ps, no FMA
  static RG_FORCEINLINE V absdiff(V a, V b) { return vabdq_f32(a, b); } // |a - b|, as abs_diff_32
  static RG_FORCEINLINE V half(V a) { return vmulq_f32(a, vdupq_n_f32(0.5f)); }
  static RG_FORCEINLINE V clamp_max(V a, const Ctx&) { return a; }
  static RG_FORCEINLINE V sel_eq(V cmp1, V cmp2, V current, V desired) { return vbslq_f32(vceqq_f32(cmp1, cmp2), desired, current); }
};

template<class P> static RG_FORCEINLINE typename P::V neon_clip(typename P::V v, typename P::V lo, typename P::V hi) {
  return P::max(P::min(v, hi), lo);
}

template<class P> static RG_FORCEINLINE void neon_sort_pair(typename P::V& a, typename P::V& b) {
  const typename P::V tmp = P::min(a, b);
  b = P::max(a, b);
  a = tmp;
}

} // namespace

// the 3x3 neighbourhood of the pixels at ptr (byte pointer, byte pitch); a kernel uses only the neighbours it needs
#define LOAD_SQUARE_NEON(P, ptr, pitch) \
  constexpr int ps_ = (int)sizeof(typename P::T); \
  [[maybe_unused]] const typename P::V a1 = P::load((ptr) - (pitch) - ps_); \
  [[maybe_unused]] const typename P::V a2 = P::load((ptr) - (pitch)); \
  [[maybe_unused]] const typename P::V a3 = P::load((ptr) - (pitch) + ps_); \
  [[maybe_unused]] const typename P::V a4 = P::load((ptr) - ps_); \
  [[maybe_unused]] const typename P::V c  = P::load((ptr)); \
  [[maybe_unused]] const typename P::V a5 = P::load((ptr) + ps_); \
  [[maybe_unused]] const typename P::V a6 = P::load((ptr) + (pitch) - ps_); \
  [[maybe_unused]] const typename P::V a7 = P::load((ptr) + (pitch)); \
  [[maybe_unused]] const typename P::V a8 = P::load((ptr) + (pitch) + ps_)

#endif // NEON_INTRINSICS

#endif // __COMMON_NEON_H__
