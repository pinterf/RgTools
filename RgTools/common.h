#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdint.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)
#include "avs/config.h"

// INTEL_INTRINSICS: the x86 SIMD code is compiled. CMake defines it (ENABLE_INTEL_SIMD),
// the MSVC solution does not: derive it from the target there.
#if defined(_MSC_VER) && !defined(INTEL_INTRINSICS) && !defined(NEON_INTRINSICS) && (defined(_M_IX86) || defined(_M_X64))
#define INTEL_INTRINSICS
#endif

typedef uint8_t Byte;

#if defined(__clang__)
// Check clang first. clang-cl also defines __MSC_VER
// We set MSVC because they are mostly compatible
#   define CLANG
#if defined(_MSC_VER)
#   define MSVC
#   define RG_FORCEINLINE __attribute__((always_inline)) inline
#else
#   define RG_FORCEINLINE __attribute__((always_inline)) inline
#endif
#elif   defined(_MSC_VER)
#   define MSVC
#   define MSVC_PURE
#   define RG_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#   define GCC
#   define RG_FORCEINLINE __attribute__((always_inline)) inline
#else
#   error Unsupported compiler.
#   define RG_FORCEINLINE inline
#   undef __forceinline
#   define __forceinline inline
#endif 


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
static RG_FORCEINLINE int adds_16_c(int x, int y) {
  constexpr int pixel_max = (1 << bits_per_pixel) - 1;
  return std::min(pixel_max, x + y);
}

template<bool chroma>
static RG_FORCEINLINE float adds_32_c(float x, float y) {
  constexpr float pixel_max = chroma ? 0.5f : 1.0f;
  return std::min(pixel_max, x + y);
}

static RG_FORCEINLINE float adds_32_c_for_diff(float x, float y) {
  constexpr float pixel_max = 1.0f;
  return std::min(pixel_max, x + y);
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

// end of mode 25 helpers

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