#ifndef __REPAIR_FUNCTIONS_C_H__
#define __REPAIR_FUNCTIONS_C_H__

#include "common.h"
#include <array>
#include <algorithm>  

template<typename pixel_t>
using CModeProcessor = pixel_t (*)(const Byte*, pixel_t val, int);

RG_FORCEINLINE Byte repair_mode1_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte mi = std::min(std::min (
        std::min(std::min(a1, a2), std::min(a3, a4)),
        std::min(std::min(a5, a6), std::min(a7, a8))
        ), c);
    Byte ma = std::max(std::max(
        std::max(std::max(a1, a2), std::max(a3, a4)),
        std::max(std::max(a5, a6), std::max(a7, a8))
        ), c);

    return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode1_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t mi = std::min(std::min (
    std::min(std::min(a1, a2), std::min(a3, a4)),
    std::min(std::min(a5, a6), std::min(a7, a8))
  ), c);
  uint16_t ma = std::max(std::max(
    std::max(std::max(a1, a2), std::max(a3, a4)),
    std::max(std::max(a5, a6), std::max(a7, a8))
  ), c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode1_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float mi = std::min(std::min (
    std::min(std::min(a1, a2), std::min(a3, a4)),
    std::min(std::min(a5, a6), std::min(a7, a8))
  ), c);
  float ma = std::max(std::max(
    std::max(std::max(a1, a2), std::max(a3, a4)),
    std::max(std::max(a5, a6), std::max(a7, a8))
  ), c);

  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode2_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));

    return clip(val, a[1], a[7]);
}

RG_FORCEINLINE uint16_t repair_mode2_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_16(val, a[1], a[7]);
}

RG_FORCEINLINE float repair_mode2_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_32(val, a[1], a[7]);
}

// ------------

RG_FORCEINLINE Byte repair_mode3_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));

    return clip(val, a[2], a[6]);
}

RG_FORCEINLINE uint16_t repair_mode3_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_16(val, a[2], a[6]);
}

RG_FORCEINLINE float repair_mode3_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_32(val, a[2], a[6]);
}

// ------------

RG_FORCEINLINE Byte repair_mode4_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));

    return clip(val, a[3], a[5]);
}

RG_FORCEINLINE uint16_t repair_mode4_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_16(val, a[3], a[5]);
}

RG_FORCEINLINE float repair_mode4_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 9> a = { a1, a2, a3, a4, c, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));

  return clip_32(val, a[3], a[5]);
}

// ------------

RG_FORCEINLINE Byte repair_mode5_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(std::max(a1, a8), c);
    auto mil1 = std::min(std::min(a1, a8), c);

    auto mal2 = std::max(std::max(a2, a7), c);
    auto mil2 = std::min(std::min(a2, a7), c);

    auto mal3 = std::max(std::max(a3, a6), c);
    auto mil3 = std::min(std::min(a3, a6), c);

    auto mal4 = std::max(std::max(a4, a5), c);
    auto mil4 = std::min(std::min(a4, a5), c);

    Byte c1 = std::abs(val-clip(val, mil1, mal1));
    Byte c2 = std::abs(val-clip(val, mil2, mal2));
    Byte c3 = std::abs(val-clip(val, mil3, mal3));
    Byte c4 = std::abs(val-clip(val, mil4, mal4));

    auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clip(val, mil4, mal4);
    if (mindiff == c2) return clip(val, mil2, mal2);
    if (mindiff == c3) return clip(val, mil3, mal3);
    return clip(val, mil1, mal1);
}

RG_FORCEINLINE uint16_t repair_mode5_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  uint16_t c1 = std::abs(val-clip_16(val, mil1, mal1));
  uint16_t c2 = std::abs(val-clip_16(val, mil2, mal2));
  uint16_t c3 = std::abs(val-clip_16(val, mil3, mal3));
  uint16_t c4 = std::abs(val-clip_16(val, mil4, mal4));

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_16(val, mil4, mal4);
  if (mindiff == c2) return clip_16(val, mil2, mal2);
  if (mindiff == c3) return clip_16(val, mil3, mal3);
  return clip_16(val, mil1, mal1);
}


RG_FORCEINLINE float repair_mode5_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  float c1 = std::abs(val-clip_32(val, mil1, mal1));
  float c2 = std::abs(val-clip_32(val, mil2, mal2));
  float c3 = std::abs(val-clip_32(val, mil3, mal3));
  float c4 = std::abs(val-clip_32(val, mil4, mal4));

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_32(val, mil4, mal4);
  if (mindiff == c2) return clip_32(val, mil2, mal2);
  if (mindiff == c3) return clip_32(val, mil3, mal3);
  return clip_32(val, mil1, mal1);
}

// ------------

RG_FORCEINLINE Byte repair_mode6_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(std::max(a1, a8), c);
    auto mil1 = std::min(std::min(a1, a8), c);

    auto mal2 = std::max(std::max(a2, a7), c);
    auto mil2 = std::min(std::min(a2, a7), c);

    auto mal3 = std::max(std::max(a3, a6), c);
    auto mil3 = std::min(std::min(a3, a6), c);

    auto mal4 = std::max(std::max(a4, a5), c);
    auto mil4 = std::min(std::min(a4, a5), c);

    int d1 = mal1 - mil1;
    int d2 = mal2 - mil2;
    int d3 = mal3 - mil3;
    int d4 = mal4 - mil4;

    Byte clipped1 = clip(val, mil1, mal1);
    Byte clipped2 = clip(val, mil2, mal2);
    Byte clipped3 = clip(val, mil3, mal3);
    Byte clipped4 = clip(val, mil4, mal4);

    int c1 = clip((std::abs(val-clipped1)<<1)+d1, 0, 255);
    int c2 = clip((std::abs(val-clipped2)<<1)+d2, 0, 255);
    int c3 = clip((std::abs(val-clipped3)<<1)+d3, 0, 255);
    int c4 = clip((std::abs(val-clipped4)<<1)+d4, 0, 255);

    int mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clip(val, mil4, mal4);
    if (mindiff == c2) return clip(val, mil2, mal2);
    if (mindiff == c3) return clip(val, mil3, mal3);
    return clip(val, mil1, mal1);
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode6_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  int d1 = mal1 - mil1;
  int d2 = mal2 - mil2;
  int d3 = mal3 - mil3;
  int d4 = mal4 - mil4;

  uint16_t clipped1 = clip_16(val, mil1, mal1);
  uint16_t clipped2 = clip_16(val, mil2, mal2);
  uint16_t clipped3 = clip_16(val, mil3, mal3);
  uint16_t clipped4 = clip_16(val, mil4, mal4);

  const int pixel_max = (1 << bits_per_pixel) - 1;
  int c1 = clip_16((std::abs(val-clipped1)<<1)+d1, 0, pixel_max);
  int c2 = clip_16((std::abs(val-clipped2)<<1)+d2, 0, pixel_max);
  int c3 = clip_16((std::abs(val-clipped3)<<1)+d3, 0, pixel_max);
  int c4 = clip_16((std::abs(val-clipped4)<<1)+d4, 0, pixel_max);

  int mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_16(val, mil4, mal4);
  if (mindiff == c2) return clip_16(val, mil2, mal2);
  if (mindiff == c3) return clip_16(val, mil3, mal3);
  return clip_16(val, mil1, mal1);
}

RG_FORCEINLINE float repair_mode6_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  float d1 = mal1 - mil1;
  float d2 = mal2 - mil2;
  float d3 = mal3 - mil3;
  float d4 = mal4 - mil4;

  float clipped1 = clip_32(val, mil1, mal1);
  float clipped2 = clip_32(val, mil2, mal2);
  float clipped3 = clip_32(val, mil3, mal3);
  float clipped4 = clip_32(val, mil4, mal4);

  float c1 = (std::abs(val - clipped1) * 2) + d1;
  float c2 = (std::abs(val - clipped2) * 2) + d2;
  float c3 = (std::abs(val - clipped3) * 2) + d3;
  float c4 = (std::abs(val - clipped4) * 2) + d4;

#if 0
  // no max_pixel_value clamp for float
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  const float pixel_min = 0.0f;
  const float pixel_max = 1.0f;
#else
  const float pixel_min = chroma ? -0.5f : 0.0f;
  const float pixel_max = chroma ? 0.5f : 1.0f;
#endif
  // special case, in this mode we allow clipping to a valid range
  c1 = clip_32(c1, pixel_min, pixel_max);
  c2 = clip_32(c2, pixel_min, pixel_max);
  c3 = clip_32(c3, pixel_min, pixel_max);
  c4 = clip_32(c4, pixel_min, pixel_max);
#endif

  float mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_32(val, mil4, mal4);
  if (mindiff == c2) return clip_32(val, mil2, mal2);
  if (mindiff == c3) return clip_32(val, mil3, mal3);
  return clip_32(val, mil1, mal1);
}

// ------------

RG_FORCEINLINE Byte repair_mode7_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(std::max(a1, a8), c);
    auto mil1 = std::min(std::min(a1, a8), c);

    auto mal2 = std::max(std::max(a2, a7), c);
    auto mil2 = std::min(std::min(a2, a7), c);

    auto mal3 = std::max(std::max(a3, a6), c);
    auto mil3 = std::min(std::min(a3, a6), c);

    auto mal4 = std::max(std::max(a4, a5), c);
    auto mil4 = std::min(std::min(a4, a5), c);

    auto d1 = mal1 - mil1;
    auto d2 = mal2 - mil2;
    auto d3 = mal3 - mil3;
    auto d4 = mal4 - mil4;

    auto clipped1 = clip(val, mil1, mal1);
    auto clipped2 = clip(val, mil2, mal2);
    auto clipped3 = clip(val, mil3, mal3);
    auto clipped4 = clip(val, mil4, mal4);

    int c1 = std::abs(val-clipped1)+d1;
    int c2 = std::abs(val-clipped2)+d2;
    int c3 = std::abs(val-clipped3)+d3;
    int c4 = std::abs(val-clipped4)+d4;

    auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clipped4;
    if (mindiff == c2) return clipped2;
    if (mindiff == c3) return clipped3;
    return clipped1;
}

RG_FORCEINLINE uint16_t repair_mode7_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  auto clipped1 = clip_16(val, mil1, mal1);
  auto clipped2 = clip_16(val, mil2, mal2);
  auto clipped3 = clip_16(val, mil3, mal3);
  auto clipped4 = clip_16(val, mil4, mal4);

  int c1 = std::abs(val-clipped1)+d1;
  int c2 = std::abs(val-clipped2)+d2;
  int c3 = std::abs(val-clipped3)+d3;
  int c4 = std::abs(val-clipped4)+d4;

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

RG_FORCEINLINE float repair_mode7_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  auto clipped1 = clip_32(val, mil1, mal1);
  auto clipped2 = clip_32(val, mil2, mal2);
  auto clipped3 = clip_32(val, mil3, mal3);
  auto clipped4 = clip_32(val, mil4, mal4);

  float c1 = std::abs(val-clipped1)+d1;
  float c2 = std::abs(val-clipped2)+d2;
  float c3 = std::abs(val-clipped3)+d3;
  float c4 = std::abs(val-clipped4)+d4;

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}


// ------------

RG_FORCEINLINE Byte repair_mode8_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(std::max(a1, a8), c);
    auto mil1 = std::min(std::min(a1, a8), c);

    auto mal2 = std::max(std::max(a2, a7), c);
    auto mil2 = std::min(std::min(a2, a7), c);

    auto mal3 = std::max(std::max(a3, a6), c);
    auto mil3 = std::min(std::min(a3, a6), c);

    auto mal4 = std::max(std::max(a4, a5), c);
    auto mil4 = std::min(std::min(a4, a5), c);

    Byte d1 = mal1 - mil1;
    Byte d2 = mal2 - mil2;
    Byte d3 = mal3 - mil3;
    Byte d4 = mal4 - mil4;

    Byte clipped1 = clip(val, mil1, mal1);
    Byte clipped2 = clip(val, mil2, mal2);
    Byte clipped3 = clip(val, mil3, mal3);
    Byte clipped4 = clip(val, mil4, mal4);

    int c1 = clip(std::abs(val-clipped1)+(d1<<1), 0, 255);
    int c2 = clip(std::abs(val-clipped2)+(d2<<1), 0, 255);
    int c3 = clip(std::abs(val-clipped3)+(d3<<1), 0, 255);
    int c4 = clip(std::abs(val-clipped4)+(d4<<1), 0, 255);

    Byte mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clipped4;
    if (mindiff == c2) return clipped2;
    if (mindiff == c3) return clipped3;
    return clipped1;
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode8_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  uint16_t d1 = mal1 - mil1;
  uint16_t d2 = mal2 - mil2;
  uint16_t d3 = mal3 - mil3;
  uint16_t d4 = mal4 - mil4;

  uint16_t clipped1 = clip_16(val, mil1, mal1);
  uint16_t clipped2 = clip_16(val, mil2, mal2);
  uint16_t clipped3 = clip_16(val, mil3, mal3);
  uint16_t clipped4 = clip_16(val, mil4, mal4);

  const int pixel_max = (1 << bits_per_pixel) - 1;
  int c1 = clip_16(std::abs(val-clipped1)+(d1<<1), 0, pixel_max);
  int c2 = clip_16(std::abs(val-clipped2)+(d2<<1), 0, pixel_max);
  int c3 = clip_16(std::abs(val-clipped3)+(d3<<1), 0, pixel_max);
  int c4 = clip_16(std::abs(val-clipped4)+(d4<<1), 0, pixel_max);

  uint16_t mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

RG_FORCEINLINE float repair_mode8_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  float d1 = mal1 - mil1;
  float d2 = mal2 - mil2;
  float d3 = mal3 - mil3;
  float d4 = mal4 - mil4;

  float clipped1 = clip_32(val, mil1, mal1);
  float clipped2 = clip_32(val, mil2, mal2);
  float clipped3 = clip_32(val, mil3, mal3);
  float clipped4 = clip_32(val, mil4, mal4);

  float c1 = std::abs(val-clipped1)+(d1 * 2);
  float c2 = std::abs(val-clipped2)+(d2 * 2);
  float c3 = std::abs(val-clipped3)+(d3 * 2);
  float c4 = std::abs(val-clipped4)+(d4 * 2);

#if 0
  // no max_pixel_value clamp for float
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  const float pixel_min = 0.0f;
  const float pixel_max = 1.0f;
#else
  const float pixel_min = chroma ? -0.5f : 0.0f;
  const float pixel_max = chroma ? 0.5f : 1.0f;
#endif
  // special case, in this mode we allow clipping to a valid range
  c1 = clip_32(c1, pixel_min, pixel_max);
  c2 = clip_32(c2, pixel_min, pixel_max);
  c3 = clip_32(c3, pixel_min, pixel_max);
  c4 = clip_32(c4, pixel_min, pixel_max);
#endif

  float mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}


// ------------

RG_FORCEINLINE Byte repair_mode9_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(std::max(a1, a8), c);
    auto mil1 = std::min(std::min(a1, a8), c);

    auto mal2 = std::max(std::max(a2, a7), c);
    auto mil2 = std::min(std::min(a2, a7), c);

    auto mal3 = std::max(std::max(a3, a6), c);
    auto mil3 = std::min(std::min(a3, a6), c);

    auto mal4 = std::max(std::max(a4, a5), c);
    auto mil4 = std::min(std::min(a4, a5), c);

    auto d1 = mal1 - mil1;
    auto d2 = mal2 - mil2;
    auto d3 = mal3 - mil3;
    auto d4 = mal4 - mil4;

    auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

    if (mindiff == d4) return clip(val, mil4, mal4);
    if (mindiff == d2) return clip(val, mil2, mal2);
    if (mindiff == d3) return clip(val, mil3, mal3);
    return clip(val, mil1, mal1);
}

RG_FORCEINLINE uint16_t repair_mode9_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_16(val, mil4, mal4);
  if (mindiff == d2) return clip_16(val, mil2, mal2);
  if (mindiff == d3) return clip_16(val, mil3, mal3);
  return clip_16(val, mil1, mal1);
}

RG_FORCEINLINE float repair_mode9_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(std::max(a1, a8), c);
  auto mil1 = std::min(std::min(a1, a8), c);

  auto mal2 = std::max(std::max(a2, a7), c);
  auto mil2 = std::min(std::min(a2, a7), c);

  auto mal3 = std::max(std::max(a3, a6), c);
  auto mil3 = std::min(std::min(a3, a6), c);

  auto mal4 = std::max(std::max(a4, a5), c);
  auto mil4 = std::min(std::min(a4, a5), c);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_32(val, mil4, mal4);
  if (mindiff == d2) return clip_32(val, mil2, mal2);
  if (mindiff == d3) return clip_32(val, mil3, mal3);
  return clip_32(val, mil1, mal1);
}

// ------------

RG_FORCEINLINE Byte repair_mode10_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::abs(val - a1);
    auto d2 = std::abs(val - a2);
    auto d3 = std::abs(val - a3);
    auto d4 = std::abs(val - a4);
    auto d5 = std::abs(val - a5);
    auto d6 = std::abs(val - a6);
    auto d7 = std::abs(val - a7);
    auto d8 = std::abs(val - a8);
    auto dc = std::abs(val - c);

    auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8), dc);

    if (mindiff == d7) return a7;
    if (mindiff == d8) return a8;
    if (mindiff == d6) return a6;
    if (mindiff == d2) return a2;
    if (mindiff == d3) return a3;
    if (mindiff == d1) return a1;
    if (mindiff == d5) return a5;
    if (mindiff == dc) return c;
    return a4;
}

RG_FORCEINLINE uint16_t repair_mode10_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::abs(val - a1);
  auto d2 = std::abs(val - a2);
  auto d3 = std::abs(val - a3);
  auto d4 = std::abs(val - a4);
  auto d5 = std::abs(val - a5);
  auto d6 = std::abs(val - a6);
  auto d7 = std::abs(val - a7);
  auto d8 = std::abs(val - a8);
  auto dc = std::abs(val - c);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8), dc);

  if (mindiff == d7) return a7;
  if (mindiff == d8) return a8;
  if (mindiff == d6) return a6;
  if (mindiff == d2) return a2;
  if (mindiff == d3) return a3;
  if (mindiff == d1) return a1;
  if (mindiff == d5) return a5;
  if (mindiff == dc) return c;
  return a4;
}

RG_FORCEINLINE float repair_mode10_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::abs(val - a1);
  auto d2 = std::abs(val - a2);
  auto d3 = std::abs(val - a3);
  auto d4 = std::abs(val - a4);
  auto d5 = std::abs(val - a5);
  auto d6 = std::abs(val - a6);
  auto d7 = std::abs(val - a7);
  auto d8 = std::abs(val - a8);
  auto dc = std::abs(val - c);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8), dc);

  if (mindiff == d7) return a7;
  if (mindiff == d8) return a8;
  if (mindiff == d6) return a6;
  if (mindiff == d2) return a2;
  if (mindiff == d3) return a3;
  if (mindiff == d1) return a1;
  if (mindiff == d5) return a5;
  if (mindiff == dc) return c;
  return a4;
}

// ------------

RG_FORCEINLINE Byte repair_mode12_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));
    Byte mi = std::min(a[1], c);
    Byte ma = std::max(a[6], c);

    return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode12_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  uint16_t mi = std::min(a[1], c);
  uint16_t ma = std::max(a[6], c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode12_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  float mi = std::min(a[1], c);
  float ma = std::max(a[6], c);

  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode13_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));
    Byte mi = std::min(a[2], c);
    Byte ma = std::max(a[5], c);

    return clip (val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode13_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  uint16_t mi = std::min(a[2], c);
  uint16_t ma = std::max(a[5], c);

  return clip_16 (val, mi, ma);
}

RG_FORCEINLINE float repair_mode13_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  float mi = std::min(a[2], c);
  float ma = std::max(a[5], c);

  return clip_32 (val, mi, ma);
}

// ------------


RG_FORCEINLINE Byte repair_mode14_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    std::array<Byte, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort(std::begin(a), std::end(a));
    Byte mi = std::min(a[3], c);
    Byte ma = std::max(a[4], c);

    return clip (val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode14_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  std::array<uint16_t, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  uint16_t mi = std::min(a[3], c);
  uint16_t ma = std::max(a[4], c);

  return clip_16 (val, mi, ma);
}

RG_FORCEINLINE float repair_mode14_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  std::array<float, 8> a = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(std::begin(a), std::end(a));
  float mi = std::min(a[3], c);
  float ma = std::max(a[4], c);

  return clip_32 (val, mi, ma);
}

// ------------


RG_FORCEINLINE Byte repair_mode15_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto clipped1 = clip(c, mil1, mal1);
    auto clipped2 = clip(c, mil2, mal2);
    auto clipped3 = clip(c, mil3, mal3);
    auto clipped4 = clip(c, mil4, mal4);

    Byte c1 = std::abs (c - clipped1);
    Byte c2 = std::abs (c - clipped2);
    Byte c3 = std::abs (c - clipped3);
    Byte c4 = std::abs (c - clipped4);

    Byte mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

    Byte mi;
    Byte ma;
    if (mindiff == c4)
    {
        mi = mil4;
        ma = mal4;
    }
    else if (mindiff == c2)
    {
        mi = mil2;
        ma = mal2;
    }
    else if (mindiff == c3)
    {
        mi = mil3;
        ma = mal3;
    }
    else
    {
        mi = mil1;
        ma = mal1;
    }

    mi = std::min(mi, c);
    ma = std::max(ma, c);

    return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode15_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto clipped1 = clip_16(c, mil1, mal1);
  auto clipped2 = clip_16(c, mil2, mal2);
  auto clipped3 = clip_16(c, mil3, mal3);
  auto clipped4 = clip_16(c, mil4, mal4);

  uint16_t c1 = std::abs (c - clipped1);
  uint16_t c2 = std::abs (c - clipped2);
  uint16_t c3 = std::abs (c - clipped3);
  uint16_t c4 = std::abs (c - clipped4);

  uint16_t mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

  uint16_t mi;
  uint16_t ma;
  if (mindiff == c4)
  {
    mi = mil4;
    ma = mal4;
  }
  else if (mindiff == c2)
  {
    mi = mil2;
    ma = mal2;
  }
  else if (mindiff == c3)
  {
    mi = mil3;
    ma = mal3;
  }
  else
  {
    mi = mil1;
    ma = mal1;
  }

  mi = std::min(mi, c);
  ma = std::max(ma, c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode15_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto clipped1 = clip_32(c, mil1, mal1);
  auto clipped2 = clip_32(c, mil2, mal2);
  auto clipped3 = clip_32(c, mil3, mal3);
  auto clipped4 = clip_32(c, mil4, mal4);

  float c1 = std::abs (c - clipped1);
  float c2 = std::abs (c - clipped2);
  float c3 = std::abs (c - clipped3);
  float c4 = std::abs (c - clipped4);

  float mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

  float mi;
  float ma;
  if (mindiff == c4)
  {
    mi = mil4;
    ma = mal4;
  }
  else if (mindiff == c2)
  {
    mi = mil2;
    ma = mal2;
  }
  else if (mindiff == c3)
  {
    mi = mil3;
    ma = mal3;
  }
  else
  {
    mi = mil1;
    ma = mal1;
  }

  mi = std::min(mi, c);
  ma = std::max(ma, c);

  return clip_32(val, mi, ma);
}

// ------------


RG_FORCEINLINE Byte repair_mode16_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto d1 = mal1 - mil1;
    auto d2 = mal2 - mil2;
    auto d3 = mal3 - mil3;
    auto d4 = mal4 - mil4;
     
    auto c1 = clip((std::abs (c - clip(c, mil1, mal1)) << 1) + d1, 0, 0xFF);
    auto c2 = clip((std::abs (c - clip(c, mil2, mal2)) << 1) + d2, 0, 0xFF);
    auto c3 = clip((std::abs (c - clip(c, mil3, mal3)) << 1) + d3, 0, 0xFF);
    auto c4 = clip((std::abs (c - clip(c, mil4, mal4)) << 1) + d4, 0, 0xFF);

    auto mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

    Byte mi;
    Byte ma;
    if (mindiff == c4)
    {
        mi = mil4;
        ma = mal4;
    }
    else if (mindiff == c2)
    {
        mi = mil2;
        ma = mal2;
    }
    else if (mindiff == c3)
    {
        mi = mil3;
        ma = mal3;
    }
    else
    {
        mi = mil1;
        ma = mal1;
    }

    mi = std::min (mi, c);
    ma = std::max (ma, c);

    return clip(val, mi, ma);
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode16_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  const int pixel_max = (1 << bits_per_pixel) - 1;
  auto c1 = clip_16((std::abs (c - clip_16(c, mil1, mal1)) << 1) + d1, 0, pixel_max);
  auto c2 = clip_16((std::abs (c - clip_16(c, mil2, mal2)) << 1) + d2, 0, pixel_max);
  auto c3 = clip_16((std::abs (c - clip_16(c, mil3, mal3)) << 1) + d3, 0, pixel_max);
  auto c4 = clip_16((std::abs (c - clip_16(c, mil4, mal4)) << 1) + d4, 0, pixel_max);

  auto mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

  uint16_t mi;
  uint16_t ma;
  if (mindiff == c4)
  {
    mi = mil4;
    ma = mal4;
  }
  else if (mindiff == c2)
  {
    mi = mil2;
    ma = mal2;
  }
  else if (mindiff == c3)
  {
    mi = mil3;
    ma = mal3;
  }
  else
  {
    mi = mil1;
    ma = mal1;
  }

  mi = std::min (mi, c);
  ma = std::max (ma, c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode16_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = mal1 - mil1;
  auto d2 = mal2 - mil2;
  auto d3 = mal3 - mil3;
  auto d4 = mal4 - mil4;

  auto c1 = (std::abs (c - clip_32(c, mil1, mal1)) * 2) + d1;
  auto c2 = (std::abs (c - clip_32(c, mil2, mal2)) * 2) + d2;
  auto c3 = (std::abs (c - clip_32(c, mil3, mal3)) * 2) + d3;
  auto c4 = (std::abs (c - clip_32(c, mil4, mal4)) * 2) + d4;

#if 0
  // no max_pixel_value clamp for float
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
  const float pixel_min = 0.0f;
  const float pixel_max = 1.0f;
#else
  const float pixel_min = chroma ? -0.5f : 0.0f;
  const float pixel_max = chroma ? 0.5f : 1.0f;
#endif
  // special case, in this mode we allow clipping to a valid range
  c1 = clip_32(c1, pixel_min, pixel_max);
  c2 = clip_32(c2, pixel_min, pixel_max);
  c3 = clip_32(c3, pixel_min, pixel_max);
  c4 = clip_32(c4, pixel_min, pixel_max);
#endif

  auto mindiff = std::min (std::min (c1, c2), std::min (c3, c4));

  float mi;
  float ma;
  if (mindiff == c4)
  {
    mi = mil4;
    ma = mal4;
  }
  else if (mindiff == c2)
  {
    mi = mil2;
    ma = mal2;
  }
  else if (mindiff == c3)
  {
    mi = mil3;
    ma = mal3;
  }
  else
  {
    mi = mil1;
    ma = mal1;
  }

  mi = std::min (mi, c);
  ma = std::max (ma, c);

  return clip_32(val, mi, ma);
}


// ------------


RG_FORCEINLINE Byte repair_mode17_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    Byte l = std::max (std::max (mil1, mil2), std::max (mil3, mil4));
    Byte u = std::min (std::min (mal1, mal2), std::min (mal3, mal4));
    
	Byte mi = std::min (std::min (l, u), c);
	Byte ma = std::max (std::max (l, u), c);

	return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode17_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  uint16_t l = std::max (std::max (mil1, mil2), std::max (mil3, mil4));
  uint16_t u = std::min (std::min (mal1, mal2), std::min (mal3, mal4));

  uint16_t mi = std::min (std::min (l, u), c);
  uint16_t ma = std::max (std::max (l, u), c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode17_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  float l = std::max (std::max (mil1, mil2), std::max (mil3, mil4));
  float u = std::min (std::min (mal1, mal2), std::min (mal3, mal4));

  float mi = std::min (std::min (l, u), c);
  float ma = std::max (std::max (l, u), c);

  return clip_32(val, mi, ma);
}


// ------------


RG_FORCEINLINE Byte repair_mode18_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
    auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
    auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
    auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

    auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

    Byte mi;
    Byte ma;
    if (mindiff == d4)
    {
        mi = std::min (a4, a5);
        ma = std::max (a4, a5);
    }
    else if (mindiff == d2)
    {
        mi = std::min (a2, a7);
        ma = std::max (a2, a7);
    }
    else if (mindiff == d3)
    {
        mi = std::min (a3, a6);
        ma = std::max (a3, a6);
    }
    else
    {
        mi = std::min (a1, a8);
        ma = std::max (a1, a8);
    }

    mi = std::min (mi, c);
    ma = std::max (ma, c);

    return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode18_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
  auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
  auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
  auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  uint16_t mi;
  uint16_t ma;
  if (mindiff == d4)
  {
    mi = std::min (a4, a5);
    ma = std::max (a4, a5);
  }
  else if (mindiff == d2)
  {
    mi = std::min (a2, a7);
    ma = std::max (a2, a7);
  }
  else if (mindiff == d3)
  {
    mi = std::min (a3, a6);
    ma = std::max (a3, a6);
  }
  else
  {
    mi = std::min (a1, a8);
    ma = std::max (a1, a8);
  }

  mi = std::min (mi, c);
  ma = std::max (ma, c);

  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode18_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
  auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
  auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
  auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  float mi;
  float ma;
  if (mindiff == d4)
  {
    mi = std::min (a4, a5);
    ma = std::max (a4, a5);
  }
  else if (mindiff == d2)
  {
    mi = std::min (a2, a7);
    ma = std::max (a2, a7);
  }
  else if (mindiff == d3)
  {
    mi = std::min (a3, a6);
    ma = std::max (a3, a6);
  }
  else
  {
    mi = std::min (a1, a8);
    ma = std::max (a1, a8);
  }

  mi = std::min (mi, c);
  ma = std::max (ma, c);

  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode19_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::abs(c - a1);
    auto d2 = std::abs(c - a2);
    auto d3 = std::abs(c - a3);
    auto d4 = std::abs(c - a4);
    auto d5 = std::abs(c - a5);
    auto d6 = std::abs(c - a6);
    auto d7 = std::abs(c - a7);
    auto d8 = std::abs(c - a8);

    auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

    return clip(val, clip(c-mindiff, 0, 255), clip(c+mindiff, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode19_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::abs(c - a1);
  auto d2 = std::abs(c - a2);
  auto d3 = std::abs(c - a3);
  auto d4 = std::abs(c - a4);
  auto d5 = std::abs(c - a5);
  auto d6 = std::abs(c - a6);
  auto d7 = std::abs(c - a7);
  auto d8 = std::abs(c - a8);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

  const int pixel_max = (1 << bits_per_pixel) - 1;
  return clip_16(val, clip_16(c-mindiff, 0, pixel_max), clip_16(c+mindiff, 0, pixel_max));
}

template<bool chroma>
RG_FORCEINLINE float repair_mode19_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::abs(c - a1);
  auto d2 = std::abs(c - a2);
  auto d3 = std::abs(c - a3);
  auto d4 = std::abs(c - a4);
  auto d5 = std::abs(c - a5);
  auto d6 = std::abs(c - a6);
  auto d7 = std::abs(c - a7);
  auto d8 = std::abs(c - a8);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

  float mi = subs_32_c<chroma>(c, mindiff);
  float ma = adds_32_c<chroma>(c, mindiff);
  // mi = clip_32(c-mindiff, pixel_min, pixel_max);
  // ma = clip_32(c+mindiff, pixel_min, pixel_max);
  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode20_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte d1 = std::abs(c - a1);
    Byte d2 = std::abs(c - a2);
    Byte d3 = std::abs(c - a3);
    Byte d4 = std::abs(c - a4);
    Byte d5 = std::abs(c - a5);
    Byte d6 = std::abs(c - a6);
    Byte d7 = std::abs(c - a7);
    Byte d8 = std::abs(c - a8);

    Byte mindiff = std::min(d1, d2);
    Byte maxdiff = std::max(d1, d2);

    maxdiff = clip(maxdiff, mindiff, d3);
    mindiff = std::min(mindiff, d3);

    maxdiff = clip(maxdiff, mindiff, d4);
    mindiff = std::min(mindiff, d4);

    maxdiff = clip(maxdiff, mindiff, d5);
    mindiff = std::min(mindiff, d5);

    maxdiff = clip(maxdiff, mindiff, d6);
    mindiff = std::min(mindiff, d6);

    maxdiff = clip(maxdiff, mindiff, d7);
    mindiff = std::min(mindiff, d7);

    maxdiff = clip(maxdiff, mindiff, d8);

    return clip(val, clip(c-maxdiff, 0, 255), clip(c+maxdiff, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode20_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t d1 = std::abs(c - a1);
  uint16_t d2 = std::abs(c - a2);
  uint16_t d3 = std::abs(c - a3);
  uint16_t d4 = std::abs(c - a4);
  uint16_t d5 = std::abs(c - a5);
  uint16_t d6 = std::abs(c - a6);
  uint16_t d7 = std::abs(c - a7);
  uint16_t d8 = std::abs(c - a8);

  uint16_t mindiff = std::min(d1, d2);
  uint16_t maxdiff = std::max(d1, d2);

  maxdiff = clip_16(maxdiff, mindiff, d3);
  mindiff = std::min(mindiff, d3);

  maxdiff = clip_16(maxdiff, mindiff, d4);
  mindiff = std::min(mindiff, d4);

  maxdiff = clip_16(maxdiff, mindiff, d5);
  mindiff = std::min(mindiff, d5);

  maxdiff = clip_16(maxdiff, mindiff, d6);
  mindiff = std::min(mindiff, d6);

  maxdiff = clip_16(maxdiff, mindiff, d7);
  mindiff = std::min(mindiff, d7);

  maxdiff = clip_16(maxdiff, mindiff, d8);

  const int max_pixel = (1 << bits_per_pixel) - 1;
  return clip_16(val, clip_16(c-maxdiff, 0, max_pixel), clip_16(c+maxdiff, 0, max_pixel));
}

template<bool chroma>
RG_FORCEINLINE float repair_mode20_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float d1 = std::abs(c - a1);
  float d2 = std::abs(c - a2);
  float d3 = std::abs(c - a3);
  float d4 = std::abs(c - a4);
  float d5 = std::abs(c - a5);
  float d6 = std::abs(c - a6);
  float d7 = std::abs(c - a7);
  float d8 = std::abs(c - a8);

  float mindiff = std::min(d1, d2);
  float maxdiff = std::max(d1, d2);

  maxdiff = clip_32(maxdiff, mindiff, d3);
  mindiff = std::min(mindiff, d3);

  maxdiff = clip_32(maxdiff, mindiff, d4);
  mindiff = std::min(mindiff, d4);

  maxdiff = clip_32(maxdiff, mindiff, d5);
  mindiff = std::min(mindiff, d5);

  maxdiff = clip_32(maxdiff, mindiff, d6);
  mindiff = std::min(mindiff, d6);

  maxdiff = clip_32(maxdiff, mindiff, d7);
  mindiff = std::min(mindiff, d7);

  maxdiff = clip_32(maxdiff, mindiff, d8);

  float mi = subs_32_c<chroma>(c, maxdiff);
  float ma = adds_32_c<chroma>(c, maxdiff);
  // mi = clip_32(mi, pixel_min, pixel_max)
  // ma = clip_32(ma, pixel_min, pixel_max)
  return clip_32(val, mi, ma);
}


// ------------

RG_FORCEINLINE Byte repair_mode21_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto d1 = clip(mal1 - c, 0, 255);
    auto d2 = clip(mal2 - c, 0, 255);
    auto d3 = clip(mal3 - c, 0, 255);
    auto d4 = clip(mal4 - c, 0, 255);

    auto rd1 = clip(c-mil1, 0, 255);
    auto rd2 = clip(c-mil2, 0, 255);
    auto rd3 = clip(c-mil3, 0, 255);
    auto rd4 = clip(c-mil4, 0, 255);

    auto u1  = std::max(d1, rd1);
    auto u2  = std::max(d2, rd2);
    auto u3  = std::max(d3, rd3);
    auto u4  = std::max(d4, rd4);

    auto u = std::min(std::min(std::min(u1, u2), u3), u4);

    return clip(val, clip(c-u, 0, 255), clip(c+u, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode21_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  const int max_pixel = (1 << bits_per_pixel) - 1;
  auto d1 = clip_16(mal1 - c, 0, max_pixel);
  auto d2 = clip_16(mal2 - c, 0, max_pixel);
  auto d3 = clip_16(mal3 - c, 0, max_pixel);
  auto d4 = clip_16(mal4 - c, 0, max_pixel);

  auto rd1 = clip_16(c-mil1, 0, max_pixel);
  auto rd2 = clip_16(c-mil2, 0, max_pixel);
  auto rd3 = clip_16(c-mil3, 0, max_pixel);
  auto rd4 = clip_16(c-mil4, 0, max_pixel);

  auto u1  = std::max(d1, rd1);
  auto u2  = std::max(d2, rd2);
  auto u3  = std::max(d3, rd3);
  auto u4  = std::max(d4, rd4);

  auto u = std::min(std::min(std::min(u1, u2), u3), u4);

  return clip_16(val, clip_16(c-u, 0, max_pixel), clip_16(c+u, 0, max_pixel));
}

template<bool chroma>
RG_FORCEINLINE float repair_mode21_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_32_c_for_diff(mal1, c);
  auto d2 = subs_32_c_for_diff(mal2, c);
  auto d3 = subs_32_c_for_diff(mal3, c);
  auto d4 = subs_32_c_for_diff(mal4, c);

  auto rd1 = subs_32_c_for_diff(c, mil1);
  auto rd2 = subs_32_c_for_diff(c, mil2);
  auto rd3 = subs_32_c_for_diff(c, mil3);
  auto rd4 = subs_32_c_for_diff(c, mil4);

  auto u1  = std::max(d1, rd1);
  auto u2  = std::max(d2, rd2);
  auto u3  = std::max(d3, rd3);
  auto u4  = std::max(d4, rd4);

  auto u = std::min(std::min(std::min(u1, u2), u3), u4);

  float mi = subs_32_c<chroma>(c, u);
  float ma = adds_32_c<chroma>(c, u);
  // mi = clip_32(mi, pixel_min, pixel_max);
  // ma = clip_32(ma, pixel_min, pixel_max);
  return clip_32(val, mi, ma);  
}

// ------------

RG_FORCEINLINE Byte repair_mode22_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::abs(val - a1);
    auto d2 = std::abs(val - a2);
    auto d3 = std::abs(val - a3);
    auto d4 = std::abs(val - a4);
    auto d5 = std::abs(val - a5);
    auto d6 = std::abs(val - a6);
    auto d7 = std::abs(val - a7);
    auto d8 = std::abs(val - a8);

    auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

    return clip(c, clip(val-mindiff, 0, 255), clip(val+mindiff, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode22_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::abs(val - a1);
  auto d2 = std::abs(val - a2);
  auto d3 = std::abs(val - a3);
  auto d4 = std::abs(val - a4);
  auto d5 = std::abs(val - a5);
  auto d6 = std::abs(val - a6);
  auto d7 = std::abs(val - a7);
  auto d8 = std::abs(val - a8);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

  const int max_pixel = (1 << bits_per_pixel) - 1;
  return clip_16(c, clip_16(val-mindiff, 0, max_pixel), clip_16(val+mindiff, 0, max_pixel));
}

template<bool chroma>
RG_FORCEINLINE float repair_mode22_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::abs(val - a1);
  auto d2 = std::abs(val - a2);
  auto d3 = std::abs(val - a3);
  auto d4 = std::abs(val - a4);
  auto d5 = std::abs(val - a5);
  auto d6 = std::abs(val - a6);
  auto d7 = std::abs(val - a7);
  auto d8 = std::abs(val - a8);

  auto mindiff = std::min(std::min(std::min(std::min(std::min(std::min(std::min(d1, d2), d3), d4), d5), d6), d7), d8);

  float mi = subs_32_c<chroma>(val, mindiff);
  float ma = adds_32_c<chroma>(val, mindiff);
  // mi = clip_32(mi, min_pixel, max_pixel);
  // ma = clip_32(ma, min_pixel, max_pixel);
  return clip_32(c, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode23_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte d1 = std::abs(val - a1);
    Byte d2 = std::abs(val - a2);
    Byte d3 = std::abs(val - a3);
    Byte d4 = std::abs(val - a4);
    Byte d5 = std::abs(val - a5);
    Byte d6 = std::abs(val - a6);
    Byte d7 = std::abs(val - a7);
    Byte d8 = std::abs(val - a8);

    Byte mindiff = std::min(d1, d2);
    Byte maxdiff = std::max(d1, d2);

    maxdiff = clip(maxdiff, mindiff, d3);
    mindiff = std::min(mindiff, d3);

    maxdiff = clip(maxdiff, mindiff, d4);
    mindiff = std::min(mindiff, d4);

    maxdiff = clip(maxdiff, mindiff, d5);
    mindiff = std::min(mindiff, d5);

    maxdiff = clip(maxdiff, mindiff, d6);
    mindiff = std::min(mindiff, d6);

    maxdiff = clip(maxdiff, mindiff, d7);
    mindiff = std::min(mindiff, d7);

    maxdiff = clip(maxdiff, mindiff, d8);

    return clip(c, clip(val-maxdiff, 0, 255), clip(val+maxdiff, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode23_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t d1 = std::abs(val - a1);
  uint16_t d2 = std::abs(val - a2);
  uint16_t d3 = std::abs(val - a3);
  uint16_t d4 = std::abs(val - a4);
  uint16_t d5 = std::abs(val - a5);
  uint16_t d6 = std::abs(val - a6);
  uint16_t d7 = std::abs(val - a7);
  uint16_t d8 = std::abs(val - a8);

  uint16_t mindiff = std::min(d1, d2);
  uint16_t maxdiff = std::max(d1, d2);

  maxdiff = clip_16(maxdiff, mindiff, d3);
  mindiff = std::min(mindiff, d3);

  maxdiff = clip_16(maxdiff, mindiff, d4);
  mindiff = std::min(mindiff, d4);

  maxdiff = clip_16(maxdiff, mindiff, d5);
  mindiff = std::min(mindiff, d5);

  maxdiff = clip_16(maxdiff, mindiff, d6);
  mindiff = std::min(mindiff, d6);

  maxdiff = clip_16(maxdiff, mindiff, d7);
  mindiff = std::min(mindiff, d7);

  maxdiff = clip_16(maxdiff, mindiff, d8);

  const int max_pixel = (1 << bits_per_pixel) - 1;
  return clip_16(c, clip_16(val-maxdiff, 0, max_pixel), clip_16(val+maxdiff, 0, max_pixel));
}

template<bool chroma>
RG_FORCEINLINE float repair_mode23_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float d1 = std::abs(val - a1);
  float d2 = std::abs(val - a2);
  float d3 = std::abs(val - a3);
  float d4 = std::abs(val - a4);
  float d5 = std::abs(val - a5);
  float d6 = std::abs(val - a6);
  float d7 = std::abs(val - a7);
  float d8 = std::abs(val - a8);

  float mindiff = std::min(d1, d2);
  float maxdiff = std::max(d1, d2);

  maxdiff = clip_32(maxdiff, mindiff, d3);
  mindiff = std::min(mindiff, d3);

  maxdiff = clip_32(maxdiff, mindiff, d4);
  mindiff = std::min(mindiff, d4);

  maxdiff = clip_32(maxdiff, mindiff, d5);
  mindiff = std::min(mindiff, d5);

  maxdiff = clip_32(maxdiff, mindiff, d6);
  mindiff = std::min(mindiff, d6);

  maxdiff = clip_32(maxdiff, mindiff, d7);
  mindiff = std::min(mindiff, d7);

  maxdiff = clip_32(maxdiff, mindiff, d8);

  float mi = subs_32_c<chroma>(val, maxdiff);
  float ma = adds_32_c<chroma>(val, maxdiff);
  // mi =  clip_32(mi, pixel_min, pixel_max)
  // ma =  clip_32(ma, pixel_min, pixel_max)
  return clip_32(c, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode24_cpp(const Byte* pSrc, Byte val, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto d1 = clip(mal1 - val, 0, 255);
    auto d2 = clip(mal2 - val, 0, 255);
    auto d3 = clip(mal3 - val, 0, 255);
    auto d4 = clip(mal4 - val, 0, 255);

    auto rd1 = clip(val-mil1, 0, 255);
    auto rd2 = clip(val-mil2, 0, 255);
    auto rd3 = clip(val-mil3, 0, 255);
    auto rd4 = clip(val-mil4, 0, 255);

    auto u1  = std::max(d1, rd1);
    auto u2  = std::max(d2, rd2);
    auto u3  = std::max(d3, rd3);
    auto u4  = std::max(d4, rd4);
    
    auto u = std::min(std::min(std::min(u1, u2), u3), u4);

    return clip(c, clip(val-u, 0, 255), clip(val+u, 0, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t repair_mode24_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  const int max_pixel = (1 << bits_per_pixel) - 1;
  auto d1 = clip_16(mal1 - val, 0, max_pixel);
  auto d2 = clip_16(mal2 - val, 0, max_pixel);
  auto d3 = clip_16(mal3 - val, 0, max_pixel);
  auto d4 = clip_16(mal4 - val, 0, max_pixel);

  auto rd1 = clip_16(val-mil1, 0, max_pixel);
  auto rd2 = clip_16(val-mil2, 0, max_pixel);
  auto rd3 = clip_16(val-mil3, 0, max_pixel);
  auto rd4 = clip_16(val-mil4, 0, max_pixel);

  auto u1  = std::max(d1, rd1);
  auto u2  = std::max(d2, rd2);
  auto u3  = std::max(d3, rd3);
  auto u4  = std::max(d4, rd4);

  auto u = std::min(std::min(std::min(u1, u2), u3), u4);

  return clip_16(c, clip_16(val-u, 0, max_pixel), clip_16(val+u, 0, max_pixel));
/* all these "clip" and "clip_16" could be rewritten to (like in RemoveGrain part) 
   to subs and adds
  return clip_16(c, subs_16_c(val, u), adds_16_c<bits_per_pixel>(val, u));
*/
}

template<bool chroma>
RG_FORCEINLINE float repair_mode24_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_32_c_for_diff(mal1, val);
  auto d2 = subs_32_c_for_diff(mal2, val);
  auto d3 = subs_32_c_for_diff(mal3, val);
  auto d4 = subs_32_c_for_diff(mal4, val);

  auto rd1 = subs_32_c_for_diff(val, mil1);
  auto rd2 = subs_32_c_for_diff(val, mil2);
  auto rd3 = subs_32_c_for_diff(val, mil3);
  auto rd4 = subs_32_c_for_diff(val, mil4);

  auto u1  = std::max(d1, rd1);
  auto u2  = std::max(d2, rd2);
  auto u3  = std::max(d3, rd3);
  auto u4  = std::max(d4, rd4);

  auto u = std::min(std::min(std::min(u1, u2), u3), u4);

  float mi = val - u;
  float ma = val + u;
  // mi = clip_32(mi, pixel_min, pixel_max)
  // ma = clip_32(ma, pixel_min, pixel_max)
  return clip_32(c, mi, ma);
}

// ------------
// mode25: does not exist in repair
// ------------
// similar to mode17
RG_FORCEINLINE Byte repair_mode26_cpp(const Byte* pSrc, Byte val, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  /*
  normal removegrain
    mi = std::min (lower, upper);
    ma = std::max (lower, upper);
    return clip(c, mi, ma);

  repair difference:
    mi = std::min (std::min (lower, upper), c);
    ma = std::max (std::max (lower, upper), c);
    return clip(val, mi, ma);
  */
  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode26_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode26_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode27_cpp(const Byte* pSrc, Byte val, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a1, a2);
  auto mil2 = std::min(a1, a2);

  auto mal3 = std::max(a7, a8);
  auto mil3 = std::min(a7, a8);

  auto mal4 = std::max(a2, a7);
  auto mil4 = std::min(a2, a7);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a2, a3);
  mil1 = std::min(a2, a3);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a3, a6);
  mil3 = std::min(a3, a6);

  mal4 = std::max(a3, a5);
  mil4 = std::min(a3, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a4, a6);
  mil1 = std::min(a4, a6);

  mal2 = std::max(a4, a5);
  mil2 = std::min(a4, a5);

  mal3 = std::max(a5, a8);
  mil3 = std::min(a5, a8);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode27_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a1, a2);
  auto mil2 = std::min(a1, a2);

  auto mal3 = std::max(a7, a8);
  auto mil3 = std::min(a7, a8);

  auto mal4 = std::max(a2, a7);
  auto mil4 = std::min(a2, a7);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a2, a3);
  mil1 = std::min(a2, a3);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a3, a6);
  mil3 = std::min(a3, a6);

  mal4 = std::max(a3, a5);
  mil4 = std::min(a3, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a4, a6);
  mil1 = std::min(a4, a6);

  mal2 = std::max(a4, a5);
  mil2 = std::min(a4, a5);

  mal3 = std::max(a5, a8);
  mil3 = std::min(a5, a8);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode27_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a1, a2);
  auto mil2 = std::min(a1, a2);

  auto mal3 = std::max(a7, a8);
  auto mil3 = std::min(a7, a8);

  auto mal4 = std::max(a2, a7);
  auto mil4 = std::min(a2, a7);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a2, a3);
  mil1 = std::min(a2, a3);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a3, a6);
  mil3 = std::min(a3, a6);

  mal4 = std::max(a3, a5);
  mil4 = std::min(a3, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a4, a6);
  mil1 = std::min(a4, a6);

  mal2 = std::max(a4, a5);
  mil2 = std::min(a4, a5);

  mal3 = std::max(a5, a8);
  mil3 = std::min(a5, a8);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_32(val, mi, ma);
}

// ------------

RG_FORCEINLINE Byte repair_mode28_cpp(const Byte* pSrc, Byte val, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a1, a8);
  mil1 = std::min(a1, a8);

  mal2 = std::max(a3, a6);
  mil2 = std::min(a3, a6);

  mal3 = std::max(a2, a7);
  mil3 = std::min(a2, a7);

  mal4 = std::max(a4, a5);
  mil4 = std::min(a4, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip(val, mi, ma);
}

RG_FORCEINLINE uint16_t repair_mode28_cpp_16(const Byte* pSrc, uint16_t val, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a1, a8);
  mil1 = std::min(a1, a8);

  mal2 = std::max(a3, a6);
  mil2 = std::min(a3, a6);

  mal3 = std::max(a2, a7);
  mil3 = std::min(a2, a7);

  mal4 = std::max(a4, a5);
  mil4 = std::min(a4, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_16(val, mi, ma);
}

RG_FORCEINLINE float repair_mode28_cpp_32(const Byte* pSrc, float val, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a2);
  auto mil1 = std::min(a1, a2);

  auto mal2 = std::max(a2, a3);
  auto mil2 = std::min(a2, a3);

  auto mal3 = std::max(a3, a5);
  auto mil3 = std::min(a3, a5);

  auto mal4 = std::max(a5, a8);
  auto mil4 = std::min(a5, a8);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  mal1 = std::max(a7, a8);
  mil1 = std::min(a7, a8);

  mal2 = std::max(a6, a7);
  mil2 = std::min(a6, a7);

  mal3 = std::max(a4, a6);
  mil3 = std::min(a4, a6);

  mal4 = std::max(a1, a4);
  mil4 = std::min(a1, a4);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  mal1 = std::max(a1, a8);
  mil1 = std::min(a1, a8);

  mal2 = std::max(a3, a6);
  mil2 = std::min(a3, a6);

  mal3 = std::max(a2, a7);
  mil3 = std::min(a2, a7);

  mal4 = std::max(a4, a5);
  mil4 = std::min(a4, a5);

  lower = std::max(std::max(std::max(std::max(mil1, mil2), mil3), mil4), lower);
  upper = std::min(std::min(std::min(std::min(mal1, mal2), mal3), mal4), upper);

  auto mi = std::min(std::min(lower, upper), c);
  auto ma = std::max(std::max(lower, upper), c);
  return clip_32(val, mi, ma);
}

#undef LOAD_SQUARE_CPP
#undef LOAD_SQUARE_CPP_16
#undef LOAD_SQUARE_CPP_32
#undef LOAD_SQUARE_CPP_T

#endif