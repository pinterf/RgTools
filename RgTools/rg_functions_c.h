#ifndef __RG_FUNCTIONS_C_H__
#define __RG_FUNCTIONS_C_H__

#include "common.h"
#include <algorithm>  


template<typename pixel_t>
using CModeProcessor = pixel_t (*)(const Byte*, int);

RG_FORCEINLINE Byte rg_mode1_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte mi = std::min (
        std::min (std::min (a1, a2), std::min (a3, a4)),
        std::min (std::min (a5, a6), std::min (a7, a8))
        );
    Byte ma = std::max (
        std::max (std::max (a1, a2), std::max (a3, a4)),
        std::max (std::max (a5, a6), std::max (a7, a8))
        );

    return clip(c, mi, ma);
}

RG_FORCEINLINE uint16_t rg_mode1_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t mi = std::min (
    std::min (std::min (a1, a2), std::min (a3, a4)),
    std::min (std::min (a5, a6), std::min (a7, a8))
  );
  uint16_t ma = std::max (
    std::max (std::max (a1, a2), std::max (a3, a4)),
    std::max (std::max (a5, a6), std::max (a7, a8))
  );

  return clip_16(c, mi, ma);
}

RG_FORCEINLINE float rg_mode1_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float mi = std::min (
    std::min (std::min (a1, a2), std::min (a3, a4)),
    std::min (std::min (a5, a6), std::min (a7, a8))
  );
  float ma = std::max (
    std::max (std::max (a1, a2), std::max (a3, a4)),
    std::max (std::max (a5, a6), std::max (a7, a8))
  );

  return clip_32(c, mi, ma);
}

// ------------

RG_FORCEINLINE Byte rg_mode2_cpp(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  Byte a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(&a [0], (&a [7]) + 1);

  return clip(c, a [2-1], a [7-1]);
}

RG_FORCEINLINE uint16_t rg_mode2_cpp_16(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP_16(pSrc, srcPitch);

    uint16_t a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort(&a [0], (&a [7]) + 1);

    return clip_16(c, a [2-1], a [7-1]);
}

RG_FORCEINLINE float rg_mode2_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(&a [0], (&a [7]) + 1);

  return clip_32(c, a [2-1], a [7-1]);
}

// ------------

RG_FORCEINLINE Byte rg_mode3_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte	a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort (&a [0], (&a [7]) + 1);

    return clip(c, a [3-1], a [6-1]);
}

RG_FORCEINLINE uint16_t rg_mode3_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t	a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort (&a [0], (&a [7]) + 1);

  return clip_16(c, a [3-1], a [6-1]);
}

RG_FORCEINLINE float rg_mode3_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float a[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(&a[0], (&a[7]) + 1);

  return clip_32(c, a[3 - 1], a[6 - 1]);
}

// ------------

RG_FORCEINLINE Byte rg_mode4_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte a[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

    std::sort(&a[0], (&a[7]) + 1);

    return clip(c, a[4 - 1], a[5 - 1]);
}

RG_FORCEINLINE uint16_t rg_mode4_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t a[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(&a[0], (&a[7]) + 1);

  return clip_16(c, a[4 - 1], a[5 - 1]);
}

RG_FORCEINLINE float rg_mode4_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float	a [8] = { a1, a2, a3, a4, a5, a6, a7, a8 };

  std::sort(&a[0], (&a[7]) + 1);

  return clip_32(c, a[4 - 1], a[5 - 1]);
}

// ------------

RG_FORCEINLINE Byte rg_mode5_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    Byte c1 = std::abs(c-clip(c, mil1, mal1));
    Byte c2 = std::abs(c-clip(c, mil2, mal2));
    Byte c3 = std::abs(c-clip(c, mil3, mal3));
    Byte c4 = std::abs(c-clip(c, mil4, mal4));

    auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clip(c, mil4, mal4);
    if (mindiff == c2) return clip(c, mil2, mal2);
    if (mindiff == c3) return clip(c, mil3, mal3);
    return clip(c, mil1, mal1);
}

RG_FORCEINLINE uint16_t rg_mode5_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  uint16_t c1 = std::abs(c-clip_16(c, mil1, mal1));
  uint16_t c2 = std::abs(c-clip_16(c, mil2, mal2));
  uint16_t c3 = std::abs(c-clip_16(c, mil3, mal3));
  uint16_t c4 = std::abs(c-clip_16(c, mil4, mal4));

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_16(c, mil4, mal4);
  if (mindiff == c2) return clip_16(c, mil2, mal2);
  if (mindiff == c3) return clip_16(c, mil3, mal3);
  return clip_16(c, mil1, mal1);
}

RG_FORCEINLINE float rg_mode5_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  float c1 = std::abs(c-clip_32(c, mil1, mal1));
  float c2 = std::abs(c-clip_32(c, mil2, mal2));
  float c3 = std::abs(c-clip_32(c, mil3, mal3));
  float c4 = std::abs(c-clip_32(c, mil4, mal4));

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_32(c, mil4, mal4);
  if (mindiff == c2) return clip_32(c, mil2, mal2);
  if (mindiff == c3) return clip_32(c, mil3, mal3);
  return clip_32(c, mil1, mal1);
}

// ------------

RG_FORCEINLINE Byte rg_mode6_cpp(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  int d1 = subs_c(mal1, mil1);
  int d2 = subs_c(mal2, mil2);
  int d3 = subs_c(mal3, mil3);
  int d4 = subs_c(mal4, mil4);

  Byte clipped1 = clip(c, mil1, mal1);
  Byte clipped2 = clip(c, mil2, mal2);
  Byte clipped3 = clip(c, mil3, mal3);
  Byte clipped4 = clip(c, mil4, mal4);

  int c1 = adds_c((std::abs(c-clipped1)<<1), d1);
  int c2 = adds_c((std::abs(c-clipped2)<<1), d2);
  int c3 = adds_c((std::abs(c-clipped3)<<1), d3);
  int c4 = adds_c((std::abs(c-clipped4)<<1), d4);

  int mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip(c, mil4, mal4);
  if (mindiff == c2) return clip(c, mil2, mal2);
  if (mindiff == c3) return clip(c, mil3, mal3);
  return clip(c, mil1, mal1);
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t rg_mode6_cpp_16(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP_16(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    int d1 = subs_16_c(mal1, mil1);
    int d2 = subs_16_c(mal2, mil2);
    int d3 = subs_16_c(mal3, mil3);
    int d4 = subs_16_c(mal4, mil4);

    uint16_t clipped1 = clip_16(c, mil1, mal1);
    uint16_t clipped2 = clip_16(c, mil2, mal2);
    uint16_t clipped3 = clip_16(c, mil3, mal3);
    uint16_t clipped4 = clip_16(c, mil4, mal4);

    int c1 = adds_16_c<bits_per_pixel>(std::abs(c-clipped1)<<1, d1);
    int c2 = adds_16_c<bits_per_pixel>(std::abs(c-clipped2)<<1, d2);
    int c3 = adds_16_c<bits_per_pixel>(std::abs(c-clipped3)<<1, d3);
    int c4 = adds_16_c<bits_per_pixel>(std::abs(c-clipped4)<<1, d4);

    int mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clip_16(c, mil4, mal4);
    if (mindiff == c2) return clip_16(c, mil2, mal2);
    if (mindiff == c3) return clip_16(c, mil3, mal3);
    return clip_16(c, mil1, mal1);
}

//template<bool chroma>
RG_FORCEINLINE float rg_mode6_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  float d1 = subs_32_c_for_diff(mal1, mil1);
  float d2 = subs_32_c_for_diff(mal2, mil2);
  float d3 = subs_32_c_for_diff(mal3, mil3);
  float d4 = subs_32_c_for_diff(mal4, mil4);

  float clipped1 = clip_32(c, mil1, mal1);
  float clipped2 = clip_32(c, mil2, mal2);
  float clipped3 = clip_32(c, mil3, mal3);
  float clipped4 = clip_32(c, mil4, mal4);

  float c1 = adds_32_c_for_diff(std::abs(c - clipped1) * 2, d1);
  float c2 = adds_32_c_for_diff(std::abs(c - clipped2) * 2, d2);
  float c3 = adds_32_c_for_diff(std::abs(c - clipped3) * 2, d3);
  float c4 = adds_32_c_for_diff(std::abs(c - clipped4) * 2, d4);

  float mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clip_32(c, mil4, mal4);
  if (mindiff == c2) return clip_32(c, mil2, mal2);
  if (mindiff == c3) return clip_32(c, mil3, mal3);
  return clip_32(c, mil1, mal1);
}


// ------------

RG_FORCEINLINE Byte rg_mode7_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto d1 = subs_c(mal1, mil1);
    auto d2 = subs_c(mal2, mil2);
    auto d3 = subs_c(mal3, mil3);
    auto d4 = subs_c(mal4, mil4);

    auto clipped1 = clip(c, mil1, mal1);
    auto clipped2 = clip(c, mil2, mal2);
    auto clipped3 = clip(c, mil3, mal3);
    auto clipped4 = clip(c, mil4, mal4);

    int c1 = adds_c(std::abs(c-clipped1), d1);
    int c2 = adds_c(std::abs(c-clipped2), d2);
    int c3 = adds_c(std::abs(c-clipped3), d3);
    int c4 = adds_c(std::abs(c-clipped4), d4);

    auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clipped4;
    if (mindiff == c2) return clipped2;
    if (mindiff == c3) return clipped3;
    return clipped1;
}

RG_FORCEINLINE uint16_t rg_mode7_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_16_c(mal1, mil1);
  auto d2 = subs_16_c(mal2, mil2);
  auto d3 = subs_16_c(mal3, mil3);
  auto d4 = subs_16_c(mal4, mil4);

  auto clipped1 = clip_16(c, mil1, mal1);
  auto clipped2 = clip_16(c, mil2, mal2);
  auto clipped3 = clip_16(c, mil3, mal3);
  auto clipped4 = clip_16(c, mil4, mal4);

  int c1 = adds_16_c<16>(std::abs(c-clipped1), d1);
  int c2 = adds_16_c<16>(std::abs(c-clipped2), d2);
  int c3 = adds_16_c<16>(std::abs(c-clipped3), d3);
  int c4 = adds_16_c<16>(std::abs(c-clipped4), d4);

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

RG_FORCEINLINE float rg_mode7_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_32_c_for_diff(mal1, mil1);
  auto d2 = subs_32_c_for_diff(mal2, mil2);
  auto d3 = subs_32_c_for_diff(mal3, mil3);
  auto d4 = subs_32_c_for_diff(mal4, mil4);

  auto clipped1 = clip_32(c, mil1, mal1);
  auto clipped2 = clip_32(c, mil2, mal2);
  auto clipped3 = clip_32(c, mil3, mal3);
  auto clipped4 = clip_32(c, mil4, mal4);

  float c1 = std::abs(c-clipped1)+d1; // no adds needed
  float c2 = std::abs(c-clipped2)+d2;
  float c3 = std::abs(c-clipped3)+d3;
  float c4 = std::abs(c-clipped4)+d4;

  auto mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

// ------------

RG_FORCEINLINE Byte rg_mode8_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    Byte d1 = subs_c(mal1, mil1);
    Byte d2 = subs_c(mal2, mil2);
    Byte d3 = subs_c(mal3, mil3);
    Byte d4 = subs_c(mal4, mil4);

    Byte clipped1 = clip(c, mil1, mal1);
    Byte clipped2 = clip(c, mil2, mal2);
    Byte clipped3 = clip(c, mil3, mal3);
    Byte clipped4 = clip(c, mil4, mal4);
    
    int c1 = adds_c(std::abs(c-clipped1), d1<<1);
    int c2 = adds_c(std::abs(c-clipped2), d2<<1);
    int c3 = adds_c(std::abs(c-clipped3), d3<<1);
    int c4 = adds_c(std::abs(c-clipped4), d4<<1);

    Byte mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

    if (mindiff == c4) return clipped4;
    if (mindiff == c2) return clipped2;
    if (mindiff == c3) return clipped3;
    return clipped1;
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t rg_mode8_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  uint16_t d1 = subs_16_c(mal1, mil1);
  uint16_t d2 = subs_16_c(mal2, mil2);
  uint16_t d3 = subs_16_c(mal3, mil3);
  uint16_t d4 = subs_16_c(mal4, mil4);

  uint16_t clipped1 = clip_16(c, mil1, mal1);
  uint16_t clipped2 = clip_16(c, mil2, mal2);
  uint16_t clipped3 = clip_16(c, mil3, mal3);
  uint16_t clipped4 = clip_16(c, mil4, mal4);

  int c1 = adds_16_c<bits_per_pixel>(std::abs(c-clipped1), d1<<1);
  int c2 = adds_16_c<bits_per_pixel>(std::abs(c-clipped2), d2<<1);
  int c3 = adds_16_c<bits_per_pixel>(std::abs(c-clipped3), d3<<1);
  int c4 = adds_16_c<bits_per_pixel>(std::abs(c-clipped4), d4<<1);

  uint16_t mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

//template<bool chroma>
RG_FORCEINLINE float rg_mode8_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  float d1 = subs_32_c_for_diff(mal1, mil1);
  float d2 = subs_32_c_for_diff(mal2, mil2);
  float d3 = subs_32_c_for_diff(mal3, mil3);
  float d4 = subs_32_c_for_diff(mal4, mil4);

  float clipped1 = clip_32(c, mil1, mal1);
  float clipped2 = clip_32(c, mil2, mal2);
  float clipped3 = clip_32(c, mil3, mal3);
  float clipped4 = clip_32(c, mil4, mal4);

  float c1 = std::abs(c - clipped1) + (d1 * 2); // no adds needed
  float c2 = std::abs(c - clipped2) + (d2 * 2);
  float c3 = std::abs(c - clipped3) + (d3 * 2);
  float c4 = std::abs(c - clipped4) + (d4 * 2);

  float mindiff = std::min(std::min(std::min(c1, c2), c3), c4);

  if (mindiff == c4) return clipped4;
  if (mindiff == c2) return clipped2;
  if (mindiff == c3) return clipped3;
  return clipped1;
}

// ------------

RG_FORCEINLINE Byte rg_mode9_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto d1 = subs_c(mal1, mil1);
    auto d2 = subs_c(mal2, mil2);
    auto d3 = subs_c(mal3, mil3);
    auto d4 = subs_c(mal4, mil4);

    auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

    if (mindiff == d4) return clip(c, mil4, mal4);
    if (mindiff == d2) return clip(c, mil2, mal2);
    if (mindiff == d3) return clip(c, mil3, mal3);
    return clip(c, mil1, mal1);
}

RG_FORCEINLINE uint16_t rg_mode9_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_16_c(mal1, mil1);
  auto d2 = subs_16_c(mal2, mil2);
  auto d3 = subs_16_c(mal3, mil3);
  auto d4 = subs_16_c(mal4, mil4);

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_16(c, mil4, mal4);
  if (mindiff == d2) return clip_16(c, mil2, mal2);
  if (mindiff == d3) return clip_16(c, mil3, mal3);
  return clip_16(c, mil1, mal1);
}

RG_FORCEINLINE float rg_mode9_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto d1 = subs_32_c_for_diff(mal1, mil1);
  auto d2 = subs_32_c_for_diff(mal2, mil2);
  auto d3 = subs_32_c_for_diff(mal3, mil3);
  auto d4 = subs_32_c_for_diff(mal4, mil4);

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_32(c, mil4, mal4);
  if (mindiff == d2) return clip_32(c, mil2, mal2);
  if (mindiff == d3) return clip_32(c, mil3, mal3);
  return clip_32(c, mil1, mal1);
}

// ------------

RG_FORCEINLINE Byte rg_mode10_cpp(const Byte* pSrc, int srcPitch) {
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
    
    if (mindiff == d7) return a7;
    if (mindiff == d8) return a8;
    if (mindiff == d6) return a6;
    if (mindiff == d2) return a2;
    if (mindiff == d3) return a3;
    if (mindiff == d1) return a1;
    if (mindiff == d5) return a5;
    return a4;
}

RG_FORCEINLINE uint16_t rg_mode10_cpp_16(const Byte* pSrc, int srcPitch) {
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

  if (mindiff == d7) return a7;
  if (mindiff == d8) return a8;
  if (mindiff == d6) return a6;
  if (mindiff == d2) return a2;
  if (mindiff == d3) return a3;
  if (mindiff == d1) return a1;
  if (mindiff == d5) return a5;
  return a4;
}

RG_FORCEINLINE float rg_mode10_cpp_32(const Byte* pSrc, int srcPitch) {
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

  if (mindiff == d7) return a7;
  if (mindiff == d8) return a8;
  if (mindiff == d6) return a6;
  if (mindiff == d2) return a2;
  if (mindiff == d3) return a3;
  if (mindiff == d1) return a1;
  if (mindiff == d5) return a5;
  return a4;
}

// ------------

RG_FORCEINLINE Byte rg_mode11_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);
    // fixme: SSE does not work like this!
    int	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
    int	val = (sum + 8) >> 4;

    return val;
}

RG_FORCEINLINE uint16_t rg_mode11_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  int	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
  int	val = (sum + 8) >> 4;

  return val;
}

RG_FORCEINLINE float rg_mode11_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
  float	val = sum / 16.0f; // (sum + 8) >> 4;

  return val;
}

// ------------

//rounding does not match
RG_FORCEINLINE Byte rg_mode12_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);


    int	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
    int	val = (sum + 8) >> 4;

    return val;
}

RG_FORCEINLINE uint16_t rg_mode12_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  int	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
  int	val = (sum + 8) >> 4;

  return val;
}

RG_FORCEINLINE float rg_mode12_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float	sum = 4 * c + 2 * (a2 + a4 + a5 + a7) + a1 + a3 + a6 + a8;
  float	val = sum / 16.0f; // (sum + 8) >> 4;

  return val;
}

// ------------

RG_FORCEINLINE Byte rg_mode13_and14_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::abs(a1 - a8);
    auto d2 = std::abs(a2 - a7);
    auto d3 = std::abs(a3 - a6);

    auto mindiff = std::min(std::min(d1, d2), d3);
    
    if (mindiff == d2) return (a2 + a7 + 1) / 2;
    if (mindiff == d3) return (a3 + a6 + 1) / 2;
    return (a1 + a8 + 1) / 2;
}

RG_FORCEINLINE uint16_t rg_mode13_and14_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::abs(a1 - a8);
  auto d2 = std::abs(a2 - a7);
  auto d3 = std::abs(a3 - a6);

  auto mindiff = std::min(std::min(d1, d2), d3);

  if (mindiff == d2) return (a2 + a7 + 1) / 2;
  if (mindiff == d3) return (a3 + a6 + 1) / 2;
  return (a1 + a8 + 1) / 2;
}

RG_FORCEINLINE float rg_mode13_and14_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::abs(a1 - a8);
  auto d2 = std::abs(a2 - a7);
  auto d3 = std::abs(a3 - a6);

  auto mindiff = std::min(std::min(d1, d2), d3);

  if (mindiff == d2) return (a2 + a7) / 2.0f; // (a2 + a7 + 1) / 2;
  if (mindiff == d3) return (a3 + a6) / 2.0f; // (a3 + a6 + 1) / 2;
  return (a1 + a8) / 2.0f; // (a1 + a8 + 1) / 2;
}

// ------------

//rounding does not match
RG_FORCEINLINE Byte rg_mode15_and16_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::abs(a1 - a8);
    auto d2 = std::abs(a2 - a7);
    auto d3 = std::abs(a3 - a6);

    auto mindiff = std::min(std::min(d1, d2), d3);

    auto average = (a1 + 2*a2 + a3 + a6 + 2*a7 + a8) / 8;

    if (mindiff == d2) return clip(average, (int)std::min(a2, a7), (int)std::max(a2, a7));
    if (mindiff == d3) return clip(average, (int)std::min(a3, a6), (int)std::max(a3, a6));
    return clip(average, (int)std::min(a1, a8), (int)std::max(a1, a8));
}

RG_FORCEINLINE uint16_t rg_mode15_and16_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::abs(a1 - a8);
  auto d2 = std::abs(a2 - a7);
  auto d3 = std::abs(a3 - a6);

  auto mindiff = std::min(std::min(d1, d2), d3);

  auto average = (a1 + 2*a2 + a3 + a6 + 2*a7 + a8 + 4) / 8;

  if (mindiff == d2) return clip_16(average, (int)std::min(a2, a7), (int)std::max(a2, a7));
  if (mindiff == d3) return clip_16(average, (int)std::min(a3, a6), (int)std::max(a3, a6));
  return clip_16(average, (int)std::min(a1, a8), (int)std::max(a1, a8));
}

RG_FORCEINLINE float rg_mode15_and16_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::abs(a1 - a8);
  auto d2 = std::abs(a2 - a7);
  auto d3 = std::abs(a3 - a6);

  auto mindiff = std::min(std::min(d1, d2), d3);

  auto average = (a1 + 2*a2 + a3 + a6 + 2*a7 + a8) / 8.0f;

  if (mindiff == d2) return clip_32(average, std::min(a2, a7), std::max(a2, a7));
  if (mindiff == d3) return clip_32(average, std::min(a3, a6), std::max(a3, a6));
  return clip_32(average, std::min(a1, a8), std::max(a1, a8));
}
// ------------

RG_FORCEINLINE Byte rg_mode17_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
    auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

    return clip(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE uint16_t rg_mode17_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  return clip_16(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE float rg_mode17_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto lower = std::max(std::max(std::max(mil1, mil2), mil3), mil4);
  auto upper = std::min(std::min(std::min(mal1, mal2), mal3), mal4);

  return clip_32(c, std::min(lower, upper), std::max(lower, upper));
}

// ------------

RG_FORCEINLINE Byte rg_mode18_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
    auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
    auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
    auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

    auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

    if (mindiff == d4) return clip(c, std::min(a4, a5),std::max(a4, a5));
    if (mindiff == d2) return clip(c, std::min(a2, a7),std::max(a2, a7));
    if (mindiff == d3) return clip(c, std::min(a3, a6),std::max(a3, a6));
    return clip(c, std::min(a1, a8), std::max(a1, a8));
}

RG_FORCEINLINE uint16_t rg_mode18_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
  auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
  auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
  auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_16(c, std::min(a4, a5),std::max(a4, a5));
  if (mindiff == d2) return clip_16(c, std::min(a2, a7),std::max(a2, a7));
  if (mindiff == d3) return clip_16(c, std::min(a3, a6),std::max(a3, a6));
  return clip_16(c, std::min(a1, a8), std::max(a1, a8));
}

RG_FORCEINLINE float rg_mode18_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto d1 = std::max(std::abs(c - a1), std::abs(c - a8));
  auto d2 = std::max(std::abs(c - a2), std::abs(c - a7));
  auto d3 = std::max(std::abs(c - a3), std::abs(c - a6));
  auto d4 = std::max(std::abs(c - a4), std::abs(c - a5));

  auto mindiff = std::min(std::min(std::min(d1, d2), d3), d4);

  if (mindiff == d4) return clip_32(c, std::min(a4, a5),std::max(a4, a5));
  if (mindiff == d2) return clip_32(c, std::min(a2, a7),std::max(a2, a7));
  if (mindiff == d3) return clip_32(c, std::min(a3, a6),std::max(a3, a6));
  return clip_32(c, std::min(a1, a8), std::max(a1, a8));
}

// ------------

RG_FORCEINLINE Byte rg_mode19_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);
    //this one is faster but rounded a bit differently. It's not like anyone will be using C anyway
    //int sum = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    //int val = (sum + 4) >> 3;

    int	val = ((((((a1 + a3 + 1) >> 1) + ((a6 + a8 + 1) >> 1) + 1) >> 1) - 1)
    						       +  ((((a2 + a5 + 1) >> 1) + ((a4 + a7 + 1) >> 1) + 1) >> 1)
    						       + 1) >> 1;

    return val;
}

RG_FORCEINLINE uint16_t rg_mode19_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);
  //this one is faster but rounded a bit differently. It's not like anyone will be using C anyway
  //int sum = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  //int val = (sum + 4) >> 3;

  int	val = ((((((a1 + a3 + 1) >> 1) + ((a6 + a8 + 1) >> 1) + 1) >> 1) - 1)
    +  ((((a2 + a5 + 1) >> 1) + ((a4 + a7 + 1) >> 1) + 1) >> 1)
    + 1) >> 1;

  return val;
}

RG_FORCEINLINE float rg_mode19_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float	val = (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) / 8.0f;

  return val;
}

// ------------

RG_FORCEINLINE Byte rg_mode20_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    int sum = a1 + a2 + a3 + a4 + c + a5 + a6 + a7 + a8;
    int val = (sum + 4) / 9;

    return val;
}

RG_FORCEINLINE uint16_t rg_mode20_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  int sum = a1 + a2 + a3 + a4 + c + a5 + a6 + a7 + a8;
  int val = (sum + 4) / 9;

  return val;
}

RG_FORCEINLINE float rg_mode20_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  float val = (a1 + a2 + a3 + a4 + c + a5 + a6 + a7 + a8) / 9.0f;
  // (sum + 4) / 9;

  return val;
}

// ------------

//rounding does not match
RG_FORCEINLINE Byte rg_mode21_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    int l1a = (a1 + a8) / 2;
    int l2a = (a2 + a7) / 2;
    int l3a = (a3 + a6) / 2;
    int l4a = (a4 + a5) / 2;

    int l1b = (a1 + a8 + 1) / 2;
    int l2b = (a2 + a7 + 1) / 2;
    int l3b = (a3 + a6 + 1) / 2;
    int l4b = (a4 + a5 + 1) / 2;

    auto ma = std::max(std::max(std::max(l1b, l2b), l3b), l4b);
    auto mi = std::min(std::min(std::min(l1a, l2a), l3a), l4a);

    return clip((int)c, mi, ma);
}

RG_FORCEINLINE uint16_t rg_mode21_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  int l1a = (a1 + a8) / 2;
  int l2a = (a2 + a7) / 2;
  int l3a = (a3 + a6) / 2;
  int l4a = (a4 + a5) / 2;

  int l1b = (a1 + a8 + 1) / 2;
  int l2b = (a2 + a7 + 1) / 2;
  int l3b = (a3 + a6 + 1) / 2;
  int l4b = (a4 + a5 + 1) / 2;

  auto ma = std::max(std::max(std::max(l1b, l2b), l3b), l4b);
  auto mi = std::min(std::min(std::min(l1a, l2a), l3a), l4a);

  return clip_16((int)c, mi, ma);
}

RG_FORCEINLINE float rg_mode21_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);
  // float: no integer tricks, same like 22
  float l1 = (a1 + a8) / 2.0f;
  float l2 = (a2 + a7) / 2.0f;
  float l3 = (a3 + a6) / 2.0f;
  float l4 = (a4 + a5) / 2.0f;

  auto ma = std::max(std::max(std::max(l1, l2), l3), l4);
  auto mi = std::min(std::min(std::min(l1, l2), l3), l4);

  return clip_32(c, mi, ma);
}


// ------------

RG_FORCEINLINE Byte rg_mode22_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    Byte l1 = (a1 + a8 + 1) / 2;
    Byte l2 = (a2 + a7 + 1) / 2;
    Byte l3 = (a3 + a6 + 1) / 2;
    Byte l4 = (a4 + a5 + 1) / 2;

    auto ma = std::max(std::max(std::max(l1, l2), l3), l4);
    auto mi = std::min(std::min(std::min(l1, l2), l3), l4);

    return clip(c, mi, ma);
}

RG_FORCEINLINE uint16_t rg_mode22_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  uint16_t l1 = (a1 + a8 + 1) / 2;
  uint16_t l2 = (a2 + a7 + 1) / 2;
  uint16_t l3 = (a3 + a6 + 1) / 2;
  uint16_t l4 = (a4 + a5 + 1) / 2;

  auto ma = std::max(std::max(std::max(l1, l2), l3), l4);
  auto mi = std::min(std::min(std::min(l1, l2), l3), l4);

  return clip_16(c, mi, ma);
}

RG_FORCEINLINE float rg_mode22_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  // float: no integer tricks, same as 21
  float l1 = (a1 + a8) / 2.0f;
  float l2 = (a2 + a7) / 2.0f;
  float l3 = (a3 + a6) / 2.0f;
  float l4 = (a4 + a5) / 2.0f;

  auto ma = std::max(std::max(std::max(l1, l2), l3), l4);
  auto mi = std::min(std::min(std::min(l1, l2), l3), l4);

  return clip_32(c, mi, ma);
}

// ------------

RG_FORCEINLINE Byte rg_mode23_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);
    
    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto linediff1 = subs_c(mal1, mil1);
    auto linediff2 = subs_c(mal2, mil2);
    auto linediff3 = subs_c(mal3, mil3);
    auto linediff4 = subs_c(mal4, mil4);

    auto u1 = std::min(subs_c(c, mal1), linediff1);
    auto u2 = std::min(subs_c(c, mal2), linediff2);
    auto u3 = std::min(subs_c(c, mal3), linediff3);
    auto u4 = std::min(subs_c(c, mal4), linediff4);
    auto u =  std::max(std::max(std::max(u1, u2), u3), u4);

    auto d1 = std::min(subs_c(mil1, c), linediff1);
    auto d2 = std::min(subs_c(mil2, c), linediff2);
    auto d3 = std::min(subs_c(mil3, c), linediff3);
    auto d4 = std::min(subs_c(mil4, c), linediff4);
    auto d = std::max(std::max(std::max(d1, d2), d3), d4);

    return adds_c(subs_c(c, u), d);
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t rg_mode23_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto linediff1 = subs_16_c(mal1, mil1);
  auto linediff2 = subs_16_c(mal2, mil2);
  auto linediff3 = subs_16_c(mal3, mil3);
  auto linediff4 = subs_16_c(mal4, mil4);

  auto u1 = std::min(subs_16_c(c, mal1), linediff1);
  auto u2 = std::min(subs_16_c(c, mal2), linediff2);
  auto u3 = std::min(subs_16_c(c, mal3), linediff3);
  auto u4 = std::min(subs_16_c(c, mal4), linediff4);
  auto u =  std::max(std::max(std::max(std::max(u1, u2), u3), u4), 0);

  auto d1 = std::min(subs_16_c(mil1, c), linediff1);
  auto d2 = std::min(subs_16_c(mil2, c), linediff2);
  auto d3 = std::min(subs_16_c(mil3, c), linediff3);
  auto d4 = std::min(subs_16_c(mil4, c), linediff4);
  auto d = std::max(std::max(std::max(std::max(d1, d2), d3), d4), 0);

  return adds_16_c<bits_per_pixel>(subs_16_c(c, u), d);
}

template<bool chroma>
RG_FORCEINLINE float rg_mode23_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto linediff1 = subs_32_c_for_diff(mal1, mil1);
  auto linediff2 = subs_32_c_for_diff(mal2, mil2);
  auto linediff3 = subs_32_c_for_diff(mal3, mil3);
  auto linediff4 = subs_32_c_for_diff(mal4, mil4);

  auto u1 = std::min(subs_32_c_for_diff(c, mal1), linediff1);
  auto u2 = std::min(subs_32_c_for_diff(c, mal2), linediff2);
  auto u3 = std::min(subs_32_c_for_diff(c, mal3), linediff3);
  auto u4 = std::min(subs_32_c_for_diff(c, mal4), linediff4);
  auto u =  std::max(std::max(std::max(std::max(u1, u2), u3), u4), 0.0f);

  auto d1 = std::min(subs_32_c_for_diff(mil1, c), linediff1);
  auto d2 = std::min(subs_32_c_for_diff(mil2, c), linediff2);
  auto d3 = std::min(subs_32_c_for_diff(mil3, c), linediff3);
  auto d4 = std::min(subs_32_c_for_diff(mil4, c), linediff4);
  auto d = std::max(std::max(std::max(std::max(d1, d2), d3), d4), 0.0f);

  return adds_32_c<chroma>(subs_32_c<chroma>(c, u), d);
}

// ------------
RG_FORCEINLINE Byte rg_mode24_cpp(const Byte* pSrc, int srcPitch) {
    LOAD_SQUARE_CPP(pSrc, srcPitch);

    auto mal1 = std::max(a1, a8);
    auto mil1 = std::min(a1, a8);

    auto mal2 = std::max(a2, a7);
    auto mil2 = std::min(a2, a7);

    auto mal3 = std::max(a3, a6);
    auto mil3 = std::min(a3, a6);

    auto mal4 = std::max(a4, a5);
    auto mil4 = std::min(a4, a5);

    auto linediff1 = subs_c(mal1, mil1);
    auto linediff2 = subs_c(mal2, mil2);
    auto linediff3 = subs_c(mal3, mil3);
    auto linediff4 = subs_c(mal4, mil4);

    auto t1 = subs_c(c, mal1);
    auto t2 = subs_c(c, mal2);
    auto t3 = subs_c(c, mal3);
    auto t4 = subs_c(c, mal4);

    auto u1 = std::min(t1, subs_c(linediff1, t1));
    auto u2 = std::min(t2, subs_c(linediff2, t2));
    auto u3 = std::min(t3, subs_c(linediff3, t3));
    auto u4 = std::min(t4, subs_c(linediff4, t4));
    auto u = std::max(std::max(std::max(std::max(u1, u2), u3), u4), 0);

    t1 = subs_c(mil1, c);
    t2 = subs_c(mil2, c);
    t3 = subs_c(mil3, c);
    t4 = subs_c(mil4, c);

    auto d1 = std::min(t1, subs_c(linediff1, t1));
    auto d2 = std::min(t2, subs_c(linediff2, t2));
    auto d3 = std::min(t3, subs_c(linediff3, t3));
    auto d4 = std::min(t4, subs_c(linediff4, t4));
    auto d = std::max(std::max(std::max(std::max(d1, d2), d3), d4), 0);

    return adds_c(subs_c(c, u), d);
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t rg_mode24_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto linediff1 = subs_16_c(mal1, mil1);
  auto linediff2 = subs_16_c(mal2, mil2);
  auto linediff3 = subs_16_c(mal3, mil3);
  auto linediff4 = subs_16_c(mal4, mil4);

  auto t1 = subs_16_c(c, mal1);
  auto t2 = subs_16_c(c, mal2);
  auto t3 = subs_16_c(c, mal3);
  auto t4 = subs_16_c(c, mal4);

  auto u1 = std::min(t1, subs_16_c(linediff1, t1));
  auto u2 = std::min(t2, subs_16_c(linediff2, t2));
  auto u3 = std::min(t3, subs_16_c(linediff3, t3));
  auto u4 = std::min(t4, subs_16_c(linediff4, t4));
  auto u = std::max(std::max(std::max(std::max(u1, u2), u3), u4), 0);

  t1 = subs_16_c(mil1, c);
  t2 = subs_16_c(mil2, c);
  t3 = subs_16_c(mil3, c);
  t4 = subs_16_c(mil4, c);

  auto d1 = std::min(t1, subs_16_c(linediff1, t1));
  auto d2 = std::min(t2, subs_16_c(linediff2, t2));
  auto d3 = std::min(t3, subs_16_c(linediff3, t3));
  auto d4 = std::min(t4, subs_16_c(linediff4, t4));
  auto d = std::max(std::max(std::max(std::max(d1, d2), d3), d4), 0);

  return adds_16_c<bits_per_pixel>(subs_16_c(c, u), d);
}

template<bool chroma>
RG_FORCEINLINE float rg_mode24_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  auto mal1 = std::max(a1, a8);
  auto mil1 = std::min(a1, a8);

  auto mal2 = std::max(a2, a7);
  auto mil2 = std::min(a2, a7);

  auto mal3 = std::max(a3, a6);
  auto mil3 = std::min(a3, a6);

  auto mal4 = std::max(a4, a5);
  auto mil4 = std::min(a4, a5);

  auto linediff1 = subs_32_c_for_diff(mal1, mil1);
  auto linediff2 = subs_32_c_for_diff(mal2, mil2);
  auto linediff3 = subs_32_c_for_diff(mal3, mil3);
  auto linediff4 = subs_32_c_for_diff(mal4, mil4);

  auto t1 = subs_32_c_for_diff(c, mal1);
  auto t2 = subs_32_c_for_diff(c, mal2);
  auto t3 = subs_32_c_for_diff(c, mal3);
  auto t4 = subs_32_c_for_diff(c, mal4);

  auto u1 = std::min(t1, subs_32_c_for_diff(linediff1, t1));
  auto u2 = std::min(t2, subs_32_c_for_diff(linediff2, t2));
  auto u3 = std::min(t3, subs_32_c_for_diff(linediff3, t3));
  auto u4 = std::min(t4, subs_32_c_for_diff(linediff4, t4));
  auto u = std::max(std::max(std::max(std::max(u1, u2), u3), u4), 0.0f);

  t1 = subs_32_c_for_diff(mil1, c);
  t2 = subs_32_c_for_diff(mil2, c);
  t3 = subs_32_c_for_diff(mil3, c);
  t4 = subs_32_c_for_diff(mil4, c);

  auto d1 = std::min(t1, subs_32_c_for_diff(linediff1, t1));
  auto d2 = std::min(t2, subs_32_c_for_diff(linediff2, t2));
  auto d3 = std::min(t3, subs_32_c_for_diff(linediff3, t3));
  auto d4 = std::min(t4, subs_32_c_for_diff(linediff4, t4));
  auto d = std::max(std::max(std::max(std::max(d1, d2), d3), d4), 0.0f);

  return adds_32_c<chroma>(subs_32_c<chroma>(c, u), d);
}

// ------------

RG_FORCEINLINE Byte rg_mode25_cpp(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP(pSrc, srcPitch);

  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  int SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  int SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results

  neighbourdiff_c<8>(SSE4, SSE5, c, a4);
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff_c<8>(SSE6, SSE7, c, a5);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a1);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a2);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a3);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a6);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a7);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<8>(SSE6, SSE7, c, a8);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  auto result = sharpen_c<8>(c, SSE4, SSE5);

  return result;
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t rg_mode25_cpp_16(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_16(pSrc, srcPitch);

  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  int SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  int SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results

  neighbourdiff_c<bits_per_pixel>(SSE4, SSE5, c, a4);
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a5);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a1);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a2);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a3);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a6);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a7);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_c<bits_per_pixel>(SSE6, SSE7, c, a8);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  auto result = sharpen_c<bits_per_pixel>(c, SSE4, SSE5);

  return result;
}

template<bool chroma>
RG_FORCEINLINE float rg_mode25_cpp_32(const Byte* pSrc, int srcPitch) {
  LOAD_SQUARE_CPP_32(pSrc, srcPitch);

  /*
  a1 a2 a3
  a4 c  a5
  a6 a7 a8
  */
  float SSE4, SSE5; // SSE4_minus, SSE5_plus; // global collectors
  float SSE6, SSE7; // SSE6_actual_minus, SSE7_actual_plus the actual results

  neighbourdiff_32_c(SSE4, SSE5, c, a4);
  // first result fill into collectors SSE4 and SSE5, no comparison

  neighbourdiff_32_c(SSE6, SSE7, c, a5);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a1);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a2);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a3);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a6);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a7);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  neighbourdiff_32_c(SSE6, SSE7, c, a8);
  SSE4 = std::min(SSE4, SSE6);
  SSE5 = std::min(SSE5, SSE7);

  auto result = sharpen_32_c<chroma>(c, SSE4, SSE5);

  return result;
}

// ------------

RG_FORCEINLINE Byte rg_mode26_cpp(const Byte* pSrc, int srcPitch) {
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

  return clip(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE uint16_t rg_mode26_cpp_16(const Byte* pSrc, int srcPitch) {
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

  return clip_16(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE float rg_mode26_cpp_32(const Byte* pSrc, int srcPitch) {
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

  return clip_32(c, std::min(lower, upper), std::max(lower, upper));
}

// ------------

RG_FORCEINLINE Byte rg_mode27_cpp(const Byte* pSrc, int srcPitch) {
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

  return clip(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE uint16_t rg_mode27_cpp_16(const Byte* pSrc, int srcPitch) {
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

  return clip_16(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE float rg_mode27_cpp_32(const Byte* pSrc, int srcPitch) {
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

  return clip_32(c, std::min(lower, upper), std::max(lower, upper));
}

// ------------

RG_FORCEINLINE Byte rg_mode28_cpp(const Byte* pSrc, int srcPitch) {
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

  return clip(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE uint16_t rg_mode28_cpp_16(const Byte* pSrc, int srcPitch) {
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

  return clip_16(c, std::min(lower, upper), std::max(lower, upper));
}

RG_FORCEINLINE float rg_mode28_cpp_32(const Byte* pSrc, int srcPitch) {
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

  return clip_32(c, std::min(lower, upper), std::max(lower, upper));
}

#undef LOAD_SQUARE_CPP
#undef LOAD_SQUARE_CPP_16
#undef LOAD_SQUARE_CPP_32
#undef LOAD_SQUARE_CPP_T

#endif