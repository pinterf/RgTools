// TemporalRepair from RemoveGrainT 1.0 package"
//
// An Avisynth plugin for removing grain from progressive video
//
// 2007 By Rainer Wittmann <gorw@gmx.de>
// 2019 Rewrite and additional work by Ferenc Pintér

// TemporalRepair core: C kernels, kernel selection

#include "RemoveGrainT_core.h"
#include "common.h"
#include <cstdio>

typedef int (temporal_repair_processor_c)(int src_1, int src_2, int src_prev, int src_next);
typedef float (temporal_repair_processor_32_c)(float src_1, float src_2, float src_prev, float src_next);

typedef int (smooth_temporal_repair_processor_c)(uint8_t* dp, const uint8_t* previous, const intptr_t pfpitch, const uint8_t* sp, const intptr_t ofpitch, const uint8_t* next, const intptr_t nfpitch);
typedef float (smooth_temporal_repair_processor_32_c)(uint8_t* dp, const uint8_t* previous, const intptr_t pfpitch, const uint8_t* sp, const intptr_t ofpitch, const uint8_t* next, const intptr_t nfpitch);

/*************************************

    Helpers

*************************************/


/*************************************

    TemporalRepair, Mode 0, Mode 4

*************************************/

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

template<typename pixel_t, temporal_repair_processor_c processor>
void temporal_repair_mode0and4_c(uint8_t* dp8, int dpitch, const uint8_t* sp1_8, int spitch1, const uint8_t* sp2_8, int spitch2, const uint8_t* pp8, int ppitch, const uint8_t* np8, int npitch, int width, int height)
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
void temporal_repair_mode0and4_32_c(uint8_t* dp8, int dpitch, const uint8_t* sp1_8, int spitch1, const uint8_t* sp2_8, int spitch2, const uint8_t* pp8, int ppitch, const uint8_t* np8, int npitch, int width, int height)
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

// 10-16 bit: one variant. The sums saturate at 16 bit (as _mm_adds_epu16) and not at the max pixel value of the
// bit depth: they are only clip bounds (modes 1-3: the upper bound of a clip of a pixel value <= max; mode 4: limited
// by min(.., max_np) or subtracted with saturation from max_np), a bound above the max gives the same result.
RG_FORCEINLINE int BRepairPixel_mode4_16_c(int src_1, int src_2, int src_prev, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);

  auto diff_src2_minnp = subs_16_c(src_2, min_np);
  auto reg5 = adds_16_c(adds_16_c(diff_src2_minnp, diff_src2_minnp, 65535), min_np, 65535); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_16_c(max_np, src_2);
  auto reg3 = subs_16_c(max_np, adds_16_c(diff_maxnp_src2, diff_maxnp_src2, 65535)); // max_np - 2 * diff_maxnp_src2

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

  auto diff_src2_minnp = subs_32_c_for_diff(src_2, min_np); // not really subsat, ordinary difference, clamped to zero
  auto reg5 = adds_32_c<chroma>(
    adds_32_c_for_diff(diff_src2_minnp, diff_src2_minnp),  // luma-like addsat, simple mul 2
    min_np
    ); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_32_c_for_diff(max_np, src_2); // not really subsat, ordinary difference, clamped to zero
  auto reg3 = subs_32_c<chroma>(
    max_np, 
    adds_32_c_for_diff(diff_maxnp_src2, diff_maxnp_src2) // luma-like addsat, simple mul 2
    ); // max_np - 2 * diff_maxnp_src2

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
    temporal_repair_mode0and4_c<uint16_t, BRepairPixel_mode4_16_c>
};

static PlaneProcessor_t* t_c_functions_32[] = {
    temporal_repair_mode0and4_32_c<float, RepairPixel_mode0_32_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_32_c<float, BRepairPixel_mode4_32_c<false>> // false: not chroma
};

// 32 bit float luma
static PlaneProcessor_t* t_c_functions_chroma_32[] = {
    temporal_repair_mode0and4_32_c<float, RepairPixel_mode0_32_c>,
    nullptr,
    nullptr,
    nullptr,
    temporal_repair_mode0and4_32_c<float, BRepairPixel_mode4_32_c<true>> // false: not chroma
};
/*****************************************

    SmoothTemporalRepair, Mode 1, 2, 3

*****************************************/


RG_FORCEINLINE void get_lu_c(int& lower, int& upper, int src_prev, int src_curr, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);
  upper = subs_c(max_np, src_curr);
  lower = subs_c(src_curr, min_np);
}

RG_FORCEINLINE void get_lu_16_c(int& lower, int& upper, int src_prev, int src_curr, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);
  upper = subs_16_c(max_np, src_curr);
  lower = subs_16_c(src_curr, min_np);
}


RG_FORCEINLINE void get_lu_32_c(float& lower, float& upper, float src_prev, float src_curr, float src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);
  upper = subs_32_c_for_diff(max_np, src_curr); // not really subsat, ordinary difference, clamped to zero
  lower = subs_32_c_for_diff(src_curr, min_np);
}

RG_FORCEINLINE int SmoothTRepair1_c(int dest, int lower, int upper, const int src_prev, const int src_curr, const int src_next)
{
  auto src_dest = dest;

  auto tmp_u = adds_c(upper, src_curr);
  auto tmp_l = subs_c(src_curr, lower);

  auto tmp_max = std::max(std::max(tmp_u, src_prev), src_next);
  auto tmp_min = std::min(std::min(tmp_l, src_prev), src_next);

  auto result = clip(src_dest, tmp_min, tmp_max);

  return result;
}

RG_FORCEINLINE int SmoothTRepair1_16_c(int dest, int lower, int upper, const int src_prev, const int src_curr, const int src_next)
{
  auto src_dest = dest;

  auto tmp_u = adds_16_c(upper, src_curr, 65535);
  auto tmp_l = subs_16_c(src_curr, lower);

  auto tmp_max = std::max(std::max(tmp_u, src_prev), src_next);
  auto tmp_min = std::min(std::min(tmp_l, src_prev), src_next);

  auto result = clip_16(src_dest, tmp_min, tmp_max);

  return result;
}

template<bool chroma>
RG_FORCEINLINE float SmoothTRepair1_32_c(float dest, float lower, float upper, const float src_prev, const float src_curr, const float src_next)
{
  auto src_dest = dest;

  auto tmp_u = adds_32_c<chroma>(upper, src_curr); // really addsat
  auto tmp_l = subs_32_c<chroma>(src_curr, lower); // really subsat

  auto tmp_max = std::max(std::max(tmp_u, src_prev), src_next);
  auto tmp_min = std::min(std::min(tmp_l, src_prev), src_next);

  auto result = clip_32(src_dest, tmp_min, tmp_max);

  return result;
}

RG_FORCEINLINE int SmoothTRepair2_8_c(int dest, int lower, int upper, int src_prev, int src_curr, int src_next)
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

RG_FORCEINLINE int SmoothTRepair2_16_c(int dest, int lower, int upper, int src_prev, int src_curr, int src_next)
{
  int lower1;
  int upper1;

  auto src_dest = dest;

  get_lu_16_c(lower1, upper1, src_prev, src_curr, src_next);
  auto uppermax = std::max(upper, upper1);
  auto lowermax = std::max(lower, lower1);
  auto upperlowermax = std::max(uppermax, lowermax);

  auto tmp_max = adds_16_c(src_curr, upperlowermax, 65535);
  auto tmp_min = subs_16_c(src_curr, upperlowermax);

  auto result = clip_16(src_dest, tmp_min, tmp_max);
  return result;
}

template<bool chroma>
RG_FORCEINLINE float SmoothTRepair2_32_c(float dest, float lower, float upper, float src_prev, float src_curr, float src_next)
{
  float lower1;
  float upper1;

  auto src_dest = dest;

  get_lu_32_c(lower1, upper1, src_prev, src_curr, src_next);
  auto uppermax = std::max(upper, upper1);
  auto lowermax = std::max(lower, lower1);
  auto upperlowermax = std::max(uppermax, lowermax);

  auto tmp_max = adds_32_c<chroma>(src_curr, upperlowermax);
  auto tmp_min = subs_32_c<chroma>(src_curr, upperlowermax);

  auto result = clip_32(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE void get2diff_8_c(int& pdiff, int& ndiff, int src_prev, int src_curr, int src_next)
{
  pdiff = std::abs(src_curr - src_prev); 
  ndiff = std::abs(src_curr - src_next);
}

RG_FORCEINLINE void get2diff_16_c(int& pdiff, int& ndiff, int src_prev, int src_curr, int src_next)
{
  pdiff = std::abs(src_curr - src_prev);
  ndiff = std::abs(src_curr - src_next);
}

RG_FORCEINLINE void get2diff_32_c(float& pdiff, float& ndiff, float src_prev, float src_curr, float src_next)
{
  pdiff = std::abs(src_curr - src_prev);
  ndiff = std::abs(src_curr - src_next);
}

RG_FORCEINLINE int SmoothTRepair3_c(int dest, int pmax, int nmax, int src_prev, int src_curr, int src_next)
{
  int pdiff;
  int ndiff;

  get2diff_8_c(pdiff, ndiff, src_prev, src_curr, src_next);
  pmax = std::max(pmax, pdiff);
  nmax = std::max(nmax, ndiff);

  pmax = std::min(pmax, nmax);
  auto src_dest = dest;
  auto tmp_max = adds_c(src_curr, pmax);
  auto tmp_min = subs_c(src_curr, pmax);
  auto result = clip(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE int SmoothTRepair3_16_c(int dest, int pmax, int nmax, int src_prev, int src_curr, int src_next)
{
  int pdiff;
  int ndiff;

  get2diff_16_c(pdiff, ndiff, src_prev, src_curr, src_next);
  pmax = std::max(pmax, pdiff);
  nmax = std::max(nmax, ndiff);

  pmax = std::min(pmax, nmax);
  auto src_dest = dest;
  auto tmp_max = adds_16_c(src_curr, pmax, 65535);
  auto tmp_min = subs_16_c(src_curr, pmax);
  auto result = clip_16(src_dest, tmp_min, tmp_max);
  return result;
}

template<bool chroma>
RG_FORCEINLINE float SmoothTRepair3_32_c(float dest, float pmax, float nmax, float src_prev, float src_curr, float src_next)
{
  float pdiff;
  float ndiff;

  get2diff_32_c(pdiff, ndiff, src_prev, src_curr, src_next);
  pmax = std::max(pmax, pdiff);
  nmax = std::max(nmax, ndiff);

  pmax = std::min(pmax, nmax);
  auto src_dest = dest;
  auto tmp_max = adds_32_c<chroma>(src_curr, pmax);
  auto tmp_min = subs_32_c<chroma>(src_curr, pmax);
  auto result = clip_32(src_dest, tmp_min, tmp_max);
  return result;
}

RG_FORCEINLINE int temporal_repair_processor_mode1_8_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
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

RG_FORCEINLINE int temporal_repair_processor_mode1_16_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = uint16_t;
  constexpr int pixelsize = sizeof(pixel_t);

  int lowermax, uppermax;
  int lower, upper;
  get_lu_16_c(lowermax, uppermax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch - pixelsize],
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch - pixelsize], 
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch + pixelsize], 
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair1_16_c(
    *(pixel_t*)& dp[0], 
    lowermax, uppermax,
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

template<bool chroma>
RG_FORCEINLINE float temporal_repair_processor_mode1_32_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = float;
  constexpr int pixelsize = sizeof(pixel_t);

  float lowermax, uppermax;
  float lower, upper;
  get_lu_32_c(lowermax, uppermax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch - pixelsize], 
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch + pixelsize], 
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair1_32_c<chroma>(
    *(pixel_t*)& dp[0], 
    lowermax, uppermax, 
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode2_8_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
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
  return SmoothTRepair2_8_c(dp[0], lowermax, uppermax, previous[0 * pfpitch + 0], sp[0 * ofpitch + 0], next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode2_16_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = uint16_t;
  constexpr int pixelsize = sizeof(pixel_t);

  int lowermax, uppermax;
  int lower, upper;
  get_lu_16_c(lowermax, uppermax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize],
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize],
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize],
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize],
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize],
    *(pixel_t*)& sp[1 * ofpitch - pixelsize],
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize],
    *(pixel_t*)& sp[1 * ofpitch + pixelsize],
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize],
    *(pixel_t*)& sp[0 * ofpitch - pixelsize],
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_16_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize],
    *(pixel_t*)& sp[0 * ofpitch + pixelsize],
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair2_16_c(
    *(pixel_t*)& dp[0], 
    lowermax, uppermax, 
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

template<bool chroma>
RG_FORCEINLINE float temporal_repair_processor_mode2_32_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = float;
  constexpr int pixelsize = sizeof(pixel_t);

  float lowermax, uppermax;
  float lower, upper;
  get_lu_32_c(lowermax, uppermax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch - pixelsize], 
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  get_lu_32_c(lower, upper, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch + pixelsize], 
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  uppermax = std::max(uppermax, upper);
  lowermax = std::max(lowermax, lower);
  return SmoothTRepair2_32_c<chroma>(
    *(pixel_t*)& dp[0], 
    lowermax, uppermax, 
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode3_8_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  int pdiffmax;
  int ndiffmax;
  int pdiff;
  int ndiff;
  get2diff_8_c(pdiffmax, ndiffmax, previous[-1 * pfpitch - 1], sp[-1 * ofpitch - 1], next[-1 * nfpitch - 1]);
  get2diff_8_c(pdiff, ndiff, previous[-1 * pfpitch + 0], sp[-1 * ofpitch + 0], next[-1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[-1 * pfpitch + 1], sp[-1 * ofpitch + 1], next[-1 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[1 * pfpitch - 1], sp[1 * ofpitch - 1], next[1 * nfpitch - 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[1 * pfpitch + 0], sp[1 * ofpitch + 0], next[1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[1 * pfpitch + 1], sp[1 * ofpitch + 1], next[1 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[0 * pfpitch - 1], sp[0 * ofpitch - 1], next[0 * nfpitch - 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_8_c(pdiff, ndiff, previous[0 * pfpitch + 1], sp[0 * ofpitch + 1], next[0 * nfpitch + 1]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  return SmoothTRepair3_c(dp[0], pdiffmax, ndiffmax, previous[0 * pfpitch + 0], sp[0 * ofpitch + 0], next[0 * nfpitch + 0]);
}

RG_FORCEINLINE int temporal_repair_processor_mode3_16_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = uint16_t;
  constexpr int pixelsize = sizeof(pixel_t);

  int pdiffmax;
  int ndiffmax;
  int pdiff;
  int ndiff;
  get2diff_16_c(pdiffmax, ndiffmax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch - pixelsize], 
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_16_c(pdiff, ndiff, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch + pixelsize], 
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  return SmoothTRepair3_16_c(
    *(pixel_t*)& dp[0], 
    pdiffmax, ndiffmax, 
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

template<bool chroma>
RG_FORCEINLINE float temporal_repair_processor_mode3_32_c(
  uint8_t* dp,
  const uint8_t* previous, const intptr_t pfpitch,
  const uint8_t* sp, const intptr_t ofpitch,
  const uint8_t* next, const intptr_t nfpitch
)
{
  using pixel_t = float;
  constexpr int pixelsize = sizeof(pixel_t);

  float pdiffmax;
  float ndiffmax;
  float pdiff;
  float ndiff;
  get2diff_32_c(pdiffmax, ndiffmax, 
    *(pixel_t*)& previous[-1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch - pixelsize]);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[-1 * pfpitch + 0], 
    *(pixel_t*)& sp[-1 * ofpitch + 0], 
    *(pixel_t*)& next[-1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[-1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[-1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[-1 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch - pixelsize], 
    *(pixel_t*)& next[1 * nfpitch - pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch + 0], 
    *(pixel_t*)& sp[1 * ofpitch + 0], 
    *(pixel_t*)& next[1 * nfpitch + 0]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[1 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[1 * ofpitch + pixelsize], 
    *(pixel_t*)& next[1 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[0 * pfpitch - pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch - pixelsize], 
    *(pixel_t*)& next[0 * nfpitch - pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  get2diff_32_c(pdiff, ndiff, 
    *(pixel_t*)& previous[0 * pfpitch + pixelsize], 
    *(pixel_t*)& sp[0 * ofpitch + pixelsize], 
    *(pixel_t*)& next[0 * nfpitch + pixelsize]);
  pdiffmax = std::max(pdiffmax, pdiff);
  ndiffmax = std::max(ndiffmax, ndiff);
  return SmoothTRepair3_32_c<chroma>(
    *(pixel_t*)& dp[0], 
    pdiffmax, ndiffmax, 
    *(pixel_t*)& previous[0 * pfpitch + 0], 
    *(pixel_t*)& sp[0 * ofpitch + 0], 
    *(pixel_t*)& next[0 * nfpitch + 0]);
}

// for all integer 8-16 bits
template<typename pixel_t, smooth_temporal_repair_processor_c processor>
void smooth_temporal_repair_mode1to3_c(uint8_t* dp, const uint8_t* previous, const uint8_t* sp, const uint8_t* next, intptr_t* pitches, int width, int height)
{
  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0];
  const intptr_t pfpitch = pitches[1];
  const intptr_t ofpitch = pitches[2];
  const intptr_t nfpitch = pitches[3];

  constexpr int pixelsize = sizeof(pixel_t);

  dp += dppitch;
  previous += pfpitch;
  sp += ofpitch;
  next += nfpitch;

  for (int y = 1; y < height - 1; y++)
  {
    for (int x = 1; x < width - 1; x++)
    {
      *(pixel_t *)(dp + x * pixelsize) = processor(dp + x * pixelsize, previous + x * pixelsize, pfpitch, sp + x * pixelsize, ofpitch, next + x * pixelsize, nfpitch);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

// specially for 32 bit float
template<smooth_temporal_repair_processor_32_c processor>
void smooth_temporal_repair_mode1to3_32_c(uint8_t* dp, const uint8_t* previous, const uint8_t* sp, const uint8_t* next, intptr_t* pitches, int width, int height)
{
  using pixel_t = float;

  // #: original_dp, previous, sp, next
  const intptr_t dppitch = pitches[0];
  const intptr_t pfpitch = pitches[1];
  const intptr_t ofpitch = pitches[2];
  const intptr_t nfpitch = pitches[3];

  constexpr int pixelsize = sizeof(pixel_t);

  dp += dppitch;
  previous += pfpitch;
  sp += ofpitch;
  next += nfpitch;

  for (int y = 1; y < height - 1; y++)
  {
    for (int x = 1; x < width - 1; x++)
    {
      *(pixel_t*)(dp + x * pixelsize) = processor(dp + x * pixelsize, previous + x * pixelsize, pfpitch, sp + x * pixelsize, ofpitch, next + x * pixelsize, nfpitch);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

static PlaneProcessor_st* st_c_functions[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode1_8_c>,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode2_8_c>,
    smooth_temporal_repair_mode1to3_c<uint8_t, temporal_repair_processor_mode3_8_c>
};

static PlaneProcessor_st* st_c_functions_16[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_c<uint16_t, temporal_repair_processor_mode1_16_c>,
    smooth_temporal_repair_mode1to3_c<uint16_t, temporal_repair_processor_mode2_16_c>,
    smooth_temporal_repair_mode1to3_c<uint16_t, temporal_repair_processor_mode3_16_c>
};

// 32 bit float luma
static PlaneProcessor_st* st_c_functions_32[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode1_32_c<false>>,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode2_32_c<false>>,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode3_32_c<false>>
};

// 32 bit float chroma
static PlaneProcessor_st* st_c_functions_chroma_32[] = {
    nullptr,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode1_32_c<true>>,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode2_32_c<true>>,
    smooth_temporal_repair_mode1to3_32_c<temporal_repair_processor_mode3_32_c<true>>
};

const char* TemporalRepairCore::init(const TemporalRepairParams& params, const RgFormat& format, int cpu_flags)
{
  constexpr int MAXTMODE = 4;
  const int mode = params.mode;
  const int opt = params.opt;
  if ((unsigned)mode > MAXTMODE) {
    snprintf(error_, sizeof(error_), "TemporalRepair: illegal mode %i", mode);
    return error_;
  }

  format_ = format;
  grey_ = params.grey;
  smooth_ = mode >= 1 && mode <= 3;

#ifdef INTEL_INTRINSICS
  bool sse2 = (cpu_flags & RG_CPU_SSE2) != 0;
  bool sse41 = (cpu_flags & RG_CPU_SSE41) != 0;

  // disable by opt
  if (opt >= 0) {
    if (opt < 1) sse2 = false;
    if (opt < 2) sse41 = false;
  }
#else
  (void)cpu_flags;
#endif
#ifdef NEON_INTRINSICS
  const bool neon = (cpu_flags & RG_CPU_NEON) != 0 && opt != 0; // 0: C
#endif

  if (!smooth_) {
    // modes 0 and 4
    switch (format.bits_per_pixel) {
    case 8:
      t_c_ = t_c_functions[mode];
      t_chroma_c_ = t_c_;
      break;
    case 32:
      t_c_ = t_c_functions_32[mode];
      t_chroma_c_ = t_c_functions_chroma_32[mode]; // float: special
      break;
    default: // 10-16 bit
      t_c_ = t_c_functions_16[mode];
      t_chroma_c_ = t_c_;
      break;
    }

    if (opt == 0) { // 0: C
      t_ = t_c_;
      t_chroma_ = t_chroma_c_;
    }
    else {
      switch (format.bits_per_pixel) {
      case 8:
#ifdef INTEL_INTRINSICS
        if (sse41)
          t_ = t_sse41_functions[mode];
        else if (sse2)
          t_ = t_sse2_functions[mode];
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon)
          t_ = t_neon_functions[mode];
        else
#endif
          t_ = t_c_;
        t_chroma_ = t_;
        break;
      case 32:
#ifdef INTEL_INTRINSICS
        if (sse41) {
          t_ = t_sse41_functions_32[mode];
          t_chroma_ = t_sse41_functions_chroma_32[mode];
        }
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon) {
          t_ = t_neon_functions_32[mode];
          t_chroma_ = t_neon_functions_chroma_32[mode];
        }
        else
#endif
        {
          t_ = t_c_;
          t_chroma_ = t_chroma_c_;
        }
        break;
      default: // 10-16 bit
#ifdef INTEL_INTRINSICS
        if (sse41)
          t_ = t_sse41_functions_16[mode];
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon)
          t_ = t_neon_functions_16[mode];
        else
#endif
          t_ = t_c_;
        t_chroma_ = t_;
        break;
      }
    }
  }
  else {
    // modes 1, 2 and 3
    switch (format.bits_per_pixel) {
    case 8:
      st_c_ = st_c_functions[mode];
      st_chroma_c_ = st_c_;
      break;
    case 32:
      st_c_ = st_c_functions_32[mode];
      st_chroma_c_ = st_c_functions_chroma_32[mode]; // float: special
      break;
    default: // 10-16 bit
      st_c_ = st_c_functions_16[mode];
      st_chroma_c_ = st_c_;
      break;
    }

    if (opt == 0) { // 0: C
      st_ = st_c_;
      st_chroma_ = st_chroma_c_;
    }
    else {
      switch (format.bits_per_pixel) {
      case 8:
#ifdef INTEL_INTRINSICS
        if (sse2)
          st_ = st_sse2_functions[mode];
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon)
          st_ = st_neon_functions[mode];
        else
#endif
          st_ = st_c_;
        st_chroma_ = st_;
        break;
      case 32:
#ifdef INTEL_INTRINSICS
        if (sse2) {
          st_ = st_sse2_functions_32[mode];
          st_chroma_ = st_sse2_functions_chroma_32[mode];
        }
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon) {
          st_ = st_neon_functions_32[mode];
          st_chroma_ = st_neon_functions_chroma_32[mode];
        }
        else
#endif
        {
          st_ = st_c_;
          st_chroma_ = st_chroma_c_;
        }
        break;
      default: // 10-16 bit
#ifdef INTEL_INTRINSICS
        if (sse41)
          st_ = st_sse41_functions_16[mode];
        else
#endif
#ifdef NEON_INTRINSICS
        if (neon)
          st_ = st_neon_functions_16[mode];
        else
#endif
          st_ = st_c_;
        st_chroma_ = st_;
        break;
      }
    }
  }
  return nullptr;
}

void TemporalRepairCore::process_plane(int plane_index, const RgPlane& plane, const uint8_t* prev, int prev_pitch,
  const uint8_t* cur, int cur_pitch, const uint8_t* next, int next_pitch) const
{
  if (grey_ && plane_index > 0)
    return; // grey: not processed, not copied

  const bool chroma = format_.family == RgColorFamily::YUV && plane_index > 0;
  const int width = plane.rowsize / format_.pixelsize;

  if (smooth_) {
    // the border pixels keep the clip
    rg_copy_rows(plane.dst, plane.dst_pitch, plane.src, plane.src_pitch, plane.rowsize, plane.height);

    intptr_t pitches[4] = { plane.dst_pitch, prev_pitch, cur_pitch, next_pitch };
    PlaneProcessor_st* processor;
    // minimum byte size for simd
    if (plane.rowsize < format_.pixelsize + 16 + format_.pixelsize)
      processor = chroma ? st_chroma_c_ : st_c_;
    else
      processor = chroma ? st_chroma_ : st_;
    // planes smaller than 3x3 have no inner pixel: the processors leave the copied plane as is
    processor(plane.dst, prev, cur, next, pitches, width, plane.height);
  }
  else {
    PlaneProcessor_t* processor;
    // minimum byte size for simd
    if (plane.rowsize < 16)
      processor = chroma ? t_chroma_c_ : t_c_;
    else
      processor = chroma ? t_chroma_ : t_;
    processor(plane.dst, plane.dst_pitch, plane.src, plane.src_pitch, cur, cur_pitch, prev, prev_pitch, next, next_pitch, width, plane.height);
  }
}
