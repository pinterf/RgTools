// TemporalRepair from RemoveGrainT 1.0 package"
//
// An Avisynth plugin for removing grain from progressive video
//
// 2007 By Rainer Wittmann <gorw@gmx.de>
// 2019 Additional work by Ferenc Pintér
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

#include "avisynth.h"
#include "common.h"
#include <Windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <cassert>
#include "RemoveGrainT.h"
#include "stdint.h"
#include "emmintrin.h"
#include "immintrin.h"
#include <math.h>
#include <algorithm>


RG_FORCEINLINE void RepairPixel(uint8_t *dest, const uint8_t *src1, const uint8_t *src2, const uint8_t *previous, const uint8_t *next)
{
	auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
	auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
	auto src_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1));
	auto src_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2));

	auto min_np2 = _mm_min_epu8(_mm_min_epu8(src_next, src_prev), src_2);
	auto max_np2 = _mm_max_epu8(_mm_max_epu8(src_next, src_prev), src_2);
	auto reg1 = _mm_max_epu8(min_np2, src_1);
	auto result = _mm_min_epu8(reg1, max_np2);

	_mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

static void temporal_repair_mode0_sse2(BYTE *dp, int dpitch, const BYTE *sp1, int spitch1, const BYTE *sp2, int spitch2, const BYTE *pp, int ppitch, const BYTE *np, int npitch, int width, int height)
{
  assert(width >= 16);

  const int wmod16 = width / 16 * 16;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod16; x += 16)
    {
      RepairPixel(dp + x, sp1 + x, sp2 + x, pp + x, np + x);
    }
    // remainder, surely unaligned, overlaps but still simd
    if (width > wmod16) {
      const int last16 = width - 16;
      RepairPixel(dp + last16, sp1 + last16, sp2 + last16, pp + last16, np + last16);
    }
    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }
}

RG_FORCEINLINE int RepairPixel_c(int src_1, int src_2, int src_prev, int src_next)
{

	auto min_np2 = std::min(std::min(src_next, src_prev), src_2); //_mm_min_epu8(_mm_min_epu8(src_next, src_prev), src_2);
	auto max_np2 = std::max(std::max(src_next, src_prev), src_2);
	auto reg1 = std::max(min_np2, src_1);
	return std::min(reg1, max_np2);
}

static void temporal_repair_mode0_c(BYTE* dp, int dpitch, const BYTE* sp1, int spitch1, const BYTE* sp2, int spitch2, const BYTE* pp, int ppitch, const BYTE* np, int npitch, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dp[x] = RepairPixel_c(sp1[x], sp2[x], pp[x], np[x]);
		}
		dp += dpitch;
		sp1 += spitch1;
		sp2 += spitch2;
		pp += ppitch;
		np += npitch;
	}
}

// SSE4.1 simulation for SSE2
// false: a, true: b
RG_FORCEINLINE __m128i _MM_BLENDV_EPI8(__m128i const& a, __m128i const& b, __m128i const& selector) {
  return _mm_or_si128(_mm_and_si128(selector, b), _mm_andnot_si128(selector, a));
}

RG_FORCEINLINE void BRepairPixel(uint8_t *dest, const uint8_t *src1, const uint8_t *src2, const uint8_t *previous, const uint8_t *next)
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
  auto equ = _mm_or_si128(equ1, equ2);// _mm_max_epu8(equ1, equ2); // pracically or. mask set if min_np == reg5 or max_np == reg3 FIXME

	reg5 = _mm_min_epu8(reg5, src_1);
	reg5 = _mm_max_epu8(reg5, reg3);

  // real SSE4.1 _mm_blendv_epi8: 2830 vs 2960 FPS, worth using it
  auto result = _MM_BLENDV_EPI8(reg5, src_2, equ); // keep src2 where equal

	_mm_storeu_si128(reinterpret_cast<__m128i*>(dest), result);
}

static void btemporal_repair_mode4_sse2(BYTE *dp, int dpitch, const BYTE *sp1, int spitch1, const BYTE *sp2, int spitch2, const BYTE *pp, int ppitch, const BYTE *np, int npitch, int width, int height)
{
  assert(width >= 16);

  const int wmod16 = width / 16 * 16;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < wmod16; x += 16)
    {
      BRepairPixel(dp + x, sp1 + x, sp2 + x, pp + x, np + x);
    }
    // remainder, surely unaligned, overlaps but still simd
    const int last16 = width - 16;
    BRepairPixel(dp + last16, sp1 + last16, sp2 + last16, pp + last16, np + last16);
    dp += dpitch;
    sp1 += spitch1;
    sp2 += spitch2;
    pp += ppitch;
    np += npitch;
  }
}

RG_FORCEINLINE int subs_int(int x, int y) {
  return x < y ? 0 : x - y;
}

RG_FORCEINLINE int adds_uint8(int x, int y) {
  int sum = x + y;
  return sum > 255 ? 255 : sum;
}

RG_FORCEINLINE int BRepairPixel_c(int src_1, int src_2, int src_prev, int src_next)
{
  auto max_np = std::max(src_next, src_prev);
  auto min_np = std::min(src_next, src_prev);

  auto diff_src2_minnp = subs_int(src_2, min_np);
  auto reg5 = adds_uint8(adds_uint8(diff_src2_minnp, diff_src2_minnp), min_np); // diff_src2_minnp * 2 - min_np

  auto diff_maxnp_src2 = subs_int(max_np, src_2);
  auto reg3 = subs_int(max_np, adds_uint8(diff_maxnp_src2, diff_maxnp_src2)); // max_np - 2 * diff_maxnp_src2

  reg5 = std::min(reg5, max_np);
  reg3 = std::max(reg3, min_np);

  auto equ = min_np == reg5 || max_np == reg3;

  reg5 = std::min(reg5, src_1);
  reg5 = std::max(reg5, reg3);

  return equ ? src_2 : reg5;
}

static void btemporal_repair_mode4_c(BYTE* dp, int dpitch, const BYTE* sp1, int spitch1, const BYTE* sp2, int spitch2, const BYTE* pp, int ppitch, const BYTE* np, int npitch, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dp[x] = BRepairPixel_c(sp1[x], sp2[x], pp[x], np[x]);
		}
		dp += dpitch;
		sp1 += spitch1;
		sp2 += spitch2;
		pp += ppitch;
		np += npitch;
	}
}

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
  void(*trepair)(BYTE *dp, int dpitch, const BYTE *sp1, int spitch1, const BYTE *sp2, int spitch2, const BYTE *pp, int ppitch, const BYTE *np, int npitch, int width, int height);
  int last_frame;
  PClip orig;
  bool grey;
  int opt;

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

      trepair(
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
    orig(oclip), opt(opt), grey(grey)
  {
    if (!planar && !vi.IsPlanar())
      env->ThrowError("TemporalRepair: only planar color spaces are supported");

    CompareVideoInfo(vi, orig->GetVideoInfo(), "TemporalRepair", env);

    assert(mode == 0 || mode == 4);
    if (opt == 0) // 0: C
      trepair = mode == 4 ? btemporal_repair_mode4_c : temporal_repair_mode0_c;
    else // SSE2 new simd
      trepair = mode == 4 ? btemporal_repair_mode4_sse2 : temporal_repair_mode0_sse2;
    last_frame = vi.num_frames >= 2 ? vi.num_frames - 2 : 0;
  }
};

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
  upper = subs_int(max_np, src_curr);
  lower = subs_int(src_curr, min_np);
}

RG_FORCEINLINE void SmoothTRepair(uint8_t *dest, __m128i &lower, __m128i &upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i*>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i*>(next));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  auto tmp_u = _mm_adds_epu8(upper, src_curr);
  auto tmp_l = _mm_subs_epu8(src_curr, lower);

  auto tmp_max = _mm_max_epu8(_mm_max_epu8(tmp_u, src_prev), src_next);
  auto tmp_min = _mm_min_epu8(_mm_min_epu8(tmp_l, src_prev), src_next);
  
  auto result = _mm_max_epu8(_mm_min_epu8(tmp_max, src_dest), tmp_min);
  
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dest), result);
}

RG_FORCEINLINE void SmoothTRepair_c(BYTE &dest, int lower, int upper, const int src_prev, const int src_curr, const int src_next)
{
  auto src_dest = dest;

  auto tmp_u = adds_uint8(upper, src_curr);
  auto tmp_l = subs_int(src_curr, lower);

  auto tmp_max = std::max(std::max(tmp_u, src_prev), src_next);
  auto tmp_min = std::min(std::min(tmp_l, src_prev), src_next);

  auto result = std::max(std::min(tmp_max, (int)src_dest), tmp_min);

  dest = result;
}

void smooth_temporal_repair1_sse2(BYTE *dp, const BYTE *previous, const BYTE *sp, const BYTE *next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1;
  // remainder: corrects pointers back to process exactly the 16 last pixels 
  const int hblocks = (width - 1) / 16;
  const int remainder = ((width - 1) & 15) - 15;

  for(int i = 0; i < 4; i++)
    pitches[i] -= (hblocks * 16 + remainder);

  for (int y = 0; y < height; y++)
  {
    __m128i lowermax = _mm_undefined_si128();
    __m128i uppermax = _mm_undefined_si128();
    __m128i lower = _mm_undefined_si128();
    __m128i upper = _mm_undefined_si128();
    for (int x = 0; x < hblocks; x++)
    {
      get_lu(lowermax, uppermax, previous, sp, next);
      get_lu(lower, upper, previous + 1, sp + 1, next + 1);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2, sp + 2, next + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch, sp + 2 * ofpitch, next + 2 * nfpitch);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch + 1, sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch + 2, sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + pfpitch, sp + ofpitch, next + nfpitch);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + pfpitch + 2, sp + ofpitch + 2, next + nfpitch + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      SmoothTRepair(dp, lowermax, uppermax, previous + pfpitch + 1, sp + ofpitch + 1, next + nfpitch + 1);
      dp += 16;
      previous += 16;
      sp += 16;
      next += 16;
    }

    // the last pixels. Do simd to the rightmost 16 pixels, overlap is possible but no problem
    sp += remainder;
    previous += remainder;
    next += remainder;
    dp += remainder;

    get_lu(lowermax, uppermax, previous, sp, next);
    get_lu(lower, upper, previous + 1, sp + 1, next + 1);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2, sp + 2, next + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch, sp + 2 * ofpitch, next + 2 * nfpitch);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch + 1, sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch + 2, sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + pfpitch, sp + ofpitch, next + nfpitch);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + pfpitch + 2, sp + ofpitch + 2, next + nfpitch + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    SmoothTRepair(dp, lowermax, uppermax, previous + pfpitch + 1, sp + ofpitch + 1, next + nfpitch + 1);

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

void smooth_temporal_repair1_c(BYTE* dp, const BYTE* previous, const BYTE* sp, const BYTE* next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1;

  for (int y = 0; y < height; y++)
  {

    for (int x = 0; x < width; x++)
    {
      int lowermax, uppermax;
      int lower, upper;    
      get_lu_c(lowermax, uppermax, previous[x], sp[x], next[x]);
      get_lu_c(lower, upper, previous[x + 1], sp[x + 1], next[x + 1]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2], sp[x + 2], next[x + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch], sp[x + 2 * ofpitch], next[x + 2 * nfpitch]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch + 1], sp[x + 2 * ofpitch + 1], next[x + 2 * nfpitch + 1]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch + 2], sp[x + 2 * ofpitch + 2], next[x + 2 * nfpitch + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + pfpitch], sp[x + ofpitch], next[x + nfpitch]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + pfpitch + 2], sp[x + ofpitch + 2], next[x + nfpitch + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      SmoothTRepair_c(dp[x], lowermax, uppermax, previous[x + pfpitch + 1], sp[x + ofpitch + 1], next[x + nfpitch + 1]);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

RG_FORCEINLINE void SmoothTRepair2(uint8_t *dest, __m128i lower, __m128i upper, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i lower1 = _mm_undefined_si128();
  __m128i upper1 = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dest));

  get_lu(lower1, upper1, previous, current, next);
  auto uppermax = _mm_max_epu8(upper, upper1);
  auto lowermax = _mm_max_epu8(lower, lower1);
  auto upperlowermax = _mm_max_epu8(uppermax, lowermax);

  auto tmp1 = _mm_adds_epu8(src_curr, upperlowermax);
  auto tmp2 = _mm_subs_epu8(src_curr, upperlowermax);

  auto result = _mm_max_epu8(_mm_min_epu8(tmp1, src_dest), tmp2);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dest), result);
}

RG_FORCEINLINE void SmoothTRepair2_c(BYTE &dest, int lower, int upper, int src_prev, int src_curr, int src_next)
{
  int lower1;
  int upper1;

  auto src_dest = dest;

  get_lu_c(lower1, upper1, src_prev, src_curr, src_next);
  auto uppermax = std::max(upper, upper1);
  auto lowermax = std::max(lower, lower1);
  auto upperlowermax = std::max(uppermax, lowermax);

  auto tmp1 = adds_uint8(src_curr, upperlowermax);
  auto tmp2 = subs_int(src_curr, upperlowermax);

  auto result = std::max(std::min(tmp1, (int)src_dest), tmp2);
  dest = result;
}

void	smooth_temporal_repair2_sse2(BYTE *dp, const BYTE *previous, const BYTE *_sp, const BYTE *next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1; //  __asm	lea			edi, [edi + pitch + 1]
  // remainder: corrects pointers back to process exactly the 16 last pixels 
  const int hblocks = (width - 1) / 16;
  const int remainder = ((width - 1) & 15) - 15;

  for (int i = 0; i < 4; i++)
    pitches[i] -= (hblocks * 16 + remainder);

  for (int y = 0; y < height; y++)
  {

    __m128i lowermax = _mm_undefined_si128();
    __m128i uppermax = _mm_undefined_si128();
    __m128i lower = _mm_undefined_si128();
    __m128i upper = _mm_undefined_si128();
    for (int x = 0; x < hblocks; x++)
    {
      get_lu(lowermax, uppermax, previous, _sp, next);
      get_lu(lower, upper, previous + 1, _sp + 1, next + 1);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2, _sp + 2, next + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch, _sp + 2 * ofpitch, next + 2 * nfpitch);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch + 1, _sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + 2 * pfpitch + 2, _sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + pfpitch, _sp + ofpitch, next + nfpitch);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      get_lu(lower, upper, previous + pfpitch + 2, _sp + ofpitch + 2, next + nfpitch + 2);
      uppermax = _mm_max_epu8(uppermax, upper);
      lowermax = _mm_max_epu8(lowermax, lower);
      SmoothTRepair2(dp, lowermax, uppermax, previous + pfpitch + 1, _sp + ofpitch + 1, next + nfpitch + 1);

      _sp += 16;
      previous += 16;
      next += 16;
      dp += 16;
    }

    // the last pixels. Do simd to the rightmost 16 pixels, overlap is possible but no problem
    _sp += remainder;
    previous += remainder;
    next += remainder;
    dp += remainder;

    get_lu(lowermax, uppermax, previous, _sp, next);
    get_lu(lower, upper, previous + 1, _sp + 1, next + 1);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2, _sp + 2, next + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch, _sp + 2 * ofpitch, next + 2 * nfpitch);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch + 1, _sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + 2 * pfpitch + 2, _sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + pfpitch, _sp + ofpitch, next + nfpitch);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    get_lu(lower, upper, previous + pfpitch + 2, _sp + ofpitch + 2, next + nfpitch + 2);
    uppermax = _mm_max_epu8(uppermax, upper);
    lowermax = _mm_max_epu8(lowermax, lower);
    SmoothTRepair2(dp, lowermax, uppermax, previous + pfpitch + 1, _sp + ofpitch + 1, next + nfpitch + 1);

    dp += pitches[0];
    previous += pitches[1];
    _sp += pitches[2];
    next += pitches[3];
  }
}


void	smooth_temporal_repair2_c(BYTE* dp, const BYTE* previous, const BYTE* sp, const BYTE* next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int lowermax, uppermax;
      int lower, upper;     
      get_lu_c(lowermax, uppermax, previous[x], sp[x], next[x]);
      get_lu_c(lower, upper, previous[x + 1], sp[x + 1], next[x + 1]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2], sp[x + 2], next[x + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch], sp[x + 2 * ofpitch], next[x + 2 * nfpitch]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch + 1], sp[x + 2 * ofpitch + 1], next[x + 2 * nfpitch + 1]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + 2 * pfpitch + 2], sp[x + 2 * ofpitch + 2], next[x + 2 * nfpitch + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + pfpitch], sp[x + ofpitch], next[x + nfpitch]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      get_lu_c(lower, upper, previous[x + pfpitch + 2], sp[x + ofpitch + 2], next[x + nfpitch + 2]);
      uppermax = std::max(uppermax, upper);
      lowermax = std::max(lowermax, lower);
      SmoothTRepair2_c(dp[x], lowermax, uppermax, previous[x + pfpitch + 1], sp[x + ofpitch + 1], next[x + nfpitch + 1]);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

RG_FORCEINLINE void get2diff(__m128i &pdiff, __m128i &ndiff, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i *>(current));
  auto src_prev = _mm_loadu_si128(reinterpret_cast<const __m128i *>(previous));
  auto src_next = _mm_loadu_si128(reinterpret_cast<const __m128i *>(next));

  auto diff_cp = _mm_subs_epu8(src_curr, src_prev);
  auto diff_pc = _mm_subs_epu8(src_prev, src_curr);
  pdiff = _mm_or_si128(diff_pc, diff_cp); // abs(c-p)

  auto diff_cn = _mm_subs_epu8(src_curr, src_next);
  auto diff_nc = _mm_subs_epu8(src_next, src_curr);
  ndiff = _mm_or_si128(diff_nc, diff_cn); // abs(c-n)
}

RG_FORCEINLINE void get2diff_c(int& pdiff, int& ndiff, int src_prev, int src_curr, int src_next)
{
  pdiff = abs(src_curr - src_prev); 
  ndiff = abs(src_curr - src_next);
}

RG_FORCEINLINE void SmoothTRepair3(uint8_t *dest, __m128i &pmax, __m128i &nmax, const uint8_t *previous, const uint8_t *current, const uint8_t *next)
{
  __m128i pdiff = _mm_undefined_si128();
  __m128i ndiff = _mm_undefined_si128();

  auto src_curr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current));

  get2diff(pdiff, ndiff, previous, current, next);
  pmax = _mm_max_epu8(pmax, pdiff);
  nmax = _mm_max_epu8(nmax, ndiff);

  pmax = _mm_min_epu8(pmax, nmax);
  auto src_dest = _mm_loadu_si128(reinterpret_cast<const __m128i *>(dest));
  auto tmp_upper = _mm_adds_epu8(src_curr, pmax);
  auto tmp_lower = _mm_subs_epu8(src_curr, pmax);
  auto result = _mm_max_epu8(_mm_min_epu8(tmp_upper, src_dest), tmp_lower);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(dest), result);
}

RG_FORCEINLINE void SmoothTRepair3_c(BYTE &dest, int& pmax, int& nmax, int src_prev, int src_curr, int src_next)
{
  int pdiff;
  int ndiff;

  get2diff_c(pdiff, ndiff, src_prev, src_curr, src_next);
  pmax = std::max(pmax, pdiff);
  nmax = std::max(nmax, ndiff);

  pmax = std::min(pmax, nmax);
  auto src_dest = dest;
  auto tmp_upper = adds_uint8(src_curr, pmax);
  auto tmp_lower = subs_int(src_curr, pmax);
  auto result = std::max(std::min(tmp_upper, (int)src_dest), tmp_lower);
  dest = result;
}


void smooth_temporal_repair3_sse2(BYTE *dp, const BYTE *previous, const BYTE *_sp, const BYTE *next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1; //  __asm	lea			edi, [edi + pitch + 1]
  // remainder: corrects pointers back to process exactly the 16 last pixels 
  const int hblocks = (width - 1) / 16;
  const int remainder = ((width - 1) & 15) - 15;

  for (int i = 0; i < 4; i++)
    pitches[i] -= (hblocks * 16 + remainder);

  for (int y = 0; y < height; y++)
  {
    __m128i pdiffmax = _mm_undefined_si128();
    __m128i ndiffmax = _mm_undefined_si128();
    __m128i pdiff = _mm_undefined_si128();
    __m128i ndiff = _mm_undefined_si128();
    for (int x = 0; x < hblocks; x++)
    {
      get2diff(pdiffmax, ndiffmax, previous, _sp, next);
      get2diff(pdiff, ndiff, previous + 1, _sp + 1, next + 1);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + 2, _sp + 2, next + 2);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + 2 * pfpitch, _sp + 2 * ofpitch, next + 2 * nfpitch);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + 2 * pfpitch + 1, _sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + 2 * pfpitch + 2, _sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff); // __asm	pmaxub		pdiffmax, pdiff // BUG Fixed by PF 2019, this line was missing
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + pfpitch, _sp + ofpitch, next + nfpitch);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      get2diff(pdiff, ndiff, previous + pfpitch + 2, _sp + ofpitch + 2, next + nfpitch + 2);
      pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
      ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
      SmoothTRepair3(dp, pdiffmax, ndiffmax, previous + pfpitch + 1, _sp + ofpitch + 1, next + nfpitch + 1);
      _sp += 16;
      previous += 16;
      next += 16;
      dp += 16;
    }
    // the last pixels. Do simd to the rightmost 16 pixels, overlap is possible but no problem
    _sp += remainder;
    previous += remainder;
    next += remainder;
    dp += remainder;

    get2diff(pdiffmax, ndiffmax, previous, _sp, next);
    get2diff(pdiff, ndiff, previous + 1, _sp + 1, next + 1);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + 2, _sp + 2, next + 2);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + 2 * pfpitch, _sp + 2 * ofpitch, next + 2 * nfpitch);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + 2 * pfpitch + 1, _sp + 2 * ofpitch + 1, next + 2 * nfpitch + 1);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + 2 * pfpitch + 2, _sp + 2 * ofpitch + 2, next + 2 * nfpitch + 2);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff); // __asm	pmaxub		pdiffmax, pdiff // BUG Fixed by PF 2019, this line was missing
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + pfpitch, _sp + ofpitch, next + nfpitch);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    get2diff(pdiff, ndiff, previous + pfpitch + 2, _sp + ofpitch + 2, next + nfpitch + 2);
    pdiffmax = _mm_max_epu8(pdiffmax, pdiff);
    ndiffmax = _mm_max_epu8(ndiffmax, ndiff);
    SmoothTRepair3(dp, pdiffmax, ndiffmax, previous + pfpitch + 1, _sp + ofpitch + 1, next + nfpitch + 1);


    dp += pitches[0];
    previous += pitches[1];
    _sp += pitches[2];
    next += pitches[3];
  }
}

void smooth_temporal_repair3_c(BYTE* dp, const BYTE* previous, const BYTE* sp, const BYTE* next, intptr_t* pitches, int width, int height)
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

  dp = dp + dppitch + 1;

  for (int y = 0; y < height; y++)
  {
    int pdiffmax;
    int ndiffmax;
    int pdiff;
    int ndiff;
    for (int x = 0; x < width; x++)
    {
      get2diff_c(pdiffmax, ndiffmax, previous[x], sp[x], next[x]);
      get2diff_c(pdiff, ndiff, previous[x + 1], sp[x + 1], next[x + 1]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + 2], sp[x + 2], next[x + 2]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + 2 * pfpitch], sp[x + 2 * ofpitch], next[x + 2 * nfpitch]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + 2 * pfpitch + 1], sp[x + 2 * ofpitch + 1], next[x + 2 * nfpitch + 1]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + 2 * pfpitch + 2], sp[x + 2 * ofpitch + 2], next[x + 2 * nfpitch + 2]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + pfpitch], sp[x + ofpitch], next[x + nfpitch]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      get2diff_c(pdiff, ndiff, previous[x + pfpitch + 2], sp[x + ofpitch + 2], next[x + nfpitch + 2]);
      pdiffmax = std::max(pdiffmax, pdiff);
      ndiffmax = std::max(ndiffmax, ndiff);
      SmoothTRepair3_c(dp[x], pdiffmax, ndiffmax, previous[x + pfpitch + 1], sp[x + ofpitch + 1], next[x + nfpitch + 1]);
    }

    dp += pitches[0];
    previous += pitches[1];
    sp += pitches[2];
    next += pitches[3];
  }
}

class SmoothTemporalRepair : public GenericVideoFilter
{
  PClip oclip;

  void(*st_repair)(BYTE *dp, const BYTE *previous, const BYTE *_sp, const BYTE *next, intptr_t* pitches, int width, int height);

  int last_frame;
  bool grey;
  int opt;
  
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

      int sfpitch = sf->GetPitch(plane);

      BYTE* dp = df->GetWritePtr(plane);
      int dppitch = df->GetPitch(plane);
      int pfpitch = pf->GetPitch(plane);
      int ofpitch = of->GetPitch(plane);
      int nfpitch = nf->GetPitch(plane);

      // copy the plane from sp to dp
      copy_plane(df, sf, plane, env);

      intptr_t pitches[4] = { dppitch, pfpitch, ofpitch, nfpitch };

      // Spatial: Edge rows/columns are unhandled: -2 pixels less
      st_repair(dp, pf->GetReadPtr(plane), of->GetReadPtr(plane), nf->GetReadPtr(plane),
        pitches, // pitch array
        df->GetRowSize(plane) / vi.ComponentSize() - 2,
        df->GetHeight(plane) - 2);
    }

    if (vi.NumComponents() == 4)
      copy_plane(df, sf, PLANAR_A, env);

    return df;
  }
public:
  SmoothTemporalRepair(PClip clip, PClip _oclip, int mode, bool grey, bool planar, int opt, IScriptEnvironment* env) : 
    GenericVideoFilter(clip), 
    oclip(_oclip),
    grey(grey),
    opt(opt)
  {
    if (!planar && !vi.IsPlanar())
      env->ThrowError("TemporalRepair: only planar color spaces are supported");

    CompareVideoInfo(vi, _oclip->GetVideoInfo(), "TemporalRepair", env);

    if (vi.IsY())
      grey = true;

    switch (mode)
    {
    case 1:
      if (opt == 0)
        st_repair = smooth_temporal_repair1_c;
      else
        st_repair = smooth_temporal_repair1_sse2;
      break;
    case 2:
      if (opt == 0)
        st_repair = smooth_temporal_repair2_c;
      else
        st_repair = smooth_temporal_repair2_sse2;
      break;
    default:
      if (opt == 0)
        st_repair = smooth_temporal_repair3_c;
      else
        st_repair = smooth_temporal_repair3_sse2;
      break;
    }

    last_frame = vi.num_frames >= 2 ? vi.num_frames - 2 : 0;

    /*FIXME: check min dimensions
      env->ThrowError("TemporalRepair: the width or height of the clip is too small");*/
  }
};

#define MAXTMODE	4	

bool spatial[MAXTMODE + 1] = { false, true, true, true, false };

AVSValue __cdecl Create_TemporalRepair(AVSValue args, void* user_data, IScriptEnvironment* env)
{
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

  if (clip->GetVideoInfo().BitsPerComponent() > 8)
    env->ThrowError("TemporalRepair: only 8 bit colorspaces are supported");

  return spatial[mode] ? (AVSValue) new SmoothTemporalRepair(clip, oclip, mode, grey, planar, opt, env)
    : (AVSValue) new TemporalRepair(clip, oclip, mode, grey, planar, opt, env);

};

