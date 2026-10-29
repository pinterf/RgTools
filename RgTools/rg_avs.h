#ifndef __RG_AVS_H__
#define __RG_AVS_H__

// AviSynth host helpers: avisynth.h, and the mapping of the AviSynth CPU flags and video info to the core types.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#pragma warning(disable: 4512 4244 4100)
#include "avisynth.h"
#pragma warning(default: 4512 4244 4100)

#include "rg_core.h"

// env->GetCPUFlags() (SetMaxCPU applied) -> RgCpu flags
static inline int rg_cpu_flags_from_avs(int64_t avs_flags) {
  int flags = 0;
  if (avs_flags & CPUF_SSE2) flags |= RG_CPU_SSE2;
  if (avs_flags & CPUF_SSE4_1) flags |= RG_CPU_SSE41;
  if (avs_flags & CPUF_AVX2) flags |= RG_CPU_AVX2;
  return flags;
}

// planar formats; YA (V13 interface, false on older hosts): Y with alpha
static inline bool rg_is_gray(const VideoInfo& vi) { return vi.IsY() || vi.IsYA(); }
static inline bool rg_has_alpha(const VideoInfo& vi) { return vi.IsYUVA() || vi.IsPlanarRGBA() || vi.IsYA(); }
// planes processed by the cores (alpha: copied by the host)
static inline int rg_num_planes(const VideoInfo& vi) { return rg_is_gray(vi) ? 1 : 3; }

// planar formats only (the filters reject the others)
static inline RgFormat rg_format_from_vi(const VideoInfo& vi) {
  RgFormat format;
  format.family = rg_is_gray(vi) ? RgColorFamily::Gray : (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? RgColorFamily::RGB : RgColorFamily::YUV;
  format.bits_per_pixel = vi.BitsPerComponent();
  format.pixelsize = vi.ComponentSize();
  format.width = vi.width;
  format.height = vi.height;
  const bool yuv = format.family == RgColorFamily::YUV;
  format.ssw = yuv ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
  format.ssh = yuv ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;
  format.has_alpha = rg_has_alpha(vi);
  return format;
}

#endif
