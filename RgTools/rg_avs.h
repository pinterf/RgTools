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

// planar formats only (the filters reject the others)
static inline RgFormat rg_format_from_vi(const VideoInfo& vi) {
  RgFormat format;
  format.family = vi.IsY() ? RgColorFamily::Gray : (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? RgColorFamily::RGB : RgColorFamily::YUV;
  format.bits_per_pixel = vi.BitsPerComponent();
  format.pixelsize = vi.ComponentSize();
  return format;
}

#endif
