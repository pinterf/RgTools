#ifndef __RG_CORE_H__
#define __RG_CORE_H__

// RgTools core: types shared by the filter cores and the host front ends (AviSynth, later VapourSynth).
// No host API here (avisynth.h, VapourSynth4.h): a core gets the parameters, the format, the CPU flags
// and plane pointers only.

#include <cstdint>
#include <cstring>

// CPU features the cores use. The host maps its own flags (AviSynth: env->GetCPUFlags(), SetMaxCPU honoured).
enum RgCpu : int {
  RG_CPU_SSE2  = 1,
  RG_CPU_SSE41 = 2,
  RG_CPU_AVX2  = 4, // the AVX2 code is compiled with FMA as well
};

enum class RgColorFamily { Gray, YUV, RGB };

// Format of a clip, from the video info of the host
struct RgFormat {
  RgColorFamily family;
  int bits_per_pixel; // 8, 10..16, 32 (float)
  int pixelsize;      // bytes per sample: 1, 2, 4
};

// One plane of a frame
struct RgPlane {
  const uint8_t* src;
  int src_pitch;      // bytes
  uint8_t* dst;
  int dst_pitch;      // bytes
  int rowsize;        // bytes
  int height;
};

// Row copy (border rows, whole planes), rowsize in bytes
static inline void rg_copy_rows(uint8_t* dst, int dst_pitch, const uint8_t* src, int src_pitch, int rowsize, int rows) {
  for (int y = 0; y < rows; ++y) {
    memcpy(dst, src, rowsize);
    dst += dst_pitch;
    src += src_pitch;
  }
}

#endif
