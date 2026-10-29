// VerticalCleaner core: C kernels, function tables, dispatch

#include "vertical_cleaner_core.h"
#include "common.h"

template<typename pixel_t>
static void vcleaner_median_c(uint8_t* pDst8, const uint8_t *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
    rg_copy_rows(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    srcPitch /= sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;

    const int width = rowsize / sizeof(pixel_t);

    for (int y = 1; y < height-1; ++y) {
        for (int x = 0; x < width; x+=1) {
            pixel_t up = pSrc[x-srcPitch];
            pixel_t center = pSrc[x];
            pixel_t down = pSrc[x+srcPitch];
            pDst[x] = std::min(std::max(std::min(up, down), center), std::max(up, down));
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    rg_copy_rows((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

static RG_FORCEINLINE uint8_t satb(int value) {
    return clip(value, 0, 255);
}

static RG_FORCEINLINE uint16_t satb_16(int value, int max_pixel_value) {
  return clip_16(value, 0, max_pixel_value);
}

static RG_FORCEINLINE float satb_32(float value) {
#if 0
  // no clamp for float
  return clip_32(value, 0.0f, 1.0f);
#else
  return value;
#endif
}

static void vcleaner_relaxed_median_c(uint8_t* pDst, const uint8_t *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);

  pSrc += srcPitch*2;
  pDst += dstPitch*2;

  const int width = rowsize;

  for (int y = 2; y < height-2; ++y) {
    for (int x = 0; x < width; x+=1) {
      uint8_t p2 = pSrc[x-srcPitch*2];
      uint8_t p1 = pSrc[x-srcPitch];
      uint8_t c = pSrc[x];
      uint8_t n1 = pSrc[x+srcPitch];
      uint8_t n2 = pSrc[x+srcPitch*2];

      uint8_t upper = std::max(std::max(std::min(satb(satb(p1-p2) + p1), satb(satb(n1-n2) + n1)), p1), n1);
      uint8_t lower = std::min(std::min(p1, n1), std::max(satb(p1 - satb(p2-p1)), satb(n1 - satb(n2-n1))));

      pDst[x] = clip(c, lower, upper);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);
}

static void vcleaner_relaxed_median_c_16(uint8_t* pDst8, const uint8_t *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
    rg_copy_rows(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

    uint16_t *pDst = reinterpret_cast<uint16_t *>(pDst8);
    const uint16_t *pSrc = reinterpret_cast<const uint16_t *>(pSrc8);

    dstPitch /= sizeof(uint16_t);
    srcPitch /= sizeof(uint16_t);

    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    const int width = rowsize / sizeof(uint16_t);

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    for (int y = 2; y < height-2; ++y) {
        for (int x = 0; x < width; x+=1) {
          uint16_t p2 = pSrc[x-srcPitch*2];
          uint16_t p1 = pSrc[x-srcPitch];
          uint16_t c  = pSrc[x];
          uint16_t n1 = pSrc[x+srcPitch];
          uint16_t n2 = pSrc[x+srcPitch*2];

          uint16_t upper = std::max(std::max(std::min(satb_16(satb_16(p1-p2, max_pixel_value) + p1, max_pixel_value), satb_16(satb_16(n1-n2, max_pixel_value) + n1, max_pixel_value)), p1), n1);
          uint16_t lower = std::min(std::min(p1, n1), std::max(satb_16(p1 - satb_16(p2-p1, max_pixel_value), max_pixel_value), satb_16(n1 - satb_16(n2-n1, max_pixel_value), max_pixel_value)));

          pDst[x] = clip_16(c, lower, upper);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    rg_copy_rows((uint8_t *)pDst, dstPitch*sizeof(uint16_t), (uint8_t *)pSrc, srcPitch*sizeof(uint16_t), rowsize, 2);
}

template<bool chroma>
static void vcleaner_relaxed_median_c_32(uint8_t* pDst8, const uint8_t *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
  rg_copy_rows(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

  float *pDst = reinterpret_cast<float *>(pDst8);
  const float *pSrc = reinterpret_cast<const float *>(pSrc8);

  dstPitch /= sizeof(float);
  srcPitch /= sizeof(float);

  pSrc += srcPitch*2;
  pDst += dstPitch*2;

  const int width = rowsize / sizeof(float);

  for (int y = 2; y < height-2; ++y) {
    for (int x = 0; x < width; x+=1) {
      float p2 = pSrc[x-srcPitch*2];
      float p1 = pSrc[x-srcPitch];
      float c  = pSrc[x];
      float n1 = pSrc[x+srcPitch];
      float n2 = pSrc[x+srcPitch*2];

      float upper = std::max(std::max(std::min(
        adds_32_c<chroma>(subs_32_c_for_diff(p1, p2), p1),
        adds_32_c<chroma>(subs_32_c_for_diff(n1, n2), n1)), p1), n1);
      float lower = std::min(std::min(p1, n1), std::max(
        subs_32_c<chroma>(p1, subs_32_c_for_diff(p2, p1)), 
        subs_32_c<chroma>(n1, subs_32_c_for_diff(n2, n1))));

      pDst[x] = clip_32(c, lower, upper);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  rg_copy_rows((uint8_t *)pDst, dstPitch*sizeof(float), (uint8_t *)pSrc, srcPitch*sizeof(float), rowsize, 2);
}


void vcleaner_copy_plane(uint8_t* pDst, const uint8_t *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {
    rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

void vcleaner_do_nothing(uint8_t* pDst, const uint8_t *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int bits_per_pixel) {

}

static VCleanerProcessor* c_functions[] = {
    vcleaner_do_nothing,
    vcleaner_copy_plane,
    vcleaner_median_c<uint8_t>,
    vcleaner_relaxed_median_c
};

// 10-16 bit (the max pixel value comes from bits_per_pixel)
static VCleanerProcessor* c_functions_16[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16
};

static VCleanerProcessor* c_functions_32[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<float>,
  vcleaner_relaxed_median_c_32<false>
};

static VCleanerProcessor* c_functions_32_chroma[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<float>,
  vcleaner_relaxed_median_c_32<true>
};

#ifdef INTEL_INTRINSICS
// vertical_cleaner_sse.cpp
extern VCleanerProcessor* vcleaner_sse2_functions[];
extern VCleanerProcessor* vcleaner_sse4_functions_uint16[];
extern VCleanerProcessor* vcleaner_sse2_functions_32[];
extern VCleanerProcessor* vcleaner_sse2_functions_32_chroma[];
#endif

static void dispatch_median(int mode, uint8_t* pDst, const uint8_t *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int pixelsize, int bits_per_pixel, bool chroma, int cpu_flags) {
  // a plane without an inner row (mode 1: 1 border row, mode 2: 2 border rows at the top and the bottom) is copied
  if ((mode == 1 && height < 3) || (mode == 2 && height < 5))
    mode = 0;
  if (pixelsize == 1) {
#ifdef INTEL_INTRINSICS
    if ((cpu_flags & RG_CPU_SSE2) && rowsize >= 16) {
      vcleaner_sse2_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
      return;
    }
#endif
    c_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
  }
  else if (pixelsize == 2) {
#ifdef INTEL_INTRINSICS
    if ((cpu_flags & RG_CPU_SSE41) && rowsize >= 16) {
      vcleaner_sse4_functions_uint16[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
      return;
    }
#endif
    c_functions_16[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
  }
  else { // if (pixelsize == 4
#ifdef INTEL_INTRINSICS
    if ((cpu_flags & RG_CPU_SSE2) && rowsize >= 16) {
      if (chroma)
        vcleaner_sse2_functions_32_chroma[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
      else
        vcleaner_sse2_functions_32[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
      return;
    }
#endif
    if (chroma)
      c_functions_32_chroma[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
    else
      c_functions_32[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, bits_per_pixel);
  }
}

const char* VerticalCleanerCore::init(const VerticalCleanerParams& params, const RgFormat& format, int cpu_flags) {
  int mode = params.mode;
  int modeU = params.modeU;
  int modeV = params.modeV;

  // modeU/modeV below -1: same as the previous plane (UNDEFINED_MODE, the default)
  if (mode <= UNDEFINED_MODE || mode > 2 || modeU > 2 || modeV > 2)
    return "Sorry, this mode does not exist";

  if (format.family == RgColorFamily::RGB && (modeU > UNDEFINED_MODE || modeV > UNDEFINED_MODE))
    return "VerticalCleaner: cannot specify U or V mode for planar RGB!";

  if (modeU <= UNDEFINED_MODE)
    modeU = mode;
  if (modeV <= UNDEFINED_MODE)
    modeV = modeU;

  mode_[0] = mode;
  mode_[1] = modeU;
  mode_[2] = modeV;
  format_ = format;
  cpu_flags_ = cpu_flags;
  return nullptr;
}

void VerticalCleanerCore::process_plane(int plane_index, const RgPlane& plane) const {
  const bool chroma = format_.family == RgColorFamily::YUV && plane_index > 0;
  dispatch_median(mode_[plane_index], plane.dst, plane.src, plane.dst_pitch, plane.src_pitch, plane.rowsize, plane.height,
    format_.pixelsize, format_.bits_per_pixel, chroma, cpu_flags_);
}
