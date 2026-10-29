#include "vertical_cleaner.h"

template<typename pixel_t>
static void vcleaner_median_c(Byte* pDst8, const Byte *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

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

    env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

static RG_FORCEINLINE Byte satb(int value) {
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

static void vcleaner_relaxed_median_c(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);

  pSrc += srcPitch*2;
  pDst += dstPitch*2;

  const int width = rowsize;

  for (int y = 2; y < height-2; ++y) {
    for (int x = 0; x < width; x+=1) {
      BYTE p2 = pSrc[x-srcPitch*2];
      BYTE p1 = pSrc[x-srcPitch];
      BYTE c = pSrc[x];
      BYTE n1 = pSrc[x+srcPitch];
      BYTE n2 = pSrc[x+srcPitch*2];

      Byte upper = std::max(std::max(std::min(satb(satb(p1-p2) + p1), satb(satb(n1-n2) + n1)), p1), n1);
      Byte lower = std::min(std::min(p1, n1), std::max(satb(p1 - satb(p2-p1)), satb(n1 - satb(n2-n1))));

      pDst[x] = clip(c, lower, upper);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2);
}

template<int bits_per_pixel>
static void vcleaner_relaxed_median_c_16(Byte* pDst8, const Byte *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

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

    env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(uint16_t), (uint8_t *)pSrc, srcPitch*sizeof(uint16_t), rowsize, 2);
}

template<bool chroma>
static void vcleaner_relaxed_median_c_32(Byte* pDst8, const Byte *pSrc8, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
  env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 2);

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

  env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(float), (uint8_t *)pSrc, srcPitch*sizeof(float), rowsize, 2);
}


void vcleaner_copy_plane(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

void vcleaner_do_nothing(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {

}

static VCleanerProcessor* c_functions[] = {
    vcleaner_do_nothing,
    vcleaner_copy_plane,
    vcleaner_median_c<uint8_t>,
    vcleaner_relaxed_median_c
};

// distinct templates for 10-12-14-16 bit for const max_pixel_value
static VCleanerProcessor* c_functions_10[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<10>
};

static VCleanerProcessor* c_functions_12[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<12>
};

static VCleanerProcessor* c_functions_14[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<14>
};

static VCleanerProcessor* c_functions_16[] = {
  vcleaner_do_nothing,
  vcleaner_copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<16>
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
extern VCleanerProcessor* vcleaner_sse4_functions_uint16_10[];
extern VCleanerProcessor* vcleaner_sse4_functions_uint16_12[];
extern VCleanerProcessor* vcleaner_sse4_functions_uint16_14[];
extern VCleanerProcessor* vcleaner_sse4_functions_uint16_16[];
extern VCleanerProcessor* vcleaner_sse2_functions_32[];
extern VCleanerProcessor* vcleaner_sse2_functions_32_chroma[];
#endif

static void dispatch_median(int mode, Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int pixelsize, int bits_per_pixel, bool chroma, IScriptEnvironment *env) {
  // a plane without an inner row (mode 1: 1 border row, mode 2: 2 border rows at the top and the bottom) is copied
  if ((mode == 1 && height < 3) || (mode == 2 && height < 5))
    mode = 0;
  if (pixelsize == 1) {
#ifdef INTEL_INTRINSICS
    if ((env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16 && is_16byte_aligned(pSrc)) {
      vcleaner_sse2_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
      return;
    }
#endif
    c_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
  }
  else if (pixelsize == 2) {
#ifdef INTEL_INTRINSICS
    if ((env->GetCPUFlags() & CPUF_SSE4) && rowsize >= 16 && is_16byte_aligned(pSrc)) {
      switch (bits_per_pixel) {
      case 10: vcleaner_sse4_functions_uint16_10[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 12: vcleaner_sse4_functions_uint16_12[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 14: vcleaner_sse4_functions_uint16_14[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 16: vcleaner_sse4_functions_uint16_16[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      }
      return;
    }
#endif
    switch (bits_per_pixel) {
    case 10: c_functions_10[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
    case 12: c_functions_12[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
    case 14: c_functions_14[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
    case 16: c_functions_16[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
    }
  }
  else { // if (pixelsize == 4
#ifdef INTEL_INTRINSICS
    if ((env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16 && is_16byte_aligned(pSrc)) {
      if (chroma)
        vcleaner_sse2_functions_32_chroma[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
      else
        vcleaner_sse2_functions_32[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
      return;
    }
#endif
    if (chroma)
      c_functions_32_chroma[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
    else
      c_functions_32[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
  }
}


VerticalCleaner::VerticalCleaner(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
: GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV) {
    // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar()) {
        env->ThrowError("VerticalCleaner works only with planar colorspaces (YUY2: use ConvertToYV16)");
    }

    // modeU/modeV below -1: same as the previous plane (UNDEFINED_MODE, the default)
    if (mode_ <= UNDEFINED_MODE || mode_ > 2 || modeU_ > 2 || modeV_ > 2) {
        env->ThrowError("Sorry, this mode does not exist");
    }

    bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
      env->ThrowError("VerticalCleaner: cannot specify U or V mode for planar RGB!");
    }

    if (modeU_ <= UNDEFINED_MODE) {
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

}

PVideoFrame VerticalCleaner::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      const bool chroma = false;
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetReadPtr(PLANAR_G), dstFrame->GetPitch(PLANAR_G), srcFrame->GetPitch(PLANAR_G),
        srcFrame->GetRowSize(PLANAR_G), srcFrame->GetHeight(PLANAR_G), pixelsize, bits_per_pixel, chroma, env);
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetReadPtr(PLANAR_B), dstFrame->GetPitch(PLANAR_B), srcFrame->GetPitch(PLANAR_B),
        srcFrame->GetRowSize(PLANAR_B), srcFrame->GetHeight(PLANAR_B), pixelsize, bits_per_pixel, chroma, env);
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetReadPtr(PLANAR_R), dstFrame->GetPitch(PLANAR_R), srcFrame->GetPitch(PLANAR_R),
        srcFrame->GetRowSize(PLANAR_R), srcFrame->GetHeight(PLANAR_R), pixelsize, bits_per_pixel, chroma, env);
    }
    else {
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
        srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y), pixelsize, bits_per_pixel, false, env);

      if (!vi.IsY()) {
        // chroma planes
        dispatch_median(modeU_, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U), pixelsize, bits_per_pixel, true, env);

        dispatch_median(modeV_, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V), pixelsize, bits_per_pixel, true, env);
      }
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    { // copy alpha
      env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
    }
    return dstFrame;
}

AVSValue __cdecl Create_VerticalCleaner(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR };
    return new VerticalCleaner(
        args[CLIP].AsClip(), 
        args[MODE].AsInt(1),
        args[MODEU].AsInt(VerticalCleaner::UNDEFINED_MODE),
        args[MODEV].AsInt(VerticalCleaner::UNDEFINED_MODE),
        args[PLANAR].AsBool(false), 
        env);
}

