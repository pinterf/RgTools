#include "vertical_cleaner.h"
#include <xutility>


static void vcleaner_median_sse2(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;

    for (int y = 1; y < height-1; ++y) {
        for (int x = 0; x < width; x+=16) {
            __m128i up     = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x - srcPitch));
            __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
            __m128i down   = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x + srcPitch));

            __m128i mi = _mm_min_epu8(up, down);
            __m128i ma = _mm_max_epu8(up, down);

            __m128i cma = _mm_max_epu8(mi, center);
            __m128i dst = _mm_min_epu8(cma, ma);

            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}

static void vcleaner_relaxed_median_sse2(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    for (int y = 2; y < height-2; ++y) {
        for (int x = 0; x < width; x+=16) {
            __m128i p2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch*2));
            __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch));
            __m128i c  = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
            __m128i n1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch));
            __m128i n2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch*2));

            __m128i pdiff = _mm_subs_epu8(p1, p2);
            __m128i ndiff = _mm_subs_epu8(n1, n2);

            __m128i pt = _mm_adds_epu8(pdiff, p1);
            __m128i nt = _mm_adds_epu8(ndiff, n1);

            __m128i upper = _mm_min_epu8(pt, nt);
            upper = _mm_max_epu8(upper, p1);
            upper = _mm_max_epu8(upper, n1);

            pdiff = _mm_subs_epu8(p2, p1);
            ndiff = _mm_subs_epu8(n2, n1);

            pt = _mm_subs_epu8(p1, pdiff);
            nt = _mm_subs_epu8(n1, ndiff);

            __m128i minpn1 = _mm_min_epu8(p1, n1);

            __m128i lower = _mm_max_epu8(pt, nt);
            lower = _mm_min_epu8(lower, minpn1);

            __m128i dst = simd_clip(c, lower, upper);

            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);
}


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

static __forceinline Byte satb(int value) {
    return clip(value, 0, 255);
}

static __forceinline uint16_t satb_16(int value, int max_pixel_value) {
  return clip_16(value, 0, max_pixel_value);
}

static __forceinline float satb_32(float value) {
  return clip_32(value, 0.0f, 1.0f);
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

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    const int width = rowsize / sizeof(uint16_t);

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

      float upper = std::max(std::max(std::min(satb_32(satb_32(p1-p2) + p1), satb_32(satb_32(n1-n2) + n1)), p1), n1);
      float lower = std::min(std::min(p1, n1), std::max(satb_32(p1 - satb_32(p2-p1)), satb_32(n1 - satb_32(n2-n1))));

      pDst[x] = clip_32(c, lower, upper);
    }

    pSrc += srcPitch;
    pDst += dstPitch;
  }

  env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(float), (uint8_t *)pSrc, srcPitch*sizeof(float), rowsize, 2);
}


static void copy_plane(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

static void do_nothing(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, IScriptEnvironment *env) {

}

/*
VCleanerProcessor* sse4_functions_uint16[] = {
  do_nothing,
  copy_plane<uint16_t>,
  vcleaner_median_sse4<uint16_t>,
  vcleaner_relaxed_median_sse4<uint16_t>
};
*/

VCleanerProcessor* sse2_functions[] = {
    do_nothing,
    copy_plane,
    vcleaner_median_sse2,
    vcleaner_relaxed_median_sse2
};

VCleanerProcessor* c_functions[] = {
    do_nothing,
    copy_plane,
    vcleaner_median_c<uint8_t>,
    vcleaner_relaxed_median_c
};

// distinct templates for 10-12-14-16 bit for const max_pixel_value
VCleanerProcessor* c_functions_10[] = {
  do_nothing,
  copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<10>
};

VCleanerProcessor* c_functions_12[] = {
  do_nothing,
  copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<12>
};

VCleanerProcessor* c_functions_14[] = {
  do_nothing,
  copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<14>
};

VCleanerProcessor* c_functions_16[] = {
  do_nothing,
  copy_plane,
  vcleaner_median_c<uint16_t>,
  vcleaner_relaxed_median_c_16<16>
};

VCleanerProcessor* c_functions_32[] = {
  do_nothing,
  copy_plane,
  vcleaner_median_c<float>,
  vcleaner_relaxed_median_c_32
};

static void dispatch_median(int mode, Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int rowsize, int height, int pixelsize, int bits_per_pixel, IScriptEnvironment *env) {
  if (pixelsize == 1) {
    if ((env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16 && is_16byte_aligned(pSrc)) {
      sse2_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
    }
    else {
      c_functions[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
    }
  }
  else if (pixelsize == 2) {
    /*
    if ((env->GetCPUFlags() & CPUF_SSE4) && width*pixelsize >= 16 && is_16byte_aligned(pSrc)) {
      sse4_functions_uint16[mode + 1](pDst, pSrc, dstPitch, srcPitch, width, height, bits_per_pixel, env);
    }
    else*/ {
      switch (bits_per_pixel) {
      case 10: c_functions_10[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 12: c_functions_12[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 14: c_functions_14[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      case 16: c_functions_16[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env); break;
      }
    }
  }
  else { // if (pixelsize == 4
    c_functions_32[mode + 1](pDst, pSrc, dstPitch, srcPitch, rowsize, height, env);
  }

}

VerticalCleaner::VerticalCleaner(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
: GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("VerticalCleaner works only with planar colorspaces");
    }

    if (mode_ > 2 || modeU_ > 2 || modeV_ > 2) {
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
}

PVideoFrame VerticalCleaner::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);

    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetReadPtr(PLANAR_G), dstFrame->GetPitch(PLANAR_G), srcFrame->GetPitch(PLANAR_G),
        srcFrame->GetRowSize(PLANAR_G) / pixelsize, srcFrame->GetHeight(PLANAR_G), pixelsize, bits_per_pixel, env);
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetReadPtr(PLANAR_B), dstFrame->GetPitch(PLANAR_B), srcFrame->GetPitch(PLANAR_B),
        srcFrame->GetRowSize(PLANAR_B) / pixelsize, srcFrame->GetHeight(PLANAR_B), pixelsize, bits_per_pixel, env);
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetReadPtr(PLANAR_R), dstFrame->GetPitch(PLANAR_R), srcFrame->GetPitch(PLANAR_R),
        srcFrame->GetRowSize(PLANAR_R) / pixelsize, srcFrame->GetHeight(PLANAR_R), pixelsize, bits_per_pixel, env);
    }
    else {
      dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
        srcFrame->GetRowSize(PLANAR_Y) / pixelsize, srcFrame->GetHeight(PLANAR_Y), pixelsize, bits_per_pixel, env);

      if (!vi.IsY()) {
        dispatch_median(modeU_, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U) / pixelsize, srcFrame->GetHeight(PLANAR_U), pixelsize, bits_per_pixel, env);

        dispatch_median(modeV_, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V) / pixelsize, srcFrame->GetHeight(PLANAR_V), pixelsize, bits_per_pixel, env);
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

