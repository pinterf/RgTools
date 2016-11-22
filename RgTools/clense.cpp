#include "clense.h"
#include <xutility>

static void check_if_match(const VideoInfo &vi, const VideoInfo &otherVi, IScriptEnvironment* env) {
    if (otherVi.height != vi.height || otherVi.width != vi.width) {
        env->ThrowError("Frame dimensions do not match");
    }
    if (!otherVi.IsSameColorspace(vi)) {
        env->ThrowError("Colorspaces do not match");
    }
}

RG_FORCEINLINE Byte clense_process_pixel_c(Byte src, Byte ref1, Byte ref2) {
    return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE uint16_t clense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE float clense_process_pixel_c_32(float src, float ref1, float ref2) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE Byte sclense_process_pixel_c(Byte src, Byte ref1, Byte ref2) {
  Byte minref = std::min(ref1, ref2);
  Byte maxref = std::max(ref1, ref2);

  return clip(src, (Byte)std::max(minref*2 - ref2, 0), (Byte)std::min(maxref*2-ref2, 255));
}

template<int bits_per_pixel>
RG_FORCEINLINE uint16_t sclense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2) {
  uint16_t minref = std::min(ref1, ref2);
  uint16_t maxref = std::max(ref1, ref2);

  const int pixel_max = (1 << bits_per_pixel) - 1;
  return clip_16(src, (uint16_t)std::max(minref*2 - ref2, 0), (uint16_t)std::min(maxref*2-ref2, pixel_max));
}

RG_FORCEINLINE float sclense_process_pixel_c_32(float src, float ref1, float ref2) {
  float minref = std::min(ref1, ref2);
  float maxref = std::max(ref1, ref2);

  return clip_32(src, std::max(minref*2 - ref2, 0.0f), std::min(maxref*2-ref2, 1.0f));
}

template<typename pixel_t, CModeProcessor<pixel_t>/*(clense_process_pixel_c)*/ processor>
static void process_plane_c(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, IScriptEnvironment *env) {
  const int width = rowsize / sizeof(pixel_t);
  for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            reinterpret_cast<pixel_t *>(pDst)[x] = processor(reinterpret_cast<const pixel_t *>(pSrc)[x], reinterpret_cast<const pixel_t *>(pRef1)[x], reinterpret_cast<const pixel_t *>(pRef2)[x]);
        }
        pDst += dstPitch;
        pSrc += srcPitch;
        pRef1 += ref1Pitch;
        pRef2 += ref2Pitch;
    }
}

RG_FORCEINLINE void clense_process_line_sse2(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int width) {
    for (int x = 0; x < width; x+=16) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
        auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
        auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

        auto minref = _mm_min_epu8(ref1, ref2);
        auto maxref = _mm_max_epu8(ref1, ref2);
        auto dst = simd_clip(src, minref, maxref);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
    }
}

RG_FORCEINLINE void sclense_process_line_sse2(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int width) {
    for (int x = 0; x < width; x+=16) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc+x));
        auto ref1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef1+x));
        auto ref2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pRef2+x));

        auto minref = _mm_min_epu8(ref1, ref2);
        auto maxref = _mm_max_epu8(ref1, ref2);

        auto ma = _mm_subs_epu8(maxref, ref2);
        auto mi = _mm_subs_epu8(ref2, minref);

        ma = _mm_adds_epu8(ma, maxref);
        mi = _mm_subs_epu8(minref, mi);

        auto dst = simd_clip(src, mi, ma);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
    }
}

template<decltype(clense_process_line_sse2) processor>
void process_plane_sse2(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int width, int height, IScriptEnvironment *env) {
    if (!is_16byte_aligned(pSrc) || !is_16byte_aligned(pRef1) || !is_16byte_aligned(pRef2)) {
        env->ThrowError("Invalid memory alignment. Used unaligned crop?"); //omg I feel so dumb
    }
    auto mod16Width = (width / 16) * 16;

    for (int y = 0; y < height; ++y) {
        processor(pDst, pSrc, pRef1, pRef2, mod16Width);

        if (mod16Width != width) {
            processor(pDst + mod16Width, pSrc + mod16Width, pRef1 + mod16Width, pRef2 + mod16Width, 16);
        }
        pDst += dstPitch;
        pSrc += srcPitch;
        pRef1 += ref1Pitch;
        pRef2 += ref2Pitch;
    }
}

Clense::Clense(PClip child, PClip previous, PClip next, bool grey, ClenseMode mode, IScriptEnvironment* env)
    : GenericVideoFilter(child), previous_(previous), next_(next), grey_(grey), mode_(mode) {
    if(!vi.IsPlanar()) {
        env->ThrowError("Clense works only with planar colorspaces");
    }

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    if (previous_ != nullptr) {
        check_if_match(vi, previous_->GetVideoInfo(), env);
    }
    if (next_ != nullptr) {
        check_if_match(vi, next_->GetVideoInfo(), env);
    }
    sse2_ = vi.width > 16 && (env->GetCPUFlags() & CPUF_SSE2);

    if (pixelsize == 1) {
      processor_ = (mode_ == ClenseMode::BOTH)
        ? (sse2_ ? process_plane_sse2<clense_process_line_sse2> : process_plane_c<uint8_t, clense_process_pixel_c>)
        : (sse2_ ? process_plane_sse2<sclense_process_line_sse2> : process_plane_c<uint8_t, sclense_process_pixel_c>);
    }
    else if (pixelsize == 2) {
      switch (bits_per_pixel) {
      case 10: processor_ = (mode_ == ClenseMode::BOTH)
        ? (process_plane_c<uint16_t, clense_process_pixel_c_16>)
        : (process_plane_c<uint16_t, sclense_process_pixel_c_16<10>>);
        break;
      case 12: processor_ = (mode_ == ClenseMode::BOTH)
        ? (process_plane_c<uint16_t, clense_process_pixel_c_16>)
        : (process_plane_c<uint16_t, sclense_process_pixel_c_16<12>>);
        break;
      case 14: processor_ = (mode_ == ClenseMode::BOTH)
        ? (process_plane_c<uint16_t, clense_process_pixel_c_16>)
        : (process_plane_c<uint16_t, sclense_process_pixel_c_16<14>>);
        break;
      case 16: processor_ = (mode_ == ClenseMode::BOTH)
        ? (process_plane_c<uint16_t, clense_process_pixel_c_16>)
        : (process_plane_c<uint16_t, sclense_process_pixel_c_16<16>>);
        break;
      default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
      }
    }
    else { // pixelsize == 4
      processor_ = (mode_ == ClenseMode::BOTH)
        ? (process_plane_c<float, clense_process_pixel_c_32>)
        : (process_plane_c<float, sclense_process_pixel_c_32>);
    }
}

PVideoFrame Clense::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);

    if (mode_ == ClenseMode::FORWARD && (n == vi.num_frames - 2 || n == vi.num_frames - 1)) {
        return srcFrame;
    }
    if (mode_ == ClenseMode::BACKWARD && (n == 1 || n == 0)) {
        return srcFrame;
    }
    if (mode_ == ClenseMode::BOTH && (n == 0 || n == vi.num_frames - 1))  {
        return srcFrame;
    }

    PVideoFrame frame1, frame2;

    if (mode_ == ClenseMode::BACKWARD) {
        frame1 = child->GetFrame(n-1, env);
        frame2 = child->GetFrame(n-2, env);
    } else if (mode_ == ClenseMode::FORWARD) {
        frame1 = child->GetFrame(n+1, env);
        frame2 = child->GetFrame(n+2, env);
    } else {
        frame1 = previous_ == nullptr ? child->GetFrame(n-1, env) : previous_->GetFrame(n-1, env);
        frame2 = next_ == nullptr ? child->GetFrame(n+1, env) : next_->GetFrame(n+1, env);
    }

    auto dstFrame = env->NewVideoFrame(vi);

    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      processor_(dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetReadPtr(PLANAR_G), frame1->GetReadPtr(PLANAR_G), frame2->GetReadPtr(PLANAR_G),
        dstFrame->GetPitch(PLANAR_G), srcFrame->GetPitch(PLANAR_G), frame1->GetPitch(PLANAR_G), frame2->GetPitch(PLANAR_G),
        srcFrame->GetRowSize(PLANAR_G), srcFrame->GetHeight(PLANAR_G), env);
      processor_(dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetReadPtr(PLANAR_B), frame1->GetReadPtr(PLANAR_B), frame2->GetReadPtr(PLANAR_B),
        dstFrame->GetPitch(PLANAR_B), srcFrame->GetPitch(PLANAR_B), frame1->GetPitch(PLANAR_B), frame2->GetPitch(PLANAR_B),
        srcFrame->GetRowSize(PLANAR_B), srcFrame->GetHeight(PLANAR_B), env);
      processor_(dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetReadPtr(PLANAR_R), frame1->GetReadPtr(PLANAR_R), frame2->GetReadPtr(PLANAR_R),
        dstFrame->GetPitch(PLANAR_R), srcFrame->GetPitch(PLANAR_R), frame1->GetPitch(PLANAR_R), frame2->GetPitch(PLANAR_R),
        srcFrame->GetRowSize(PLANAR_R), srcFrame->GetHeight(PLANAR_R), env);
    } else {
      processor_(dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), frame1->GetReadPtr(PLANAR_Y), frame2->GetReadPtr(PLANAR_Y),
        dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), frame1->GetPitch(PLANAR_Y), frame2->GetPitch(PLANAR_Y),
        srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y), env);

      if (!vi.IsY() && !grey_) {
        processor_(dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), frame1->GetReadPtr(PLANAR_U), frame2->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), frame1->GetPitch(PLANAR_U), frame2->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U), env);

        processor_(dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), frame1->GetReadPtr(PLANAR_V), frame2->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), frame1->GetPitch(PLANAR_V), frame2->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V), env);
      }
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    { // copy alpha
      env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
    }
    return dstFrame;
}

AVSValue __cdecl Create_Clense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, PREVIOUS, NEXT, GREY };
    return new Clense(args[CLIP].AsClip(), args[PREVIOUS].Defined() ? args[PREVIOUS].AsClip() : nullptr, args[NEXT].Defined() ? args[NEXT].AsClip() : nullptr, args[GREY].AsBool(false), ClenseMode::BOTH, env);
}

AVSValue __cdecl Create_ForwardClense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, GREY };
    return new Clense(args[CLIP].AsClip(), nullptr, nullptr, args[GREY].AsBool(false), ClenseMode::FORWARD, env);
}

AVSValue __cdecl Create_BackwardClense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, GREY };
    return new Clense(args[CLIP].AsClip(), nullptr, nullptr, args[GREY].AsBool(false), ClenseMode::BACKWARD, env);
}