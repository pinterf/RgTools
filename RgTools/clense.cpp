#include "clense.h"

static void check_if_match(const VideoInfo &vi, const VideoInfo &otherVi, IScriptEnvironment* env) {
    if (otherVi.height != vi.height || otherVi.width != vi.width) {
        env->ThrowError("Frame dimensions do not match");
    }
    if (!otherVi.IsSameColorspace(vi)) {
        env->ThrowError("Colorspaces do not match");
    }
}

RG_FORCEINLINE Byte clense_process_pixel_c(Byte src, Byte ref1, Byte ref2, int /*max_pixel_value*/) {
    return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE uint16_t clense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2, int /*max_pixel_value*/) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE float clense_process_pixel_c_32(float src, float ref1, float ref2, int /*max_pixel_value*/) {
  return std::min(std::max(src, std::min(ref1, ref2)), std::max(ref1, ref2));
}

RG_FORCEINLINE Byte sclense_process_pixel_c(Byte src, Byte ref1, Byte ref2, int /*max_pixel_value*/) {
  Byte minref = std::min(ref1, ref2);
  Byte maxref = std::max(ref1, ref2);

  return clip(src, (Byte)std::max(minref*2 - ref2, 0), (Byte)std::min(maxref*2-ref2, 255));
}

RG_FORCEINLINE uint16_t sclense_process_pixel_c_16(uint16_t src, uint16_t ref1, uint16_t ref2, int max_pixel_value) {
  uint16_t minref = std::min(ref1, ref2);
  uint16_t maxref = std::max(ref1, ref2);

  uint16_t mi = subs_16_c(minref, subs_16_c(ref2, minref)); // minref - (ref2 - minref)
  uint16_t ma = adds_16_c(subs_16_c(maxref, ref2), maxref, max_pixel_value); // (maxref - ref2) + maxref

  return clip_16(src, mi, ma);
}

template<bool chroma>
RG_FORCEINLINE float sclense_process_pixel_c_32(float src, float ref1, float ref2, int /*max_pixel_value*/) {
  float minref = std::min(ref1, ref2);
  float maxref = std::max(ref1, ref2);

  float mi = subs_32_c<chroma>(minref, subs_32_c<chroma>(ref2, minref)); // minref - (ref2 - minref)
  float ma = adds_32_c<chroma>(subs_32_c<chroma>(maxref, ref2), maxref); // (maxref - ref2) + maxref

  return clip_32(src, mi, ma);
}

template<typename pixel_t, CModeProcessor<pixel_t>/*(clense_process_pixel_c)*/ processor>
static void process_plane_c(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int rowsize, int height, int bits_per_pixel, IScriptEnvironment *env) {
  const int width = rowsize / sizeof(pixel_t);
  const int max_pixel_value = sizeof(pixel_t) == 2 ? (1 << bits_per_pixel) - 1 : 0; // used by the 10-16 bit pixel functions
  for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            reinterpret_cast<pixel_t *>(pDst)[x] = processor(reinterpret_cast<const pixel_t *>(pSrc)[x], reinterpret_cast<const pixel_t *>(pRef1)[x], reinterpret_cast<const pixel_t *>(pRef2)[x], max_pixel_value);
        }
        pDst += dstPitch;
        pSrc += srcPitch;
        pRef1 += ref1Pitch;
        pRef2 += ref2Pitch;
    }
}

Clense::Clense(PClip child, PClip previous, PClip next, bool grey, bool reduceflicker, ClenseMode mode, bool skip_cs_check, IScriptEnvironment* env)
    : GenericVideoFilter(child), previous_(previous), next_(next), grey_(grey), mode_(mode), reduceflicker_(reduceflicker) {
    // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar()) {
        env->ThrowError("Clense works only with planar colorspaces (YUY2: use ConvertToYV16)");
    }

    if(grey_ && (vi.IsPlanarRGB() || vi.IsPlanarRGBA()))
      env->ThrowError("Clense: cannot speficy grey for planar RGB colorspaces");

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    lastDstFrame = nullptr;
    lastRequestedFrameNo = -1;

    if (previous_ != nullptr) {
        check_if_match(vi, previous_->GetVideoInfo(), env);
    }
    if (next_ != nullptr) {
        check_if_match(vi, next_->GetVideoInfo(), env);
    }

    int worst_case_width = vi.width;
    if (vi.IsYUV() && vi.NumComponents() >= 3)
      worst_case_width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);

#ifdef INTEL_INTRINSICS
    sse2_ = worst_case_width > 16 && (env->GetCPUFlags() & CPUF_SSE2);
    sse4_ = worst_case_width > 16 && (env->GetCPUFlags() & CPUF_SSE4);
#else
    sse2_ = sse4_ = false;
#endif

    if (pixelsize == 1) {
      processor_ = (mode_ == ClenseMode::BOTH)
        ? process_plane_c<uint8_t, clense_process_pixel_c>
        : process_plane_c<uint8_t, sclense_process_pixel_c>;
    }
    else if (pixelsize == 2) {
      processor_ = (mode_ == ClenseMode::BOTH)
        ? process_plane_c<uint16_t, clense_process_pixel_c_16>
        : process_plane_c<uint16_t, sclense_process_pixel_c_16>;
    }
    else { // pixelsize == 4
      processor_ = (mode_ == ClenseMode::BOTH)
        ? process_plane_c<float, clense_process_pixel_c_32>
        : process_plane_c<float, sclense_process_pixel_c_32<false>>;
      processor_chroma_ = (mode_ == ClenseMode::BOTH)
        ? process_plane_c<float, clense_process_pixel_c_32>
        : process_plane_c<float, sclense_process_pixel_c_32<true>>;
    }

#ifdef INTEL_INTRINSICS
    // 8 bit and float: SSE2, 10-16 bit: SSE4.1
    if (pixelsize == 2 ? sse4_ : sse2_) {
      processor_ = clense_sse_processor(mode_, pixelsize, false);
      if (pixelsize == 4)
        processor_chroma_ = clense_sse_processor(mode_, pixelsize, true);
    }
#endif
    // only float has a separate chroma variant (chroma range -0.5..0.5)
    if (pixelsize != 4)
      processor_chroma_ = processor_;
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
      // reduceflicker works when frames are requested sequentally
      // maybe multi_instance works
      if (reduceflicker_ && lastRequestedFrameNo == n - 1) // from v0.9
        frame1 = lastDstFrame;
      else
        frame1 = previous_ == nullptr ? child->GetFrame(n-1, env) : previous_->GetFrame(n-1, env);

      frame2 = next_ == nullptr ? child->GetFrame(n+1, env) : next_->GetFrame(n+1, env);
    }

    auto dstFrame = 
      has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      processor_(dstFrame->GetWritePtr(PLANAR_G), srcFrame->GetReadPtr(PLANAR_G), frame1->GetReadPtr(PLANAR_G), frame2->GetReadPtr(PLANAR_G),
        dstFrame->GetPitch(PLANAR_G), srcFrame->GetPitch(PLANAR_G), frame1->GetPitch(PLANAR_G), frame2->GetPitch(PLANAR_G),
        srcFrame->GetRowSize(PLANAR_G), srcFrame->GetHeight(PLANAR_G), bits_per_pixel, env);
      processor_(dstFrame->GetWritePtr(PLANAR_B), srcFrame->GetReadPtr(PLANAR_B), frame1->GetReadPtr(PLANAR_B), frame2->GetReadPtr(PLANAR_B),
        dstFrame->GetPitch(PLANAR_B), srcFrame->GetPitch(PLANAR_B), frame1->GetPitch(PLANAR_B), frame2->GetPitch(PLANAR_B),
        srcFrame->GetRowSize(PLANAR_B), srcFrame->GetHeight(PLANAR_B), bits_per_pixel, env);
      processor_(dstFrame->GetWritePtr(PLANAR_R), srcFrame->GetReadPtr(PLANAR_R), frame1->GetReadPtr(PLANAR_R), frame2->GetReadPtr(PLANAR_R),
        dstFrame->GetPitch(PLANAR_R), srcFrame->GetPitch(PLANAR_R), frame1->GetPitch(PLANAR_R), frame2->GetPitch(PLANAR_R),
        srcFrame->GetRowSize(PLANAR_R), srcFrame->GetHeight(PLANAR_R), bits_per_pixel, env);
    } else {
      processor_(dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), frame1->GetReadPtr(PLANAR_Y), frame2->GetReadPtr(PLANAR_Y),
        dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), frame1->GetPitch(PLANAR_Y), frame2->GetPitch(PLANAR_Y),
        srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y), bits_per_pixel, env);

      if (!vi.IsY() && !grey_) {
        processor_chroma_(dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), frame1->GetReadPtr(PLANAR_U), frame2->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), frame1->GetPitch(PLANAR_U), frame2->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U), bits_per_pixel, env);

        processor_chroma_(dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), frame1->GetReadPtr(PLANAR_V), frame2->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), frame1->GetPitch(PLANAR_V), frame2->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V), bits_per_pixel, env);
      }
      // grey: U and V are not processed and not copied (random content, for speed; as documented for RemoveGrain)
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    { // copy alpha, also with grey (as TemporalRepair)
      env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
    }

    if (reduceflicker_) {
      lastDstFrame = dstFrame;
      lastRequestedFrameNo = n;
    }

    return dstFrame;
}

AVSValue __cdecl Create_Clense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, PREVIOUS, NEXT, GREY, FLICKER, PLANAR, CACHE };
    return new Clense(args[CLIP].AsClip(),
      args[PREVIOUS].Defined() ? args[PREVIOUS].AsClip() : nullptr,
      args[NEXT].Defined() ? args[NEXT].AsClip() : nullptr, args[GREY].AsBool(false), args[FLICKER].AsBool(false), ClenseMode::BOTH, args[PLANAR].AsBool(false), env);
    // planar and cache are dummy parameters for compatibility reasons
}

AVSValue __cdecl Create_ForwardClense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, GREY, PLANAR};
    return new Clense(args[CLIP].AsClip(), nullptr, nullptr, args[GREY].AsBool(false), false, ClenseMode::FORWARD, args[PLANAR].AsBool(false), env);
}

AVSValue __cdecl Create_BackwardClense(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, GREY, PLANAR};
    return new Clense(args[CLIP].AsClip(), nullptr, nullptr, args[GREY].AsBool(false), false, ClenseMode::BACKWARD, args[PLANAR].AsBool(false), env);
}