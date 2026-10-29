// Clense, ForwardClense, BackwardClense: AviSynth front end (the processing is in clense_core.cpp)

#include "clense.h"

static void check_if_match(const VideoInfo &vi, const VideoInfo &otherVi, IScriptEnvironment* env) {
    if (otherVi.height != vi.height || otherVi.width != vi.width) {
        env->ThrowError("Frame dimensions do not match");
    }
    if (!otherVi.IsSameColorspace(vi)) {
        env->ThrowError("Colorspaces do not match");
    }
}

Clense::Clense(PClip child, PClip previous, PClip next, bool grey, bool reduceflicker, ClenseMode mode, bool skip_cs_check, IScriptEnvironment* env)
    : GenericVideoFilter(child), previous_(previous), next_(next), mode_(mode), reduceflicker_(reduceflicker) {
    // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar()) {
        env->ThrowError("Clense works only with planar colorspaces (YUY2: use ConvertToYV16)");
    }

    const ClenseParams params = { mode, grey };
    const char* error = core_.init(params, rg_format_from_vi(vi), rg_cpu_flags_from_avs(env->GetCPUFlags()));
    if (error)
        env->ThrowError("%s", error);

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    lastDstFrame = nullptr;
    lastRequestedFrameNo = -1;

    if (previous_ != nullptr) {
        check_if_match(vi, previous_->GetVideoInfo(), env);
    }
    if (next_ != nullptr) {
        check_if_match(vi, next_->GetVideoInfo(), env);
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

    const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int planes_r[3] = { PLANAR_G, PLANAR_B, PLANAR_R };
    const int* planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;
    const int num_planes = vi.IsY() ? 1 : 3;

    for (int i = 0; i < num_planes; ++i) {
        const int plane = planes[i];
        const RgPlane p = { srcFrame->GetReadPtr(plane), srcFrame->GetPitch(plane), dstFrame->GetWritePtr(plane), dstFrame->GetPitch(plane),
          srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane) };
        core_.process_plane(i, p, frame1->GetReadPtr(plane), frame1->GetPitch(plane), frame2->GetReadPtr(plane), frame2->GetPitch(plane));
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