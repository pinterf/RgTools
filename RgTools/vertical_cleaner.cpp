// VerticalCleaner: AviSynth front end (the processing is in vertical_cleaner_core.cpp)

#include "vertical_cleaner.h"

VerticalCleaner::VerticalCleaner(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
: GenericVideoFilter(child) {
    // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar()) {
        env->ThrowError("VerticalCleaner works only with planar colorspaces (YUY2: use ConvertToYV16)");
    }

    const VerticalCleanerParams params = { mode, modeU, modeV };
    const char* error = core_.init(params, rg_format_from_vi(vi), rg_cpu_flags_from_avs(env->GetCPUFlags()));
    if (error)
        env->ThrowError("%s", error);

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }
}

PVideoFrame VerticalCleaner::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

    const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int planes_r[3] = { PLANAR_G, PLANAR_B, PLANAR_R };
    const int* planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;
    const int num_planes = vi.IsY() ? 1 : 3;

    for (int i = 0; i < num_planes; ++i) {
        const int plane = planes[i];
        const RgPlane p = { srcFrame->GetReadPtr(plane), srcFrame->GetPitch(plane), dstFrame->GetWritePtr(plane), dstFrame->GetPitch(plane),
          srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane) };
        core_.process_plane(i, p);
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
        args[MODEU].AsInt(VerticalCleanerCore::UNDEFINED_MODE),
        args[MODEV].AsInt(VerticalCleanerCore::UNDEFINED_MODE),
        args[PLANAR].AsBool(false),
        env);
}
