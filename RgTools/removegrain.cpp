// RemoveGrain: AviSynth front end (the processing is in removegrain_core.cpp)

#include "removegrain.h"

RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(child) {
    // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar()) {
        env->ThrowError("RemoveGrain works only with planar colorspaces (YUY2: use ConvertToYV16)");
    }

    const RemoveGrainParams params = { mode, modeU, modeV, opt };
    const char* error = core_.init(params, rg_format_from_vi(vi), rg_cpu_flags_from_avs(env->GetCPUFlags()));
    if (error)
        env->ThrowError("%s", error);

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }
}

PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

  const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  const int planes_r[3] = { PLANAR_G, PLANAR_B, PLANAR_R };
  const bool rgb = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  const int* planes = rgb ? planes_r : planes_y;
  const int num_planes = rg_num_planes(vi);

  for (int i = 0; i < num_planes; ++i) {
    const int plane = planes[i];
    const RgPlane p = { srcFrame->GetReadPtr(plane), srcFrame->GetPitch(plane), dstFrame->GetWritePtr(plane), dstFrame->GetPitch(plane),
      srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane) };
    core_.process_plane(i, p);
  }

  if (rg_has_alpha(vi))
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR, OPT };
    AVSValue retval = new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrainCore::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrainCore::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), args[OPT].AsInt(0), env);
    return retval;
}
