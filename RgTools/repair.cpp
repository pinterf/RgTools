// Repair: AviSynth front end (the processing is in repair_core.cpp)

#include "repair.h"

Repair::Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
  : GenericVideoFilter(child), ref_(ref) {

  auto refVi = ref_->GetVideoInfo();

  // planar=true (skip_cs_check) is a dummy compatibility parameter: YUY2 and packed RGB are not supported
  if (!vi.IsPlanar() || !refVi.IsPlanar()) {
    env->ThrowError("Repair works only with planar colorspaces (YUY2: use ConvertToYV16)");
  }

  const RepairParams params = { mode, modeU, modeV, opt };
  const char* error = core_.init(params, rg_format_from_vi(vi), rg_format_from_vi(refVi), rg_cpu_flags_from_avs(env->GetCPUFlags()));
  if (error)
    env->ThrowError("%s", error);

  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }
}

PVideoFrame Repair::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto refFrame = ref_->GetFrame(n, env);
  auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

  const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
  const int planes_r[3] = { PLANAR_G, PLANAR_B, PLANAR_R };
  const bool rgb = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  const int* planes = rgb ? planes_r : planes_y;
  const int num_planes = vi.IsY() ? 1 : 3;

  for (int i = 0; i < num_planes; ++i) {
    const int plane = planes[i];
    // the SSE2 kernels of the inner part use aligned loads (YUV: checked at the first plane and at U)
    if (!rgb && i < 2 && (!is_16byte_aligned(srcFrame->GetReadPtr(plane)) || !is_16byte_aligned(refFrame->GetReadPtr(plane))))
      env->ThrowError("Repair: Invalid memory alignment. Unaligned crop?");
    const RgPlane p = { srcFrame->GetReadPtr(plane), srcFrame->GetPitch(plane), dstFrame->GetWritePtr(plane), dstFrame->GetPitch(plane),
      srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane) };
    core_.process_plane(i, p, refFrame->GetReadPtr(plane), refFrame->GetPitch(plane));
  }

  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, REF, MODE, MODEU, MODEV, PLANAR, OPT };
    return new Repair(args[CLIP].AsClip(), args[REF].AsClip(), 
      args[MODE].AsInt(1), 
      args[MODEU].AsInt(RepairCore::UNDEFINED_MODE), 
      args[MODEV].AsInt(RepairCore::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), // not used, compatibility parameter
      args[OPT].AsInt(0),
      env);
}
