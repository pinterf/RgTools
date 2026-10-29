// TemporalRepair from RemoveGrainT 1.0 package"
//
// An Avisynth plugin for removing grain from progressive video
//
// 2007 By Rainer Wittmann <gorw@gmx.de>
// 2019 Rewrite and additional work by Ferenc Pintér

#include "RemoveGrainT.h"

// TemporalRepair: AviSynth front end (the processing is in RemoveGrainT_core.cpp)

static void CompareVideoInfo(VideoInfo& vi1, const VideoInfo& vi2, const char* progname, IScriptEnvironment* env)
{
  if ((vi1.width != vi2.width) || (vi1.height != vi2.height) || !vi1.IsSameColorspace(vi2))
  {
    env->ThrowError("%s: clips must be of equal type", progname);
  }
  if (vi1.num_frames > vi2.num_frames)
    vi1.num_frames = vi2.num_frames;
}

static void copy_plane(PVideoFrame& destf, PVideoFrame& currf, int plane, IScriptEnvironment* env) {
  const uint8_t* srcp = currf->GetReadPtr(plane);
  int src_pitch = currf->GetPitch(plane);
  int height = currf->GetHeight(plane);
  int row_size = currf->GetRowSize(plane);
  uint8_t* destp = destf->GetWritePtr(plane);
  int dst_pitch = destf->GetPitch(plane);
  env->BitBlt(destp, dst_pitch, srcp, src_pitch, row_size, height);
}

// modes 0, 4: TemporalRepair; modes 1, 2, 3: SmoothTemporalRepair
class TemporalRepair : public GenericVideoFilter
{
  PClip orig;
  TemporalRepairCore core_;
  int last_frame;
  bool has_at_least_v8;

  // MT mode Registration for Avisynth+
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override
  {
    if (n <= 0 || n - 1 >= last_frame)
      return child->GetFrame(n, env);
    PVideoFrame pf = orig->GetFrame(n - 1, env);
    PVideoFrame of = orig->GetFrame(n, env);
    PVideoFrame nf = orig->GetFrame(n + 1, env);
    PVideoFrame cf = child->GetFrame(n, env);
    PVideoFrame df = has_at_least_v8 ? env->NewVideoFrameP(vi, &cf) : env->NewVideoFrame(vi);

    const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int planes_r[3] = { PLANAR_G, PLANAR_B, PLANAR_R };
    const int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    const int num_planes = std::min(vi.NumComponents(), 3); // no Alpha plane processing

    for (int i = 0; i < num_planes; ++i) {
      const int plane = planes[i];
      const RgPlane p = { cf->GetReadPtr(plane), cf->GetPitch(plane), df->GetWritePtr(plane), df->GetPitch(plane),
        cf->GetRowSize(plane), cf->GetHeight(plane) };
      core_.process_plane(i, p, pf->GetReadPtr(plane), pf->GetPitch(plane), of->GetReadPtr(plane), of->GetPitch(plane),
        nf->GetReadPtr(plane), nf->GetPitch(plane));
    }

    // alpha: modes 0, 4 from the original clip, modes 1, 2, 3 from the clip
    if (vi.NumComponents() == 4)
      copy_plane(df, core_.smooth() ? cf : of, PLANAR_A, env);

    return df;
  }

public:
  TemporalRepair(PClip clip, PClip oclip, int mode, bool grey, bool planar, int opt, IScriptEnvironment* env) :
    GenericVideoFilter(clip),
    orig(oclip)
  {
    // planar=true is a dummy compatibility parameter: YUY2 and packed RGB are not supported
    if (!vi.IsPlanar())
      env->ThrowError("TemporalRepair works only with planar colorspaces (YUY2: use ConvertToYV16)");

    CompareVideoInfo(vi, orig->GetVideoInfo(), "TemporalRepair", env);

    const TemporalRepairParams params = { mode, grey, opt };
    const char* error = core_.init(params, rg_format_from_vi(vi), rg_cpu_flags_from_avs(env->GetCPUFlags()));
    if (error)
      env->ThrowError("%s", error);

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    last_frame = vi.num_frames >= 2 ? vi.num_frames - 2 : 0;
  }
};

AVSValue __cdecl Create_TemporalRepair(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  enum ARGS { CLIP, OCLIP, MODE, SMOOTH, GREY, PLANAR, OPT };
  // mode and smooth are the same, probably for historical reasons.
  const int mode = args[MODE].AsInt(args[SMOOTH].AsInt(0));
  return new TemporalRepair(args[CLIP].AsClip(), args[OCLIP].AsClip(), mode, args[GREY].AsBool(false), args[PLANAR].AsBool(false),
    args[OPT].AsInt(-1), env);
}
