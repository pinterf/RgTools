#include "repair_functions_c.h"
#include "repair_functions_sse.h"
#include "repair.h"


template<SseModeProcessor processor, InstructionSet optLevel>
static void process_plane_sse(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int width, int height) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
    int mod16_width = width / 16 * 16;

    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < mod16_width-1; x+=16) {
            __m128i val = simd_loadu_si128<optLevel>(pSrc+x);
            __m128i result = processor(pRef+x, val, refPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod16_width != width) {
            __m128i val = simd_loadu_si128<optLevel>(pSrc+width-17);
            __m128i result = processor(pRef+width-17, val, refPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+width-17), result);
        }


        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
        pRef += refPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}

template<CModeProcessor processor>
static void process_plane_c(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int width, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  for (int y = 1; y < height-1; ++y) {
    pDst[0] = pSrc[0];
    for (int x = 1; x < width-1; x+=1) {
      BYTE result = processor(pRef + x, pSrc[x], srcPitch);
      pDst[x] = result;
    }
    pDst[width-1] = pSrc[width-1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}


static void doNothing(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int width, int height) {

}

static void copyPlane(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int width, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, height);
}


RepairPlaneProcessor* sse3_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<repair_mode1_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode2_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode3_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode4_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode5_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode6_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode7_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode8_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode9_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode10_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode1_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode12_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode13_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode14_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode15_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode16_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode17_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode18_sse<SSE3>, SSE3>,
    process_plane_sse<repair_mode19_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode20_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode21_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode22_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode23_sse<SSE3>, SSE3>, 
    process_plane_sse<repair_mode24_sse<SSE3>, SSE3> 
};

RepairPlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<repair_mode1_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode2_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode3_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode4_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode5_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode6_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode7_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode8_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode9_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode10_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode1_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode12_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode13_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode14_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode15_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode16_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode17_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode18_sse<SSE2>, SSE2>,
    process_plane_sse<repair_mode19_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode20_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode21_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode22_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode23_sse<SSE2>, SSE2>, 
    process_plane_sse<repair_mode24_sse<SSE2>, SSE2> 
};

RepairPlaneProcessor* c_functions[] = {
  doNothing,
  copyPlane,
  process_plane_c<repair_mode1_cpp>,
  process_plane_c<repair_mode2_cpp>,
  process_plane_c<repair_mode3_cpp>,
  process_plane_c<repair_mode4_cpp>,
  process_plane_c<repair_mode5_cpp>, 
  process_plane_c<repair_mode6_cpp>, 
  process_plane_c<repair_mode7_cpp>, 
  process_plane_c<repair_mode8_cpp>, 
  process_plane_c<repair_mode9_cpp>, 
  process_plane_c<repair_mode10_cpp>,
  process_plane_c<repair_mode1_cpp>,
  process_plane_c<repair_mode12_cpp>,
  process_plane_c<repair_mode13_cpp>,
  process_plane_c<repair_mode14_cpp>,
  process_plane_c<repair_mode15_cpp>,
  process_plane_c<repair_mode16_cpp>,
  process_plane_c<repair_mode17_cpp>,
  process_plane_c<repair_mode18_cpp>,
  process_plane_c<repair_mode19_cpp>, 
  process_plane_c<repair_mode20_cpp>, 
  process_plane_c<repair_mode21_cpp>, 
  process_plane_c<repair_mode22_cpp>, 
  process_plane_c<repair_mode23_cpp>, 
  process_plane_c<repair_mode24_cpp> 
};

Repair::Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
  : GenericVideoFilter(child), ref_(ref), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {

    auto refVi = ref_->GetVideoInfo();

    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("Repair works only with planar colorspaces");
    }

    if (vi.width != refVi.width || vi.height != refVi.height) {
        env->ThrowError("Clips should be of the same size!");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 24 || modeU_ > 24 || modeV_ > 24) {
        env->ThrowError("Repair mode should be between -1 and 24!");
    }

    //now change undefined mode value and EVERYTHING WILL BREAK
    if (modeU_ <= UNDEFINED_MODE) { 
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    if (vi.IsPlanar() && !vi.IsY() && (modeU_ != -1 || modeV_ != -1)) {
        if (!vi.IsSameColorspace(refVi)) {
            env->ThrowError("Both clips should have the same colorspace!");
        }
    }

    functions = (env->GetCPUFlags() & CPUF_SSE3) ? sse3_functions 
        : (env->GetCPUFlags() & CPUF_SSE2) ? sse2_functions
        : c_functions;

    if (vi.width < 17) { //not enough for XMM
        functions = c_functions;
    }
}


PVideoFrame Repair::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto refFrame = ref_->GetFrame(n, env);
  auto dstFrame = env->NewVideoFrame(vi);

  functions[mode_+1](env, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), refFrame->GetReadPtr(PLANAR_Y),
      dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), refFrame->GetPitch(PLANAR_Y),
      srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y));

  if (vi.IsPlanar() && !vi.IsY()) {
      functions[modeU_+1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

      functions[modeV_+1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
  }
  return dstFrame;
}


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, REF, MODE, MODEU, MODEV, PLANAR };
    return new Repair(args[CLIP].AsClip(), args[REF].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(Repair::UNDEFINED_MODE), args[MODEV].AsInt(Repair::UNDEFINED_MODE), args[PLANAR].AsBool(false), env);
}
