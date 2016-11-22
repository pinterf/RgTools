#include "repair_functions_c.h"
#include "repair_functions_sse.h"
#include "repair.h"


template<SseModeProcessor processor, InstructionSet optLevel>
static void process_plane_sse(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
    int mod16_width = rowsize / 16 * 16;

    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < mod16_width-1; x+=16) {
            __m128i val = simd_loadu_si128<optLevel>(pSrc+x);
            __m128i result = processor(pRef+x, val, refPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod16_width != rowsize) {
            __m128i val = simd_loadu_si128<optLevel>(pSrc+rowsize-17);
            __m128i result = processor(pRef+rowsize-17, val, refPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+rowsize-17), result);
        }


        pDst[rowsize-1] = pSrc[rowsize-1];

        pSrc += srcPitch;
        pDst += dstPitch;
        pRef += refPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_plane_c(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);

  const int width = rowsize / sizeof(pixel_t);

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  for (int y = 1; y < height-1; ++y) {
    reinterpret_cast<pixel_t *>(pDst)[0] = reinterpret_cast<const pixel_t *>(pSrc)[0];
    for (int x = 1; x < width-1; x+=1) {
      pixel_t result = processor(pRef + x*sizeof(pixel_t), reinterpret_cast<const pixel_t *>(pSrc)[x], srcPitch);
      reinterpret_cast<pixel_t *>(pDst)[x] = result;
    }
    reinterpret_cast<pixel_t *>(pDst)[width-1] = reinterpret_cast<const pixel_t *>(pSrc)[width-1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}


static void doNothing(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {

}

static void copyPlane(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
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
  process_plane_c<uint8_t,repair_mode1_cpp>,
  process_plane_c<uint8_t,repair_mode2_cpp>,
  process_plane_c<uint8_t,repair_mode3_cpp>,
  process_plane_c<uint8_t,repair_mode4_cpp>,
  process_plane_c<uint8_t,repair_mode5_cpp>, 
  process_plane_c<uint8_t,repair_mode6_cpp>, 
  process_plane_c<uint8_t,repair_mode7_cpp>, 
  process_plane_c<uint8_t,repair_mode8_cpp>, 
  process_plane_c<uint8_t,repair_mode9_cpp>, 
  process_plane_c<uint8_t,repair_mode10_cpp>,
  process_plane_c<uint8_t,repair_mode1_cpp>,
  process_plane_c<uint8_t,repair_mode12_cpp>,
  process_plane_c<uint8_t,repair_mode13_cpp>,
  process_plane_c<uint8_t,repair_mode14_cpp>,
  process_plane_c<uint8_t,repair_mode15_cpp>,
  process_plane_c<uint8_t,repair_mode16_cpp>,
  process_plane_c<uint8_t,repair_mode17_cpp>,
  process_plane_c<uint8_t,repair_mode18_cpp>,
  process_plane_c<uint8_t,repair_mode19_cpp>, 
  process_plane_c<uint8_t,repair_mode20_cpp>, 
  process_plane_c<uint8_t,repair_mode21_cpp>, 
  process_plane_c<uint8_t,repair_mode22_cpp>, 
  process_plane_c<uint8_t,repair_mode23_cpp>, 
  process_plane_c<uint8_t,repair_mode24_cpp> 
};

RepairPlaneProcessor* c_functions_10[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<10>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<10>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<10>> 
};

RepairPlaneProcessor* c_functions_12[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<12>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<12>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<12>> 
};

RepairPlaneProcessor* c_functions_14[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<14>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<14>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<14>> 
};

RepairPlaneProcessor* c_functions_16[] = {
  doNothing,
  copyPlane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16<16>>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16<16>>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16<16>> 
};

RepairPlaneProcessor* c_functions_32[] = {
  doNothing,
  copyPlane,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode2_cpp_32>,
  process_plane_c<float,repair_mode3_cpp_32>,
  process_plane_c<float,repair_mode4_cpp_32>,
  process_plane_c<float,repair_mode5_cpp_32>, 
  process_plane_c<float,repair_mode6_cpp_32>, 
  process_plane_c<float,repair_mode7_cpp_32>, 
  process_plane_c<float,repair_mode8_cpp_32>, 
  process_plane_c<float,repair_mode9_cpp_32>, 
  process_plane_c<float,repair_mode10_cpp_32>,
  process_plane_c<float,repair_mode1_cpp_32>,
  process_plane_c<float,repair_mode12_cpp_32>,
  process_plane_c<float,repair_mode13_cpp_32>,
  process_plane_c<float,repair_mode14_cpp_32>,
  process_plane_c<float,repair_mode15_cpp_32>,
  process_plane_c<float,repair_mode16_cpp_32>,
  process_plane_c<float,repair_mode17_cpp_32>,
  process_plane_c<float,repair_mode18_cpp_32>,
  process_plane_c<float,repair_mode19_cpp_32>, 
  process_plane_c<float,repair_mode20_cpp_32>, 
  process_plane_c<float,repair_mode21_cpp_32>, 
  process_plane_c<float,repair_mode22_cpp_32>, 
  process_plane_c<float,repair_mode23_cpp_32>, 
  process_plane_c<float,repair_mode24_cpp_32> 
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

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (pixelsize == 1) {

    functions = (env->GetCPUFlags() & CPUF_SSE3) ? sse3_functions
      : (env->GetCPUFlags() & CPUF_SSE2) ? sse2_functions
      : c_functions;

    if (vi.width < 17) { //not enough for XMM
      functions = c_functions;
    }
  }
  else if (pixelsize == 2) {
    switch (bits_per_pixel) {
    case 10: functions = c_functions_10; break;
    case 12: functions = c_functions_12; break;
    case 14: functions = c_functions_14; break;
    case 16: functions = c_functions_16; break;
    default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
    }
  }
  else {// if (pixelsize == 4) 
    functions = c_functions_32;
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
