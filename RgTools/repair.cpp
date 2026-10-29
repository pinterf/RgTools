#include "repair.h"
#include "repair_functions_c.h"

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


void repair_do_nothing(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {

}

void repair_copy_plane(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height) {
  env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}

static RepairPlaneProcessor* c_functions[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<uint8_t,repair_mode24_cpp>,
  repair_do_nothing,
  process_plane_c<uint8_t,repair_mode26_cpp>,
  process_plane_c<uint8_t,repair_mode27_cpp>,
  process_plane_c<uint8_t,repair_mode28_cpp>
};

static RepairPlaneProcessor* c_functions_10[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<10>>,
  repair_do_nothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_12[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<12>>,
  repair_do_nothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_14[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<14>>,
  repair_do_nothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_16[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<uint16_t,repair_mode24_cpp_16<16>>,
  repair_do_nothing,
  process_plane_c<uint16_t,repair_mode26_cpp_16>,
  process_plane_c<uint16_t,repair_mode27_cpp_16>,
  process_plane_c<uint16_t,repair_mode28_cpp_16>,
};

static RepairPlaneProcessor* c_functions_32[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<float,repair_mode19_cpp_32<false>>,
  process_plane_c<float,repair_mode20_cpp_32<false>>,
  process_plane_c<float,repair_mode21_cpp_32<false>>,
  process_plane_c<float,repair_mode22_cpp_32<false>>,
  process_plane_c<float,repair_mode23_cpp_32<false>>,
  process_plane_c<float,repair_mode24_cpp_32<false>>,
  repair_do_nothing,
  process_plane_c<float,repair_mode26_cpp_32>,
  process_plane_c<float,repair_mode27_cpp_32>,
  process_plane_c<float,repair_mode28_cpp_32>
};

static RepairPlaneProcessor* c_functions_32_chroma[] = {
  repair_do_nothing,
  repair_copy_plane,
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
  process_plane_c<float,repair_mode19_cpp_32<true>>,
  process_plane_c<float,repair_mode20_cpp_32<true>>,
  process_plane_c<float,repair_mode21_cpp_32<true>>,
  process_plane_c<float,repair_mode22_cpp_32<true>>,
  process_plane_c<float,repair_mode23_cpp_32<true>>,
  process_plane_c<float,repair_mode24_cpp_32<true>>,
  repair_do_nothing,
  process_plane_c<float,repair_mode26_cpp_32>,
  process_plane_c<float,repair_mode27_cpp_32>,
  process_plane_c<float,repair_mode28_cpp_32>
};

#ifdef INTEL_INTRINSICS
// repair_sse.cpp
extern RepairPlaneProcessor* repair_sse4_functions[];
extern RepairPlaneProcessor* repair_sse2_functions[];
extern RepairPlaneProcessor* repair_sse4_functions_16_10[];
extern RepairPlaneProcessor* repair_sse4_functions_16_12[];
extern RepairPlaneProcessor* repair_sse4_functions_16_14[];
extern RepairPlaneProcessor* repair_sse4_functions_16_16[];
extern RepairPlaneProcessor* repair_sse4_functions_32[];
extern RepairPlaneProcessor* repair_sse4_functions_32_chroma[];
#endif

Repair::Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
  : GenericVideoFilter(child), ref_(ref), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr), functions_chroma(nullptr) {

  auto refVi = ref_->GetVideoInfo();

  if (!(vi.IsPlanar() || skip_cs_check)) {
    env->ThrowError("Repair works only with planar colorspaces");
  }

  if (vi.width != refVi.width || vi.height != refVi.height) {
    env->ThrowError("Clips should be of the same size!");
  }

  if (mode <= UNDEFINED_MODE || mode_ > 28 || modeU_ > 28 || modeV_ > 28 ||
    mode_ == 25 || modeU_ == 25 || modeV_ == 25 // mode 25 exists only in RemoveGrain
    ) {
    env->ThrowError("Repair mode should be between -1 and 28, except 25!");
  }

  bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
    env->ThrowError("Repair: cannot specify U or V mode for planar RGB!");
  }

  //now change undefined mode value and EVERYTHING WILL BREAK
  if (modeU_ <= UNDEFINED_MODE) {
    modeU_ = mode_;
  }
  if (modeV_ <= UNDEFINED_MODE) {
    modeV_ = modeU_;
  }

  // luma (or the RGB planes) is always processed: same bit depth and color family in any case
  if (vi.BitsPerComponent() != refVi.BitsPerComponent() || vi.IsRGB() != refVi.IsRGB()) {
    env->ThrowError("Both clips should have the same colorspace!");
  }
  // processed chroma: same subsampling as well
  if (vi.IsPlanar() && !vi.IsY() && (modeU_ != -1 || modeV_ != -1)) {
    if (!vi.IsSameColorspace(refVi)) {
      env->ThrowError("Both clips should have the same colorspace!");
    }
  }

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  has_at_least_v8 = true;
  try { env->CheckVersion(8); }
  catch (const AvisynthError&) { has_at_least_v8 = false; }

  int worst_case_width = vi.width;
  if (vi.IsYUV() && vi.NumComponents() >= 3)
    worst_case_width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);

  // 0: auto
  // 1: c
  // 2: sse2
  // 3: sse4.1
  // 4: avx2
#ifdef INTEL_INTRINSICS
  //const bool use_avx2 = (opt == 0 || opt >= 4) && !!(env->GetCPUFlags() & CPUF_AVX2);
  const bool use_sse41 = (opt == 0 || opt >= 3) && !!(env->GetCPUFlags() & CPUF_SSE4_1);
  const bool use_sse2 = (opt == 0 || opt >= 2) && !!(env->GetCPUFlags() & CPUF_SSE2);
#endif

  // no avx2 here

  if (pixelsize == 1) {
    functions = c_functions;
#ifdef INTEL_INTRINSICS
    if (worst_case_width >= 17) { // enough for XMM
      if (use_sse41)
        functions = repair_sse4_functions;
      else if (use_sse2)
        functions = repair_sse2_functions;
    }
#endif
  }
  else if (pixelsize == 2) {
    switch (bits_per_pixel) {
    case 10: functions = c_functions_10; break;
    case 12: functions = c_functions_12; break;
    case 14: functions = c_functions_14; break;
    case 16: functions = c_functions_16; break;
    default: env->ThrowError("Illegal bit-depth: %d!", bits_per_pixel);
    }
#ifdef INTEL_INTRINSICS
    if (use_sse41 && worst_case_width >= (16 / sizeof(uint16_t) + 1)) {
      switch (bits_per_pixel) {
      case 10: functions = repair_sse4_functions_16_10; break;
      case 12: functions = repair_sse4_functions_16_12; break;
      case 14: functions = repair_sse4_functions_16_14; break;
      case 16: functions = repair_sse4_functions_16_16; break;
      }
    }
#endif
  }
  else {// if (pixelsize == 4) 
    functions = c_functions_32;
    functions_chroma = c_functions_32_chroma;
#ifdef INTEL_INTRINSICS
    if (use_sse41 && worst_case_width >= (16 / sizeof(float) + 1)) {
      functions = repair_sse4_functions_32;
      functions_chroma = repair_sse4_functions_32_chroma;
    }
#endif
  }
}


PVideoFrame Repair::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto refFrame = ref_->GetFrame(n, env);
  auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {

    for (int p = 0; p < 3; ++p) {
      const int plane = planes[p];

      functions[mode_ + 1](env, dstFrame->GetWritePtr(plane), srcFrame->GetReadPtr(plane), refFrame->GetReadPtr(plane),
        dstFrame->GetPitch(plane), srcFrame->GetPitch(plane), refFrame->GetPitch(plane),
        srcFrame->GetRowSize(plane), srcFrame->GetHeight(plane));
    }
  }
  else {
    if(!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_Y)) || !is_16byte_aligned(refFrame->GetReadPtr(PLANAR_Y)))
      env->ThrowError("Repair: Invalid memory alignment. Unaligned crop?");

    functions[mode_ + 1](env, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), refFrame->GetReadPtr(PLANAR_Y),
      dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), refFrame->GetPitch(PLANAR_Y),
      srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
      if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_U)) || !is_16byte_aligned(refFrame->GetReadPtr(PLANAR_U)))
        env->ThrowError("Repair: Invalid memory alignment. Unaligned crop?");

      if (functions_chroma != nullptr) {
        // for float
        functions_chroma[modeU_ + 1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

        functions_chroma[modeV_ + 1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
      }
      else {
        functions[modeU_ + 1](env, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), refFrame->GetReadPtr(PLANAR_U),
          dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U), refFrame->GetPitch(PLANAR_U),
          srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U));

        functions[modeV_ + 1](env, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), refFrame->GetReadPtr(PLANAR_V),
          dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V), refFrame->GetPitch(PLANAR_V),
          srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V));
      }
    }
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
      args[MODEU].AsInt(Repair::UNDEFINED_MODE), 
      args[MODEV].AsInt(Repair::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), // not used, compatibility parameter
      args[OPT].AsInt(0),
      env);
}
