// Repair core: C plane loop, function tables, kernel selection

#include "repair_core.h"
#include "common.h"
#include "repair_functions_c.h"

template<typename pixel_t, auto processor>
static void process_plane_c(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); 

  const int width = rowsize / sizeof(pixel_t);
  const int max_pixel_value = sizeof(pixel_t) == 2 ? (1 << bits_per_pixel) - 1 : 0;

  pSrc += srcPitch;
  pDst += dstPitch;
  pRef += refPitch;
  for (int y = 1; y < height-1; ++y) {
    reinterpret_cast<pixel_t *>(pDst)[0] = reinterpret_cast<const pixel_t *>(pSrc)[0];
    for (int x = 1; x < width-1; x+=1) {
      pixel_t result;
      if constexpr (std::is_invocable_v<decltype(processor), const uint8_t*, pixel_t, int, int>)
        result = processor(pRef + x*sizeof(pixel_t), reinterpret_cast<const pixel_t *>(pSrc)[x], refPitch, max_pixel_value);
      else
        result = processor(pRef + x*sizeof(pixel_t), reinterpret_cast<const pixel_t *>(pSrc)[x], refPitch);
      reinterpret_cast<pixel_t *>(pDst)[x] = result;
    }
    reinterpret_cast<pixel_t *>(pDst)[width-1] = reinterpret_cast<const pixel_t *>(pSrc)[width-1];

    pSrc += srcPitch;
    pDst += dstPitch;
    pRef += refPitch;
  }

  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1);
}


void repair_do_nothing(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int /*bits_per_pixel*/) {

}

void repair_copy_plane(uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int /*bits_per_pixel*/) {
  rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
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

// 10-16 bit (the max pixel value comes from bits_per_pixel)
static RepairPlaneProcessor* c_functions_16[] = {
  repair_do_nothing,
  repair_copy_plane,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode2_cpp_16>,
  process_plane_c<uint16_t,repair_mode3_cpp_16>,
  process_plane_c<uint16_t,repair_mode4_cpp_16>,
  process_plane_c<uint16_t,repair_mode5_cpp_16>, 
  process_plane_c<uint16_t,repair_mode6_cpp_16>, 
  process_plane_c<uint16_t,repair_mode7_cpp_16>, 
  process_plane_c<uint16_t,repair_mode8_cpp_16>, 
  process_plane_c<uint16_t,repair_mode9_cpp_16>, 
  process_plane_c<uint16_t,repair_mode10_cpp_16>,
  process_plane_c<uint16_t,repair_mode1_cpp_16>,
  process_plane_c<uint16_t,repair_mode12_cpp_16>,
  process_plane_c<uint16_t,repair_mode13_cpp_16>,
  process_plane_c<uint16_t,repair_mode14_cpp_16>,
  process_plane_c<uint16_t,repair_mode15_cpp_16>,
  process_plane_c<uint16_t,repair_mode16_cpp_16>,
  process_plane_c<uint16_t,repair_mode17_cpp_16>,
  process_plane_c<uint16_t,repair_mode18_cpp_16>,
  process_plane_c<uint16_t,repair_mode19_cpp_16>, 
  process_plane_c<uint16_t,repair_mode20_cpp_16>, 
  process_plane_c<uint16_t,repair_mode21_cpp_16>, 
  process_plane_c<uint16_t,repair_mode22_cpp_16>, 
  process_plane_c<uint16_t,repair_mode23_cpp_16>, 
  process_plane_c<uint16_t,repair_mode24_cpp_16>,
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
extern RepairPlaneProcessor* repair_sse4_functions_16[];
extern RepairPlaneProcessor* repair_sse4_functions_32[];
extern RepairPlaneProcessor* repair_sse4_functions_32_chroma[];
#endif
#ifdef NEON_INTRINSICS
// repair_neon.cpp
extern RepairPlaneProcessor* repair_neon_functions[];
extern RepairPlaneProcessor* repair_neon_functions_16[];
extern RepairPlaneProcessor* repair_neon_functions_32[];
extern RepairPlaneProcessor* repair_neon_functions_32_chroma[];
#endif

// A plane of less than 3 rows has border rows only: it is copied (the plane functions need an inner row).
static void process_plane_or_copy(RepairPlaneProcessor* f, int mode, uint8_t* pDst, const uint8_t* pSrc, const uint8_t* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height, int bits_per_pixel) {
  if (mode > 0 && height < 3)
    f = repair_copy_plane;
  f(pDst, pSrc, pRef, dstPitch, srcPitch, refPitch, rowsize, height, bits_per_pixel);
}

const char* RepairCore::init(const RepairParams& params, const RgFormat& format, const RgFormat& ref_format, int cpu_flags) {
  (void)cpu_flags; // C-only builds
  int mode = params.mode;
  int modeU = params.modeU;
  int modeV = params.modeV;
  const int opt = params.opt;
  (void)opt; // C-only builds

  if (format.width != ref_format.width || format.height != ref_format.height)
    return "Clips should be of the same size!";

  if (mode <= UNDEFINED_MODE || mode > 28 || modeU > 28 || modeV > 28 ||
    mode == 25 || modeU == 25 || modeV == 25 // mode 25 exists only in RemoveGrain
    )
    return "Repair mode should be between -1 and 28, except 25!";

  if (format.family == RgColorFamily::RGB && (modeU > UNDEFINED_MODE || modeV > UNDEFINED_MODE))
    return "Repair: cannot specify U or V mode for planar RGB!";

  if (modeU <= UNDEFINED_MODE)
    modeU = mode;
  if (modeV <= UNDEFINED_MODE)
    modeV = modeU;

  // luma (or the RGB planes) is always processed: same bit depth and color family in any case
  if (format.bits_per_pixel != ref_format.bits_per_pixel || (format.family == RgColorFamily::RGB) != (ref_format.family == RgColorFamily::RGB))
    return "Both clips should have the same colorspace!";
  // processed chroma: same subsampling as well
  if (format.family != RgColorFamily::Gray && (modeU != -1 || modeV != -1)) {
    if (format.family != ref_format.family || format.ssw != ref_format.ssw || format.ssh != ref_format.ssh || format.has_alpha != ref_format.has_alpha)
      return "Both clips should have the same colorspace!";
  }

  mode_[0] = mode;
  mode_[1] = modeU;
  mode_[2] = modeV;
  format_ = format;
  functions_chroma_ = nullptr;

  // 0: auto
  // 1: c
  // 2: sse2
  // 3: sse4.1
  // 4: avx2
  // aarch64: 1: c, 0 and the others: NEON
#if defined(INTEL_INTRINSICS) || defined(NEON_INTRINSICS)
  // the narrowest plane (subsampled chroma) must be wider than one vector
  int worst_case_width = format.width;
  if (format.family == RgColorFamily::YUV)
    worst_case_width >>= format.ssw;
#endif
#ifdef NEON_INTRINSICS
  const bool use_neon = opt != 1 && !!(cpu_flags & RG_CPU_NEON);
#endif
#ifdef INTEL_INTRINSICS
  //const bool use_avx2 = (opt == 0 || opt >= 4) && !!(cpu_flags & RG_CPU_AVX2);
  const bool use_sse41 = (opt == 0 || opt >= 3) && !!(cpu_flags & RG_CPU_SSE41);
  const bool use_sse2 = (opt == 0 || opt >= 2) && !!(cpu_flags & RG_CPU_SSE2);
#endif

  // no avx2 here

  if (format.pixelsize == 1) {
    functions_ = c_functions;
#ifdef INTEL_INTRINSICS
    if (worst_case_width >= 17) { // enough for XMM
      if (use_sse41)
        functions_ = repair_sse4_functions;
      else if (use_sse2)
        functions_ = repair_sse2_functions;
    }
#endif
#ifdef NEON_INTRINSICS
    if (use_neon && worst_case_width >= 16 + 1)
      functions_ = repair_neon_functions;
#endif
  }
  else if (format.pixelsize == 2) {
    functions_ = c_functions_16;
#ifdef INTEL_INTRINSICS
    if (use_sse41 && worst_case_width >= (16 / sizeof(uint16_t) + 1)) {
      functions_ = repair_sse4_functions_16;
    }
#endif
#ifdef NEON_INTRINSICS
    if (use_neon && worst_case_width >= 8 + 1)
      functions_ = repair_neon_functions_16;
#endif
  }
  else {// if (format.pixelsize == 4) 
    functions_ = c_functions_32;
    functions_chroma_ = c_functions_32_chroma;
#ifdef INTEL_INTRINSICS
    if (use_sse41 && worst_case_width >= (16 / sizeof(float) + 1)) {
      functions_ = repair_sse4_functions_32;
      functions_chroma_ = repair_sse4_functions_32_chroma;
    }
#endif
#ifdef NEON_INTRINSICS
    if (use_neon && worst_case_width >= 4 + 1) {
      functions_ = repair_neon_functions_32;
      functions_chroma_ = repair_neon_functions_32_chroma;
    }
#endif
  }
  return nullptr;
}

void RepairCore::process_plane(int plane_index, const RgPlane& plane, const uint8_t* ref, int ref_pitch) const {
  const bool chroma = format_.family == RgColorFamily::YUV && plane_index > 0;
  const int mode = mode_[plane_index];
  RepairPlaneProcessor** functions = (chroma && functions_chroma_ != nullptr) ? functions_chroma_ : functions_;
  process_plane_or_copy(functions[mode + 1], mode, plane.dst, plane.src, ref, plane.dst_pitch, plane.src_pitch, ref_pitch,
    plane.rowsize, plane.height, format_.bits_per_pixel);
}
