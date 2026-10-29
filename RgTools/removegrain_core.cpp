// RemoveGrain core: C plane loops, function tables, kernel selection

#include "removegrain_core.h"
#include "common.h"
#include "rg_functions_c.h"


template<typename pixel_t, auto processor>
static void process_plane_c(const uint8_t* pSrc8, uint8_t* pDst8, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
    rg_copy_rows(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    // 10-16 bit kernels that depend on the max pixel value get it as a 3rd parameter
    constexpr bool with_max = std::is_invocable_v<decltype(processor), const uint8_t*, int, int>;
    const int pixel_max = sizeof(pixel_t) == 2 ? (1 << bits_per_pixel) - 1 : 0;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result;
            if constexpr (with_max) result = processor((uint8_t *)(pSrc + x), srcPitchOrig, pixel_max);
            else result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    rg_copy_rows((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, auto processor>
static void process_halfplane_c(const uint8_t* pSrc8, uint8_t* pDst8, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    // 10-16 bit kernels that depend on the max pixel value get it as a 3rd parameter
    constexpr bool with_max = std::is_invocable_v<decltype(processor), const uint8_t*, int, int>;
    const int pixel_max = sizeof(pixel_t) == 2 ? (1 << bits_per_pixel) - 1 : 0;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no round
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result;
            if constexpr (with_max) result = processor((uint8_t *)(pSrc + x), srcPitchOrig, pixel_max);
            else result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        rg_copy_rows((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, auto processor>
static void process_even_rows_c(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
    rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_c<pixel_t, processor>(pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch, bits_per_pixel);
    if (height & 1) // odd height: the last row is an even row without a row below it: border, copied
        rg_copy_rows(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1);
}

template<typename pixel_t, auto processor>
static void process_odd_rows_c(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
    rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    // height + 1: with an odd height the last odd row (height-2) has both neighbours and is processed too
    process_halfplane_c<pixel_t, processor>(pSrc, pDst, rowsize, height + 1, srcPitch, dstPitch, bits_per_pixel);

    rg_copy_rows(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

void rg_do_nothing(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {

}

void rg_copy_plane(const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
    rg_copy_rows(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
}


static PlaneProcessor* c_functions[] = {
    rg_do_nothing,
    rg_copy_plane,
    process_plane_c<uint8_t, rg_mode1_cpp>,
    process_plane_c<uint8_t, rg_mode2_cpp>,
    process_plane_c<uint8_t, rg_mode3_cpp>,
    process_plane_c<uint8_t, rg_mode4_cpp>,
    process_plane_c<uint8_t, rg_mode5_cpp>,
    process_plane_c<uint8_t, rg_mode6_cpp>,
    process_plane_c<uint8_t, rg_mode7_cpp>,
    process_plane_c<uint8_t, rg_mode8_cpp>,
    process_plane_c<uint8_t, rg_mode9_cpp>,
    process_plane_c<uint8_t, rg_mode10_cpp>,
    process_plane_c<uint8_t, rg_mode11_cpp>,
    process_plane_c<uint8_t, rg_mode12_cpp>,
    process_even_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_odd_rows_c<uint8_t, rg_mode13_and14_cpp>,
    process_even_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_odd_rows_c<uint8_t, rg_mode15_and16_cpp>,
    process_plane_c<uint8_t, rg_mode17_cpp>,
    process_plane_c<uint8_t, rg_mode18_cpp>,
    process_plane_c<uint8_t, rg_mode19_cpp>,
    process_plane_c<uint8_t, rg_mode20_cpp>,
    process_plane_c<uint8_t, rg_mode21_cpp>,
    process_plane_c<uint8_t, rg_mode22_cpp>,
    process_plane_c<uint8_t, rg_mode23_cpp>,
    process_plane_c<uint8_t, rg_mode24_cpp>,
    process_plane_c<uint8_t, rg_mode25_cpp>,
    process_plane_c<uint8_t, rg_mode26_cpp>,
    process_plane_c<uint8_t, rg_mode27_cpp>,
    process_plane_c<uint8_t, rg_mode28_cpp>,
};


// 10-16 bit (the max pixel value comes from bits_per_pixel)
static PlaneProcessor* c_functions_16[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16>,
  process_plane_c<uint16_t, rg_mode9_cpp_16>,
  process_plane_c<uint16_t, rg_mode10_cpp_16>,
  process_plane_c<uint16_t, rg_mode11_cpp_16>,
  process_plane_c<uint16_t, rg_mode12_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode13_and14_cpp_16>,
  process_even_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_odd_rows_c<uint16_t, rg_mode15_and16_cpp_16>,
  process_plane_c<uint16_t, rg_mode17_cpp_16>,
  process_plane_c<uint16_t, rg_mode18_cpp_16>,
  process_plane_c<uint16_t, rg_mode19_cpp_16>,
  process_plane_c<uint16_t, rg_mode20_cpp_16>,
  process_plane_c<uint16_t, rg_mode21_cpp_16>,
  process_plane_c<uint16_t, rg_mode22_cpp_16>,
  process_plane_c<uint16_t, rg_mode23_cpp_16>,
  process_plane_c<uint16_t, rg_mode24_cpp_16>,
  process_plane_c<uint16_t, rg_mode25_cpp_16>,
  process_plane_c<uint16_t, rg_mode26_cpp_16>,
  process_plane_c<uint16_t, rg_mode27_cpp_16>,
  process_plane_c<uint16_t, rg_mode28_cpp_16>,
};

static PlaneProcessor* c_functions_32_luma[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<float, rg_mode1_cpp_32>,
  process_plane_c<float, rg_mode2_cpp_32>,
  process_plane_c<float, rg_mode3_cpp_32>,
  process_plane_c<float, rg_mode4_cpp_32>,
  process_plane_c<float, rg_mode5_cpp_32>,
  process_plane_c<float, rg_mode6_cpp_32>,
  process_plane_c<float, rg_mode7_cpp_32>,
  process_plane_c<float, rg_mode8_cpp_32>,
  process_plane_c<float, rg_mode9_cpp_32>,
  process_plane_c<float, rg_mode10_cpp_32>,
  process_plane_c<float, rg_mode11_cpp_32>,
  process_plane_c<float, rg_mode12_cpp_32>,
  process_even_rows_c<float, rg_mode13_and14_cpp_32>,
  process_odd_rows_c<float, rg_mode13_and14_cpp_32>,
  process_even_rows_c<float, rg_mode15_and16_cpp_32>,
  process_odd_rows_c<float, rg_mode15_and16_cpp_32>,
  process_plane_c<float, rg_mode17_cpp_32>,
  process_plane_c<float, rg_mode18_cpp_32>,
  process_plane_c<float, rg_mode19_cpp_32>,
  process_plane_c<float, rg_mode20_cpp_32>,
  process_plane_c<float, rg_mode21_cpp_32>,
  process_plane_c<float, rg_mode22_cpp_32>,
  process_plane_c<float, rg_mode23_cpp_32<false>>,
  process_plane_c<float, rg_mode24_cpp_32<false>>,
  process_plane_c<float, rg_mode25_cpp_32<false>>, // false: luma, true: chroma
  process_plane_c<float, rg_mode26_cpp_32>,
  process_plane_c<float, rg_mode27_cpp_32>,
  process_plane_c<float, rg_mode28_cpp_32>,
};

static PlaneProcessor* c_functions_32_chroma[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<float, rg_mode1_cpp_32>,
  process_plane_c<float, rg_mode2_cpp_32>,
  process_plane_c<float, rg_mode3_cpp_32>,
  process_plane_c<float, rg_mode4_cpp_32>,
  process_plane_c<float, rg_mode5_cpp_32>,
  process_plane_c<float, rg_mode6_cpp_32>,
  process_plane_c<float, rg_mode7_cpp_32>,
  process_plane_c<float, rg_mode8_cpp_32>,
  process_plane_c<float, rg_mode9_cpp_32>,
  process_plane_c<float, rg_mode10_cpp_32>,
  process_plane_c<float, rg_mode11_cpp_32>,
  process_plane_c<float, rg_mode12_cpp_32>,
  process_even_rows_c<float, rg_mode13_and14_cpp_32>,
  process_odd_rows_c<float, rg_mode13_and14_cpp_32>,
  process_even_rows_c<float, rg_mode15_and16_cpp_32>,
  process_odd_rows_c<float, rg_mode15_and16_cpp_32>,
  process_plane_c<float, rg_mode17_cpp_32>,
  process_plane_c<float, rg_mode18_cpp_32>,
  process_plane_c<float, rg_mode19_cpp_32>,
  process_plane_c<float, rg_mode20_cpp_32>,
  process_plane_c<float, rg_mode21_cpp_32>,
  process_plane_c<float, rg_mode22_cpp_32>,
  process_plane_c<float, rg_mode23_cpp_32<true>>,
  process_plane_c<float, rg_mode24_cpp_32<true>>,
  process_plane_c<float, rg_mode25_cpp_32<true>>, // false: luma, true: chroma
  process_plane_c<float, rg_mode26_cpp_32>,
  process_plane_c<float, rg_mode27_cpp_32>,
  process_plane_c<float, rg_mode28_cpp_32>,
};


#ifdef INTEL_INTRINSICS
// removegrain_sse.cpp
extern PlaneProcessor* rg_sse2_functions[];
extern PlaneProcessor* rg_sse4_functions[];
extern PlaneProcessor* rg_sse4_functions_16[];
extern PlaneProcessor* rg_sse4_functions_32_luma[];
extern PlaneProcessor* rg_sse4_functions_32_chroma[];
// removegrain_avx2.cpp
extern PlaneProcessor* rg_avx2_functions[];
extern PlaneProcessor* rg_avx2_functions_16[];
extern PlaneProcessor* rg_avx2_functions_32_luma[];
extern PlaneProcessor* rg_avx2_functions_32_chroma[];
#endif
#ifdef NEON_INTRINSICS
// removegrain_neon.cpp
extern PlaneProcessor* rg_neon_functions[];
extern PlaneProcessor* rg_neon_functions_16[];
extern PlaneProcessor* rg_neon_functions_32_luma[];
extern PlaneProcessor* rg_neon_functions_32_chroma[];
#endif

// A plane of less than 3 rows has border rows only: it is copied (the plane functions need an inner row).
static void process_plane_or_copy(PlaneProcessor* f, int mode, const uint8_t* pSrc, uint8_t* pDst, int rowsize, int height, int srcPitch, int dstPitch, int bits_per_pixel) {
  if (mode > 0 && height < 3)
    f = rg_copy_plane;
  f(pSrc, pDst, rowsize, height, srcPitch, dstPitch, bits_per_pixel);
}

const char* RemoveGrainCore::init(const RemoveGrainParams& params, const RgFormat& format, int cpu_flags) {
    (void)cpu_flags; // C-only builds
    int mode = params.mode;
    int modeU = params.modeU;
    int modeV = params.modeV;
    const int opt = params.opt;
    (void)opt; // C-only builds

    if (mode <= UNDEFINED_MODE || mode > 28 || modeU > 28 || modeV > 28)
        return "RemoveGrain mode should be between -1 and 28!";

    if (format.family == RgColorFamily::RGB && (modeU > UNDEFINED_MODE || modeV > UNDEFINED_MODE))
        return "RemoveGrain: cannot specify U or V mode for planar RGB!";

    if (modeU <= UNDEFINED_MODE)
        modeU = mode;
    if (modeV <= UNDEFINED_MODE)
        modeV = modeU;

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
    const bool use_avx2 = (opt == 0 || opt >= 4) && !!(cpu_flags & RG_CPU_AVX2);
    const bool use_sse41 = (opt == 0 || opt >= 3) && !!(cpu_flags & RG_CPU_SSE41);
    const bool use_sse2 = (opt == 0 || opt >= 2) && !!(cpu_flags & RG_CPU_SSE2);
#endif

    if (format.pixelsize == 1) {
      functions_ = c_functions;
#ifdef INTEL_INTRINSICS
      if (use_avx2)
        functions_ = rg_avx2_functions;
      else if (use_sse41)
        functions_ = rg_sse4_functions;
      else if (use_sse2)
        functions_ = rg_sse2_functions;

      if (worst_case_width < 32 + 1 && use_avx2) { //not enough for YMM, try SSE4
        functions_ = rg_sse4_functions;
      }
      if (worst_case_width < 16+1) { //not enough for XMM
        functions_ = c_functions;
      }
#endif
#ifdef NEON_INTRINSICS
      if (use_neon && worst_case_width >= 16 + 1)
        functions_ = rg_neon_functions;
#endif
    }
    else if (format.pixelsize == 2) {
      functions_ = c_functions_16;
#ifdef INTEL_INTRINSICS
      if (use_avx2 && worst_case_width >= (32 / sizeof(uint16_t) + 1)) {
        functions_ = rg_avx2_functions_16;
      }
      else if (use_sse41 && worst_case_width >= (16/sizeof(uint16_t) + 1)) {
        functions_ = rg_sse4_functions_16;
      }
#endif
#ifdef NEON_INTRINSICS
      if (use_neon && worst_case_width >= 8 + 1)
        functions_ = rg_neon_functions_16;
#endif
    }
    else {// if (format.pixelsize == 4) 
      functions_ = c_functions_32_luma;
      functions_chroma_ = c_functions_32_chroma;
#ifdef INTEL_INTRINSICS
      if (use_avx2 && worst_case_width >= (32 / sizeof(float) + 1)) {
        functions_ = rg_avx2_functions_32_luma;
        functions_chroma_ = rg_avx2_functions_32_chroma;
      }
      else if (use_sse41 && worst_case_width >= (16 / sizeof(float) + 1)) {
        functions_ = rg_sse4_functions_32_luma;
        functions_chroma_ = rg_sse4_functions_32_chroma;
      }
#endif
#ifdef NEON_INTRINSICS
      if (use_neon && worst_case_width >= 4 + 1) {
        functions_ = rg_neon_functions_32_luma;
        functions_chroma_ = rg_neon_functions_32_chroma;
      }
#endif
    }
    return nullptr;
}

void RemoveGrainCore::process_plane(int plane_index, const RgPlane& plane) const {
    const bool chroma = format_.family == RgColorFamily::YUV && plane_index > 0;
    const int mode = mode_[plane_index];
    PlaneProcessor** functions = (chroma && functions_chroma_ != nullptr) ? functions_chroma_ : functions_;
    process_plane_or_copy(functions[mode + 1], mode, plane.src, plane.dst, plane.rowsize, plane.height, plane.src_pitch, plane.dst_pitch,
      format_.bits_per_pixel);
}
