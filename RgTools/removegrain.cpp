#include "removegrain.h"
#include "rg_functions_c.h"


template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_plane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst8, dstPitch, pSrc8, srcPitch, rowsize, 1);

    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_halfplane_c(IScriptEnvironment* env, const BYTE* pSrc8, BYTE* pDst8, int rowsize, int height, int srcPitch, int dstPitch) {
    pixel_t *pDst = reinterpret_cast<pixel_t *>(pDst8);
    const pixel_t *pSrc = reinterpret_cast<const pixel_t *>(pSrc8);

    dstPitch /= sizeof(pixel_t);
    const int srcPitchOrig = srcPitch;
    srcPitch /= sizeof(pixel_t);

    const int width = rowsize / sizeof(pixel_t);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no round
        for (int x = 1; x < width-1; x+=1) {
            pixel_t result = processor((uint8_t *)(pSrc + x), srcPitchOrig);
            pDst[x] = result;
        }
        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + (sizeof(pixel_t)==4 ? 0 : 1)) / 2; // float: no +1 rounding
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt((uint8_t *)pDst, dstPitch*sizeof(pixel_t), (uint8_t *)pSrc, srcPitch*sizeof(pixel_t), rowsize, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_even_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 2); //copy first two lines

    process_halfplane_c<pixel_t, processor>(env, pSrc+srcPitch, pDst+dstPitch, rowsize, height, srcPitch, dstPitch);
    if (height & 1) // odd height: the last row is an even row without a row below it: border, copied
        env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1);
}

template<typename pixel_t, CModeProcessor<pixel_t> processor>
static void process_odd_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, 1); //top border

    // height + 1: with an odd height the last odd row (height-2) has both neighbours and is processed too
    process_halfplane_c<pixel_t, processor>(env, pSrc, pDst, rowsize, height + 1, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, rowsize, 1); //bottom border
}

void rg_do_nothing(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {

}

void rg_copy_plane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, rowsize, height);
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

static PlaneProcessor* c_functions_10[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<10>>,
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<10>>,
  process_plane_c<uint16_t, rg_mode26_cpp_16>,
  process_plane_c<uint16_t, rg_mode27_cpp_16>,
  process_plane_c<uint16_t, rg_mode28_cpp_16>,
};

static PlaneProcessor* c_functions_12[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<12>>,
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<12>>,
  process_plane_c<uint16_t, rg_mode26_cpp_16>,
  process_plane_c<uint16_t, rg_mode27_cpp_16>,
  process_plane_c<uint16_t, rg_mode28_cpp_16>,
};

static PlaneProcessor* c_functions_14[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<14>>,
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<14>>,
  process_plane_c<uint16_t, rg_mode26_cpp_16>,
  process_plane_c<uint16_t, rg_mode27_cpp_16>,
  process_plane_c<uint16_t, rg_mode28_cpp_16>,
};


static PlaneProcessor* c_functions_16[] = {
  rg_do_nothing,
  rg_copy_plane,
  process_plane_c<uint16_t, rg_mode1_cpp_16>,
  process_plane_c<uint16_t, rg_mode2_cpp_16>,
  process_plane_c<uint16_t, rg_mode3_cpp_16>,
  process_plane_c<uint16_t, rg_mode4_cpp_16>,
  process_plane_c<uint16_t, rg_mode5_cpp_16>,
  process_plane_c<uint16_t, rg_mode6_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode7_cpp_16>,
  process_plane_c<uint16_t, rg_mode8_cpp_16<16>>,
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
  process_plane_c<uint16_t, rg_mode23_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode24_cpp_16<16>>,
  process_plane_c<uint16_t, rg_mode25_cpp_16<16>>,
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
extern PlaneProcessor* rg_sse4_functions_16_10[];
extern PlaneProcessor* rg_sse4_functions_16_12[];
extern PlaneProcessor* rg_sse4_functions_16_14[];
extern PlaneProcessor* rg_sse4_functions_16_16[];
extern PlaneProcessor* rg_sse4_functions_32_luma[];
extern PlaneProcessor* rg_sse4_functions_32_chroma[];
// removegrain_avx2.cpp
extern PlaneProcessor* rg_avx2_functions[];
extern PlaneProcessor* rg_avx2_functions_16_10[];
extern PlaneProcessor* rg_avx2_functions_16_12[];
extern PlaneProcessor* rg_avx2_functions_16_14[];
extern PlaneProcessor* rg_avx2_functions_16_16[];
extern PlaneProcessor* rg_avx2_functions_32_luma[];
extern PlaneProcessor* rg_avx2_functions_32_chroma[];
#endif

RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr), functions_chroma(nullptr) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("RemoveGrain works only with planar colorspaces");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 28 || modeU_ > 28 || modeV_ > 28) {
        env->ThrowError("RemoveGrain mode should be between -1 and 28!");
    }

    bool isPlanarRGB = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    if (isPlanarRGB && ((modeU_ > UNDEFINED_MODE) || (modeV_ > UNDEFINED_MODE))) {
      env->ThrowError("RemoveGrain: cannot specify U or V mode for planar RGB!");
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    //now change undefined mode value and EVERYTHING WILL BREAK
    if (modeU_ <= UNDEFINED_MODE) { 
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    int worst_case_width = vi.width;
    if (vi.IsYUV() && vi.NumComponents() >= 3)
      worst_case_width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);

    pixelsize = vi.ComponentSize();
    bits_per_pixel = vi.BitsPerComponent();

    // 0: auto
    // 1: c
    // 2: sse2
    // 3: sse4.1
    // 4: avx2
#ifdef INTEL_INTRINSICS
    const bool use_avx2 = (opt == 0 || opt >= 4) && !!(env->GetCPUFlags() & CPUF_AVX2);
    const bool use_sse41 = (opt == 0 || opt >= 3) && !!(env->GetCPUFlags() & CPUF_SSE4_1);
    const bool use_sse2 = (opt == 0 || opt >= 2) && !!(env->GetCPUFlags() & CPUF_SSE2);
#endif

    if (pixelsize == 1) {
      functions = c_functions;
#ifdef INTEL_INTRINSICS
      if (use_avx2)
        functions = rg_avx2_functions;
      else if (use_sse41)
        functions = rg_sse4_functions;
      else if (use_sse2)
        functions = rg_sse2_functions;

      if (worst_case_width < 32 + 1 && use_avx2) { //not enough for YMM, try SSE4
        functions = rg_sse4_functions;
      }
      if (worst_case_width < 16+1) { //not enough for XMM
        functions = c_functions;
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
      if (use_avx2 && worst_case_width >= (32 / sizeof(uint16_t) + 1)) {
        // mode 6 and 8 bitdepth clamp specific
        switch (bits_per_pixel) {
        case 10: functions = rg_avx2_functions_16_10; break;
        case 12: functions = rg_avx2_functions_16_12; break;
        case 14: functions = rg_avx2_functions_16_14; break;
        case 16: functions = rg_avx2_functions_16_16; break;
        }
      }
      else if (use_sse41 && worst_case_width >= (16/sizeof(uint16_t) + 1)) {
        // mode 6 and 8 bitdepth clamp specific
        switch (bits_per_pixel) {
        case 10: functions = rg_sse4_functions_16_10; break;
        case 12: functions = rg_sse4_functions_16_12; break;
        case 14: functions = rg_sse4_functions_16_14; break;
        case 16: functions = rg_sse4_functions_16_16; break;
        }
      }
#endif
    }
    else {// if (pixelsize == 4) 
      functions = c_functions_32_luma;
      functions_chroma = c_functions_32_chroma;
#ifdef INTEL_INTRINSICS
      if (use_avx2 && worst_case_width >= (32 / sizeof(float) + 1)) {
        functions = rg_avx2_functions_32_luma;
        functions_chroma = rg_avx2_functions_32_chroma;
      }
      else if (use_sse41 && worst_case_width >= (16 / sizeof(float) + 1)) {
        functions = rg_sse4_functions_32_luma;
        functions_chroma = rg_sse4_functions_32_chroma;
      }
#endif
    }
}


// A plane of less than 3 rows has border rows only: it is copied (the plane functions need an inner row).
static void process_plane(PlaneProcessor* f, int mode, IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch) {
  if (mode > 0 && height < 3)
    f = rg_copy_plane;
  f(env, pSrc, pDst, rowsize, height, srcPitch, dstPitch);
}

PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
  auto srcFrame = child->GetFrame(n, env);
  auto dstFrame = has_at_least_v8 ? env->NewVideoFrameP(vi, &srcFrame) : env->NewVideoFrame(vi);

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int* planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;

  // remark: no special alignment required for AVX2

  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
    for (int p = 0; p < 3; ++p) {
      const int plane = planes[p];

      process_plane(functions[mode_ + 1], mode_, env, srcFrame->GetReadPtr(plane), dstFrame->GetWritePtr(plane), srcFrame->GetRowSize(plane),
        srcFrame->GetHeight(plane), srcFrame->GetPitch(plane), dstFrame->GetPitch(plane));
    }
  }
  else {
    if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_Y)))
      env->ThrowError("RemoveGrain: Invalid memory alignment. Unaligned crop?");

    process_plane(functions[mode_ + 1], mode_, env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y),
      srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
      if (!is_16byte_aligned(srcFrame->GetReadPtr(PLANAR_U)))
        env->ThrowError("RemoveGrain: Invalid memory alignment. Unaligned crop?");

      if (functions_chroma != nullptr) {
        // for float
        process_plane(functions_chroma[modeU_ + 1], modeU_, env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));
        process_plane(functions_chroma[modeV_ + 1], modeV_, env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
      else {
        process_plane(functions[modeU_ + 1], modeU_, env, srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U),
          srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));

        process_plane(functions[modeV_ + 1], modeV_, env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V),
          srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
      }
    }
  }
  if (vi.IsYUVA() || vi.IsPlanarRGBA())
  { // copy alpha
    env->BitBlt(dstFrame->GetWritePtr(PLANAR_A), dstFrame->GetPitch(PLANAR_A), srcFrame->GetReadPtr(PLANAR_A), srcFrame->GetPitch(PLANAR_A), srcFrame->GetRowSize(PLANAR_A_ALIGNED), srcFrame->GetHeight(PLANAR_A));
  }
  return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR, OPT };
    AVSValue retval = new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrain::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrain::UNDEFINED_MODE), 
      args[PLANAR].AsBool(false), args[OPT].AsInt(0), env);
    return retval;
}
