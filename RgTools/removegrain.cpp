#include "rg_functions_c.h"
#include "rg_functions_sse.h"
#include "removegrain.h"


template<SseModeProcessor processor>
static void process_plane_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;
    int mod16_width = width / 16 * 16;

    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];

        for (int x = 1; x < mod16_width-1; x+=16) {
            __m128i result = processor(pSrc+x, srcPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod16_width != width) {
            __m128i result = processor(pSrc+width-17, srcPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+width-17), result);
        }

        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}


template<SseModeProcessor processor>
static void process_halfplane_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    pSrc += srcPitch;
    pDst += dstPitch;
    int mod16_width = width / 16 * 16;

    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + 1) / 2;
        for (int x = 1; x < mod16_width-1; x+=16) {
            __m128i result = processor(pSrc+x, srcPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+x), result);
        }

        if (mod16_width != width) {
            __m128i result = processor(pSrc+width-17, srcPitch);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pDst+width-17), result);
        }

        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + 1) / 2;
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<SseModeProcessor processor>
static void process_even_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2); //copy first two lines

    process_halfplane_sse<processor>(env, pSrc+srcPitch, pDst+dstPitch, width, height, srcPitch, dstPitch);
}

template<SseModeProcessor processor>
static void process_odd_rows_sse(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1); //top border

    process_halfplane_sse<processor>(env, pSrc, pDst, width, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, width, 1); //bottom border
}

template<CModeProcessor processor>
static void process_plane_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height-1; ++y) {
        pDst[0] = pSrc[0];
        for (int x = 1; x < width-1; x+=1) {
            BYTE result = processor(pSrc + x, srcPitch);
            pDst[x] = result;
        }
        pDst[width-1] = pSrc[width-1];

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}

template<CModeProcessor processor>
static void process_halfplane_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    pSrc += srcPitch;
    pDst += dstPitch;
    for (int y = 1; y < height/2; ++y) {
        pDst[0] = (pSrc[srcPitch] + pSrc[-srcPitch] + 1) / 2;
        for (int x = 1; x < width-1; x+=1) {
            BYTE result = processor(pSrc + x, srcPitch);
            pDst[x] = result;
        }
        pDst[width-1] = (pSrc[width-1 + srcPitch] + pSrc[width-1 - srcPitch] + 1) / 2;
        pSrc += srcPitch;
        pDst += dstPitch;

        env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1); //other field

        pSrc += srcPitch;
        pDst += dstPitch;
    }
}

template<CModeProcessor processor>
static void process_even_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2); //copy first two lines

    process_halfplane_c<processor>(env, pSrc+srcPitch, pDst+dstPitch, width, height, srcPitch, dstPitch);
}

template<CModeProcessor processor>
static void process_odd_rows_c(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1); //top border

    process_halfplane_c<processor>(env, pSrc, pDst, width, height, srcPitch, dstPitch);

    env->BitBlt(pDst+dstPitch*(height-1), dstPitch, pSrc+srcPitch*(height-1), srcPitch, width, 1); //bottom border
}

static void doNothing(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {

}

static void copyPlane(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int width, int height, int srcPitch, int dstPitch) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, height);
}


PlaneProcessor* sse2_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<rg_mode1_sse<SSE2>>,
    process_plane_sse<rg_mode2_sse<SSE2>>,
    process_plane_sse<rg_mode3_sse<SSE2>>,
    process_plane_sse<rg_mode4_sse<SSE2>>,
    process_plane_sse<rg_mode5_sse<SSE2>>,
    process_plane_sse<rg_mode6_sse<SSE2>>,
    process_plane_sse<rg_mode7_sse<SSE2>>,
    process_plane_sse<rg_mode8_sse<SSE2>>,
    process_plane_sse<rg_mode9_sse<SSE2>>,
    process_plane_sse<rg_mode10_sse<SSE2>>,
    process_plane_sse<rg_mode11_sse<SSE2>>,
    process_plane_sse<rg_mode12_sse<SSE2>>,
    process_even_rows_sse<rg_mode13_and14_sse<SSE2>>,
    process_odd_rows_sse<rg_mode13_and14_sse<SSE2>>,
    process_even_rows_sse<rg_mode15_and16_sse<SSE2>>,
    process_odd_rows_sse<rg_mode15_and16_sse<SSE2>>,
    process_plane_sse<rg_mode17_sse<SSE2>>,
    process_plane_sse<rg_mode18_sse<SSE2>>,
    process_plane_sse<rg_mode19_sse<SSE2>>,
    process_plane_sse<rg_mode20_sse<SSE2>>,
    process_plane_sse<rg_mode21_sse<SSE2>>,
    process_plane_sse<rg_mode22_sse<SSE2>>,
    process_plane_sse<rg_mode23_sse<SSE2>>,
    process_plane_sse<rg_mode24_sse<SSE2>>,
};

PlaneProcessor* sse3_functions[] = {
    doNothing,
    copyPlane,
    process_plane_sse<rg_mode1_sse<SSE3>>,
    process_plane_sse<rg_mode2_sse<SSE3>>,
    process_plane_sse<rg_mode3_sse<SSE3>>,
    process_plane_sse<rg_mode4_sse<SSE3>>,
    process_plane_sse<rg_mode5_sse<SSE3>>,
    process_plane_sse<rg_mode6_sse<SSE3>>,
    process_plane_sse<rg_mode7_sse<SSE3>>,
    process_plane_sse<rg_mode8_sse<SSE3>>,
    process_plane_sse<rg_mode9_sse<SSE3>>,
    process_plane_sse<rg_mode10_sse<SSE3>>,
    process_plane_sse<rg_mode11_sse<SSE3>>,
    process_plane_sse<rg_mode12_sse<SSE3>>,
    process_even_rows_sse<rg_mode13_and14_sse<SSE3>>,
    process_odd_rows_sse<rg_mode13_and14_sse<SSE3>>,
    process_even_rows_sse<rg_mode15_and16_sse<SSE3>>,
    process_odd_rows_sse<rg_mode15_and16_sse<SSE3>>,
    process_plane_sse<rg_mode17_sse<SSE3>>,
    process_plane_sse<rg_mode18_sse<SSE3>>,
    process_plane_sse<rg_mode19_sse<SSE3>>,
    process_plane_sse<rg_mode20_sse<SSE3>>,
    process_plane_sse<rg_mode21_sse<SSE3>>,
    process_plane_sse<rg_mode22_sse<SSE3>>,
    process_plane_sse<rg_mode23_sse<SSE3>>,
    process_plane_sse<rg_mode24_sse<SSE3>>,
};

PlaneProcessor* c_functions[] = {
    doNothing,
    copyPlane,
    process_plane_c<rg_mode1_cpp>,
    process_plane_c<rg_mode2_cpp>,
    process_plane_c<rg_mode3_cpp>,
    process_plane_c<rg_mode4_cpp>,
    process_plane_c<rg_mode5_cpp>,
    process_plane_c<rg_mode6_cpp>,
    process_plane_c<rg_mode7_cpp>,
    process_plane_c<rg_mode8_cpp>,
    process_plane_c<rg_mode9_cpp>,
    process_plane_c<rg_mode10_cpp>,
    process_plane_c<rg_mode11_cpp>,
    process_plane_c<rg_mode12_cpp>,
    process_even_rows_c<rg_mode13_and14_cpp>,
    process_odd_rows_c<rg_mode13_and14_cpp>,
    process_even_rows_c<rg_mode15_and16_cpp>,
    process_odd_rows_c<rg_mode15_and16_cpp>,
    process_plane_c<rg_mode17_cpp>,
    process_plane_c<rg_mode18_cpp>,
    process_plane_c<rg_mode19_cpp>,
    process_plane_c<rg_mode20_cpp>,
    process_plane_c<rg_mode21_cpp>,
    process_plane_c<rg_mode22_cpp>,
    process_plane_c<rg_mode23_cpp>,
    process_plane_c<rg_mode24_cpp>
};

RemoveGrain::RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
    : GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV), functions(nullptr) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("RemoveGrain works only with planar colorspaces");
    }

    if (mode <= UNDEFINED_MODE || mode_ > 24 || modeU_ > 24 || modeV_ > 24) {
        env->ThrowError("RemoveGrain mode should be between -1 and 24!");
    }

    //now change undefined mode value and EVERYTHING WILL BREAK
    if (modeU_ <= UNDEFINED_MODE) { 
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }

    functions = (env->GetCPUFlags() & CPUF_SSE3) ? sse3_functions 
        : (env->GetCPUFlags() & CPUF_SSE2) ? sse2_functions
        : c_functions;

    if (vi.width < 17) { //not enough for XMM
        functions = c_functions;
    }
}


PVideoFrame RemoveGrain::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);
    
    functions[mode_+1](env, srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetRowSize(PLANAR_Y), 
        srcFrame->GetHeight(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y));

    if (vi.IsPlanar() && !vi.IsY()) {
        functions[modeU_+1](env,srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetRowSize(PLANAR_U), 
            srcFrame->GetHeight(PLANAR_U), srcFrame->GetPitch(PLANAR_U), dstFrame->GetPitch(PLANAR_U));

        functions[modeV_+1](env, srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetRowSize(PLANAR_V), 
            srcFrame->GetHeight(PLANAR_V), srcFrame->GetPitch(PLANAR_V), dstFrame->GetPitch(PLANAR_V));
    }
    return dstFrame;
}


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR };
    return new RemoveGrain(args[CLIP].AsClip(), args[MODE].AsInt(1), args[MODEU].AsInt(RemoveGrain::UNDEFINED_MODE), args[MODEV].AsInt(RemoveGrain::UNDEFINED_MODE), args[PLANAR].AsBool(false), env);
}
