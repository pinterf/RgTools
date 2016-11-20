#include "vertical_cleaner.h"
#include <xutility>


static void vcleaner_median_sse2(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;

    for (int y = 1; y < height-1; ++y) {
        for (int x = 0; x < width; x+=16) {
            __m128i up     = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x - srcPitch));
            __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
            __m128i down   = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x + srcPitch));

            __m128i mi = _mm_min_epu8(up, down);
            __m128i ma = _mm_max_epu8(up, down);

            __m128i cma = _mm_max_epu8(mi, center);
            __m128i dst = _mm_min_epu8(cma, ma);

            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}

static void vcleaner_relaxed_median_sse2(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    for (int y = 2; y < height-2; ++y) {
        for (int x = 0; x < width; x+=16) {
            __m128i p2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch*2));
            __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x-srcPitch));
            __m128i c  = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x));
            __m128i n1 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch));
            __m128i n2 = _mm_load_si128(reinterpret_cast<const __m128i*>(pSrc + x+srcPitch*2));

            __m128i pdiff = _mm_subs_epu8(p1, p2);
            __m128i ndiff = _mm_subs_epu8(n1, n2);

            __m128i pt = _mm_adds_epu8(pdiff, p1);
            __m128i nt = _mm_adds_epu8(ndiff, n1);

            __m128i upper = _mm_min_epu8(pt, nt);
            upper = _mm_max_epu8(upper, p1);
            upper = _mm_max_epu8(upper, n1);

            pdiff = _mm_subs_epu8(p2, p1);
            ndiff = _mm_subs_epu8(n2, n1);

            pt = _mm_subs_epu8(p1, pdiff);
            nt = _mm_subs_epu8(n1, ndiff);

            __m128i minpn1 = _mm_min_epu8(p1, n1);

            __m128i lower = _mm_max_epu8(pt, nt);
            lower = _mm_min_epu8(lower, minpn1);

            __m128i dst = simd_clip(c, lower, upper);

            _mm_store_si128(reinterpret_cast<__m128i*>(pDst+x), dst);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);
}



static void vcleaner_median_c(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);

    pSrc += srcPitch;
    pDst += dstPitch;

    for (int y = 1; y < height-1; ++y) {
        for (int x = 0; x < width; x+=1) {
            BYTE up = pSrc[x-srcPitch];
            BYTE center = pSrc[x];
            BYTE down = pSrc[x+srcPitch];
            pDst[x] = std::min(std::max(std::min(up, down), center), std::max(up, down));
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 1);
}

static __forceinline Byte satb(int value) {
    return clip(value, 0, 255);
}

static void vcleaner_relaxed_median_c(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);

    pSrc += srcPitch*2;
    pDst += dstPitch*2;

    for (int y = 2; y < height-2; ++y) {
        for (int x = 0; x < width; x+=1) {
            BYTE p2 = pSrc[x-srcPitch*2];
            BYTE p1 = pSrc[x-srcPitch];
            BYTE c = pSrc[x];
            BYTE n1 = pSrc[x+srcPitch];
            BYTE n2 = pSrc[x+srcPitch*2];

            Byte upper = std::max(std::max(std::min(satb(satb(p1-p2) + p1), satb(satb(n1-n2) + n1)), p1), n1);
            Byte lower = std::min(std::min(p1, n1), std::max(satb(p1 - satb(p2-p1)), satb(n1 - satb(n2-n1))));

            pDst[x] = clip(c, lower, upper);
        }

        pSrc += srcPitch;
        pDst += dstPitch;
    }

    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, 2);
}

static void copy_plane(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    env->BitBlt(pDst, dstPitch, pSrc, srcPitch, width, height);
}

static void do_nothing(Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {

}

VCleanerProcessor* sse2_functions[] = {
    do_nothing,
    copy_plane,
    vcleaner_median_sse2,
    vcleaner_relaxed_median_sse2
};

VCleanerProcessor* c_functions[] = {
    do_nothing,
    copy_plane,
    vcleaner_median_c,
    vcleaner_relaxed_median_c
};

static void dispatch_median(int mode, Byte* pDst, const Byte *pSrc, int dstPitch, int srcPitch, int width, int height, IScriptEnvironment *env) {
    if ((env->GetCPUFlags() & CPUF_SSE2) && width >= 16 && is_16byte_aligned(pSrc)) {
        sse2_functions[mode+1](pDst, pSrc, dstPitch, srcPitch, width, height, env);
    } else {
        c_functions[mode+1](pDst, pSrc, dstPitch, srcPitch, width, height, env);
    }
}

VerticalCleaner::VerticalCleaner(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env)
: GenericVideoFilter(child), mode_(mode), modeU_(modeU), modeV_(modeV) {
    if (!(vi.IsPlanar() || skip_cs_check)) {
        env->ThrowError("VerticalCleaner works only with planar colorspaces");
    }

    if (mode_ > 2 || modeU_ > 2 || modeV_ > 2) {
        env->ThrowError("Sorry, this mode does not exist");
    }

    if (modeU_ <= UNDEFINED_MODE) {
        modeU_ = mode_;
    }
    if (modeV_ <= UNDEFINED_MODE) {
        modeV_ = modeU_;
    }
}

PVideoFrame VerticalCleaner::GetFrame(int n, IScriptEnvironment* env) {
    auto srcFrame = child->GetFrame(n, env);
    auto dstFrame = env->NewVideoFrame(vi);

    dispatch_median(mode_, dstFrame->GetWritePtr(PLANAR_Y), srcFrame->GetReadPtr(PLANAR_Y), dstFrame->GetPitch(PLANAR_Y), srcFrame->GetPitch(PLANAR_Y),
        srcFrame->GetRowSize(PLANAR_Y), srcFrame->GetHeight(PLANAR_Y), env);

    if (!vi.IsY()) {
        dispatch_median(modeU_, dstFrame->GetWritePtr(PLANAR_U), srcFrame->GetReadPtr(PLANAR_U), dstFrame->GetPitch(PLANAR_U), srcFrame->GetPitch(PLANAR_U),
            srcFrame->GetRowSize(PLANAR_U), srcFrame->GetHeight(PLANAR_U), env);

        dispatch_median(modeV_, dstFrame->GetWritePtr(PLANAR_V), srcFrame->GetReadPtr(PLANAR_V), dstFrame->GetPitch(PLANAR_V), srcFrame->GetPitch(PLANAR_V),
            srcFrame->GetRowSize(PLANAR_V), srcFrame->GetHeight(PLANAR_V), env);
    }
    return dstFrame;
}

AVSValue __cdecl Create_VerticalCleaner(AVSValue args, void*, IScriptEnvironment* env) {
    enum { CLIP, MODE, MODEU, MODEV, PLANAR };
    return new VerticalCleaner(
        args[CLIP].AsClip(), 
        args[MODE].AsInt(1),
        args[MODEU].AsInt(VerticalCleaner::UNDEFINED_MODE),
        args[MODEV].AsInt(VerticalCleaner::UNDEFINED_MODE),
        args[PLANAR].AsBool(false), 
        env);
}

