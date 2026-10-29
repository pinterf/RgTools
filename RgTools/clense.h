#ifndef __CLENSE_H__
#define __CLENSE_H__

#include "common.h"

template<typename pixel_t>
using CModeProcessor = pixel_t (*)(pixel_t, pixel_t, pixel_t, int); // src, ref1, ref2, max_pixel_value (10-16 bit)

enum class ClenseMode {
    FORWARD,
    BACKWARD,
    BOTH
};

typedef void (ClenseProcessor)(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int width, int height, int bits_per_pixel, IScriptEnvironment *env);

#ifdef INTEL_INTRINSICS
// clense_sse.cpp
ClenseProcessor* clense_sse_processor(ClenseMode mode, int pixelsize, bool chroma);
#endif

class Clense : public GenericVideoFilter {

public:
    Clense(PClip child, PClip previous, PClip next, bool grey, bool reduceflicker, ClenseMode mode, bool skip_cs_check, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      // depends on working mode
      return cachehints == CACHE_GET_MTMODE ? (reduceflicker_ ? MT_MULTI_INSTANCE : MT_NICE_FILTER) : 0;
    }

private:
    PClip previous_;
    PClip next_;
    bool grey_;
    bool sse2_;
    bool sse4_;
    ClenseMode mode_;
    bool reduceflicker_;

    int pixelsize;
    int bits_per_pixel;

    bool has_at_least_v8;

    // for reduceflicker
    PVideoFrame lastDstFrame;
    int lastRequestedFrameNo;

    ClenseProcessor* processor_;
    ClenseProcessor* processor_chroma_; // for 32 bit
};


AVSValue __cdecl Create_Clense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_ForwardClense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_BackwardClense(AVSValue args, void*, IScriptEnvironment* env);

#endif