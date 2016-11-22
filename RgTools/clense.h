#ifndef __CLENSE_H__
#define __CLENSE_H__

#include "common.h"

template<typename pixel_t>
using CModeProcessor = pixel_t (*)(pixel_t, pixel_t, pixel_t);

enum class ClenseMode {
    FORWARD,
    BACKWARD,
    BOTH
};

class Clense : public GenericVideoFilter {



public:
    Clense(PClip child, PClip previous, PClip next, bool grey, ClenseMode mode, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

private:
    PClip previous_;
    PClip next_;
    bool grey_;
    bool sse2_;
    bool sse4_;
    ClenseMode mode_;

    int pixelsize;
    int bits_per_pixel;

    typedef void (ClenseProcessor)(Byte* pDst, const Byte *pSrc, const Byte* pRef1, const Byte* pRef2, int dstPitch, int srcPitch, int ref1Pitch, int ref2Pitch, int width, int height, IScriptEnvironment *env);

    ClenseProcessor* processor_;
};


AVSValue __cdecl Create_Clense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_ForwardClense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_BackwardClense(AVSValue args, void*, IScriptEnvironment* env);

#endif