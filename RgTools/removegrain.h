#ifndef __REMOVEGRAIN_H__
#define __REMOVEGRAIN_H__

#include "common.h"


typedef void (PlaneProcessor)(IScriptEnvironment* env, const BYTE* pSrc, BYTE* pDst, int rowsize, int height, int srcPitch, int dstPitch);


class RemoveGrain : public GenericVideoFilter {
public:
    RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

    const static int UNDEFINED_MODE = -2;

private:
    int mode_;
    int modeU_;
    int modeV_;

    int pixelsize;
    int bits_per_pixel;

    PlaneProcessor **functions;
};


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env);

#endif