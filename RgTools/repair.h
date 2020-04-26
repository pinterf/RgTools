#ifndef __REPAIR_H__
#define __REPAIR_H__

#include "common.h"


typedef void (RepairPlaneProcessor)(IScriptEnvironment* env, BYTE* pDst, const BYTE* pSrc, const BYTE* pRef, int dstPitch, int srcPitch, int refPitch, int rowsize, int height);


class Repair : public GenericVideoFilter {
public:
    Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

    const static int UNDEFINED_MODE = -2;

private:
    PClip ref_;

    int mode_;
    int modeU_;
    int modeV_;

    int pixelsize;
    int bits_per_pixel;

    bool has_at_least_v8;

    RepairPlaneProcessor **functions;
    RepairPlaneProcessor** functions_chroma; // for float
};


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env);

#endif