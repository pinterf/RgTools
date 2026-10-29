#ifndef __CLENSE_H__
#define __CLENSE_H__

// Clense, ForwardClense, BackwardClense: AviSynth front end

#include "common.h"
#include "rg_avs.h"
#include "clense_core.h"

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
    ClenseMode mode_;
    bool reduceflicker_;

    ClenseCore core_;
    bool has_at_least_v8;

    // for reduceflicker
    PVideoFrame lastDstFrame;
    int lastRequestedFrameNo;
};


AVSValue __cdecl Create_Clense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_ForwardClense(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_BackwardClense(AVSValue args, void*, IScriptEnvironment* env);

#endif