#ifndef __REMOVEGRAIN_H__
#define __REMOVEGRAIN_H__

// RemoveGrain: AviSynth front end

#include "common.h"
#include "rg_avs.h"
#include "removegrain_core.h"


class RemoveGrain : public GenericVideoFilter {
public:
    RemoveGrain(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

private:
    RemoveGrainCore core_;
    bool has_at_least_v8;
};


AVSValue __cdecl Create_RemoveGrain(AVSValue args, void*, IScriptEnvironment* env);

#endif