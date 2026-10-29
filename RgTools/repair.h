#ifndef __REPAIR_H__
#define __REPAIR_H__

// Repair: AviSynth front end

#include "common.h"
#include "rg_avs.h"
#include "repair_core.h"


class Repair : public GenericVideoFilter {
public:
    Repair(PClip child, PClip ref, int mode, int modeU, int modeV, bool skip_cs_check, int opt, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

private:
    PClip ref_;
    RepairCore core_;
    bool has_at_least_v8;
};


AVSValue __cdecl Create_Repair(AVSValue args, void*, IScriptEnvironment* env);

#endif