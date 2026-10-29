#ifndef __VERTICAL_CLEANER_H__
#define __VERTICAL_CLEANER_H__

// VerticalCleaner: AviSynth front end

#include "common.h"
#include "rg_avs.h"
#include "vertical_cleaner_core.h"

class VerticalCleaner : public GenericVideoFilter {
public:
    VerticalCleaner(PClip child, int mode, int modeU, int modeV, bool skip_cs_check, IScriptEnvironment* env);

    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

    int __stdcall SetCacheHints(int cachehints, int frame_range) override {
      return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }

private:
    VerticalCleanerCore core_;
    bool has_at_least_v8;
};


AVSValue __cdecl Create_VerticalCleaner(AVSValue args, void*, IScriptEnvironment* env);

#endif
