#include "removegrain.h"
#include "clense.h"
#include "repair.h"
#include "vertical_cleaner.h"
#include "RemoveGrainT.h"



const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;

    env->AddFunction("RemoveGrain", "c[mode]i[modeU]i[modeV]i[planar]b[optavx2]b", Create_RemoveGrain, 0);
    env->AddFunction("Repair", "cc[mode]i[modeU]i[modeV]i[planar]b", Create_Repair, 0);
    env->AddFunction("Clense", "c[previous]c[next]c[grey]b[reduceflicker]b[planar]b[cache]i", Create_Clense, 0);
    env->AddFunction("ForwardClense", "c[grey]b[planar]b[cache]i", Create_ForwardClense, 0);
    env->AddFunction("BackwardClense", "c[grey]b[planar]b[cache]i", Create_BackwardClense, 0);
    env->AddFunction("VerticalCleaner", "c[mode]i[modeU]i[modeV]i[planar]b", Create_VerticalCleaner, 0);
    env->AddFunction("TemporalRepair", "cc[mode]i[smooth]i[grey]b[planar]b[opt]i", Create_TemporalRepair, 0);
    return "Itai, onii-chan!";
}
