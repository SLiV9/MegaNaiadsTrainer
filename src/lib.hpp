#pragma once

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__ ((visibility ("default")))
#endif

class Module;

extern "C"
{
    EXPORT Module* module_allocate();
    EXPORT void module_deallocate(Module* module);
    EXPORT void module_load(Module* module, const char* filepath);
    EXPORT void module_evaluate(Module* module, const float* input,
        int* wantsToPass, int* wantsToSwap, int* tableCard, int* ownCard);
}
