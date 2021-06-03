#include "lib.hpp"

//#include "module.hpp"
//#include "stateloader.hpp"

class Module
{
    // Empty
};


extern "C" Module* module_allocate()
{
    return new Module();
}

extern "C" void module_deallocate(Module* module)
{
    delete module;
}

extern "C" void module_load(Module* module, const char* filepath)
{
    //load_state_dict(*module, filepath);
    //module->to(torch::kCPU, torch::kFloat);
}

extern "C" void module_evaluate(Module* module, const float* input,
    int* wantsToPass, int* wantsToSwap, int* tableCard, int* ownCard)
{
    // TODO inputTensor
}
