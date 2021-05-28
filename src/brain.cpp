#include "brain.hpp"

#include "module.hpp"


Brain::Brain() :
	_module(new Module())
{
	if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}
