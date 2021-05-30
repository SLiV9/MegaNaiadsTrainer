#include "module.hpp"

#include "const.hpp"


#define INNER_SIZE 500

Module::Module() :
	_fc1(register_module("fc1", torch::nn::Linear(
		NUM_VIEW_SETS * NUM_CARDS,
		INNER_SIZE))),
	_fc2(register_module("fc2", torch::nn::Linear(
		INNER_SIZE,
		INNER_SIZE))),
	_fc3(register_module("fc3", torch::nn::Linear(
		INNER_SIZE,
		INNER_SIZE))),
	_fc4(register_module("fc4", torch::nn::Linear(
		INNER_SIZE,
		INNER_SIZE))),
	_fc5(register_module("fc5", torch::nn::Linear(
		INNER_SIZE,
		ACTION_SIZE)))
{
	// We do not use the module's training mode for evolutionary training.
	eval();
}

Module::Module(Module&& other) :
	_fc1(register_module("fc1", std::move(other._fc1))),
	_fc2(register_module("fc2", std::move(other._fc2))),
	_fc3(register_module("fc3", std::move(other._fc3))),
	_fc4(register_module("fc4", std::move(other._fc4))),
	_fc5(register_module("fc5", std::move(other._fc5)))
{}

Module& Module::operator=(Module&& other)
{
	if (this != &other)
	{
		_fc1 = register_module("fc1", std::move(other._fc1));
		_fc2 = register_module("fc2", std::move(other._fc2));
		_fc3 = register_module("fc3", std::move(other._fc3));
		_fc4 = register_module("fc4", std::move(other._fc4));
		_fc5 = register_module("fc5", std::move(other._fc5));
	}
	return *this;
}

void Module::forward(const torch::Tensor& input, torch::Tensor& output) const
{
	torch::Tensor s;
	s = torch::relu(torch::linear(input, _fc1->weight, _fc1->bias));
	s = torch::relu(torch::linear(s, _fc2->weight, _fc2->bias));
	s = torch::relu(torch::linear(s, _fc3->weight, _fc3->bias));
	s = torch::relu(torch::linear(s, _fc4->weight, _fc4->bias));
	s = torch::sigmoid(torch::linear(s, _fc5->weight, _fc5->bias));
	output = s.to(torch::kCPU, torch::kFloat, /*non_blocking=*/true);
}
