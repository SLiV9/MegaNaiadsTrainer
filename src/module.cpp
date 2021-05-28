#include "module.hpp"

#include "const.hpp"


#define ACTION_SIZE NUM_CARDS
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
		ACTION_SIZE * 2,
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

// We prevent calling the forward functions of the underlying modules so we can
// declare this function const and thus guarantee it is thread-safe.
torch::Tensor Module::forward(torch::Tensor& s) const
{
	s = torch::relu(torch::linear(s, _fc1->weight, _fc1->bias));
	s = torch::relu(torch::linear(s, _fc2->weight, _fc2->bias));
	s = torch::relu(torch::linear(s, _fc3->weight, _fc3->bias));
	s = torch::relu(torch::linear(s, _fc4->weight, _fc4->bias));

	torch::Tensor pi = torch::linear(s, _fc5->weight, _fc5->bias);

	return torch::sigmoid(pi);
}