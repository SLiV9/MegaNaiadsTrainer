#include "module.hpp"

#include "const.hpp"

#include <random>


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

torch::Tensor Module::forward(const torch::Tensor& input) const
{
	torch::Tensor s;
	s = torch::relu(torch::linear(input, _fc1->weight, _fc1->bias));
	s = torch::relu(torch::linear(s, _fc2->weight, _fc2->bias));
	s = torch::relu(torch::linear(s, _fc3->weight, _fc3->bias));
	s = torch::relu(torch::linear(s, _fc4->weight, _fc4->bias));
	s = torch::sigmoid(torch::linear(s, _fc5->weight, _fc5->bias));
	return s;
}

void Module::mutate(double deviationFactor)
{
	std::vector<torch::Tensor>& myParams = parameters();
	for (size_t i = 0; i < myParams.size(); i++)
	{
		torch::Tensor& param = myParams[i];
		// Take the standard normal deviation.
		torch::Tensor mutationTensor = torch::randn(param.sizes(),
			torch::TensorOptions().device(param.device()).dtype(param.dtype()));
		// Set half the values to 0.
		torch::Tensor selectionTensor = torch::randint(0, 2, param.sizes(),
			torch::TensorOptions().device(param.device()).dtype(torch::kBool));
		mutationTensor.mul_(selectionTensor);
		// Scale it down to the deviationFactor.
		mutationTensor.mul_(deviationFactor);
		// Add that to the existing parameter.
		param.add_(mutationTensor);
	}
}

void Module::spliceWith(const Module& other)
{
	std::vector<torch::Tensor>& myParams = parameters();
	const std::vector<torch::Tensor>& otherParams = other.parameters();
	for (size_t i = 0; i < myParams.size() && i < otherParams.size(); i++)
	{
		torch::Tensor& param = myParams[i];
		// Select half the weights of this parameter.
		torch::Tensor selectionTensor = torch::randint(0, 2, param.sizes(),
			torch::TensorOptions().device(param.device()).dtype(torch::kBool));
		// Set the selected weights to zero, as they will be replaced.
		param.mul_(selectionTensor);
		// Invert the selection.
		selectionTensor.logical_not_();
		// Take those weights from the other module.
		selectionTensor = selectionTensor.to(param.dtype());
		selectionTensor.mul_(otherParams[i]);
		// Add the mutations.
		param.add_(selectionTensor);
	}
}
