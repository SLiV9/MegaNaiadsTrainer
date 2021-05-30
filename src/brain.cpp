#include "brain.hpp"

#include <random>

#include "module.hpp"


// We are not backpropagating, so no need for gradient calculation.
static torch::NoGradGuard no_grad;

static size_t _brainSerialNumber = 0;

Brain::Brain(char p, std::shared_ptr<Module> module) :
	_module(module),
	personality(p),
	serialNumber(++_brainSerialNumber)
{
	if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

Brain::Brain(char personality) :
	Brain(personality, std::make_shared<Module>())
{}

void Brain::reset(size_t seat)
{
	size_t n = numGamesPerSeat[seat] * NUM_VIEW_SETS * NUM_CARDS;
	viewBufferPerSeat[seat].resize(n, 0);
	viewTensorPerSeat[seat] = torch::zeros(
			{int(numGamesPerSeat[seat]), int(NUM_VIEW_SETS * NUM_CARDS)},
			torch::kFloat);
	if (ENABLE_CUDA)
	{
		viewTensorPerSeat[seat] = viewTensorPerSeat[seat].contiguous().to(
			torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
	}
}

void Brain::evaluate(size_t seat)
{
	assert(seat < NUM_SEATS);
	_module->forward(viewTensorPerSeat[seat], outputTensorPerSeat[seat]);
}

void Brain::cycle(size_t seat)
{
	torch::Tensor bufferTensor = torch::from_blob(
		viewBufferPerSeat[seat].data(),
		{
			int(viewBufferPerSeat[seat].size() / (NUM_VIEW_SETS * NUM_CARDS)),
			int(NUM_VIEW_SETS * NUM_CARDS)
		},
		torch::kFloat);
	if (ENABLE_CUDA)
	{
		viewTensorPerSeat[seat] = bufferTensor.contiguous().to(
			torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
	}
	else
	{
		viewTensorPerSeat[seat] = bufferTensor;
	}
}

Brain Brain::clone()
{
	return Brain(personality,
		std::make_shared<Module>(*_module));
}

void Brain::mutate(float deviationFactor)
{
	std::vector<torch::Tensor>& myParams = _module->parameters();

	std::vector<uint8_t> yesOrNo(myParams.size(), false);
	std::random_device rd;
	std::mt19937 rng(rd());
	for (size_t i = 1; i < yesOrNo.size(); i += 2)
	{
		yesOrNo[i] = true;
	}
	std::shuffle(yesOrNo.begin(), yesOrNo.end(), rng);

	for (size_t i = 0; i < myParams.size(); i++)
	{
		if (yesOrNo[i])
		{
			torch::Tensor& param = myParams[i];
			// Take the standard normal deviation.
			torch::Tensor mutationTensor = torch::randn(param.sizes(),
				torch::TensorOptions().device(param.device()).dtype(param.dtype()));
			// Scale it down to the deviationFactor.
			mutationTensor.mul_(deviationFactor);
			// Add that to the existing parameter.
			param.add_(mutationTensor);
		}
	}
}

void Brain::spliceWith(const Brain& other)
{
	std::vector<torch::Tensor>& myParams = _module->parameters();

	std::vector<uint8_t> yesOrNo(myParams.size(), false);
	std::random_device rd;
	std::mt19937 rng(rd());
	for (size_t i = 1; i < yesOrNo.size(); i += 2)
	{
		yesOrNo[i] = true;
	}
	std::shuffle(yesOrNo.begin(), yesOrNo.end(), rng);

	const std::vector<torch::Tensor>& otherParams = other._module->parameters();
	for (size_t i = 0; i < myParams.size() && i < otherParams.size(); i++)
	{
		if (yesOrNo[i])
		{
			// Copy the parameter of the other module in its entirity.
			myParams[i].copy_(otherParams[i], /*non_blocking=*/true);
		}
	}
}
