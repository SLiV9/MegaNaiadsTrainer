#include "brain.hpp"

#include "module.hpp"


// We are not backpropagating, so no need for gradient calculation.
static torch::NoGradGuard no_grad;

static size_t _brainSerialNumber = 0;

Brain::Brain(char p, size_t mNum, size_t fNum,
		std::shared_ptr<Module> module) :
	_module(module),
	personality(p),
	serialNumber(++_brainSerialNumber),
	motherNumber(mNum),
	fatherNumber(fNum)
{
	if (!_module) {}
	else if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

Brain::Brain(char personality) :
	Brain(personality, 0, 0,
		(personality == 'D')
			? std::shared_ptr<Module>()
			: std::make_shared<Module>()
	)
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
	outputTensorPerSeat[seat] = torch::zeros(
			{int(numGamesPerSeat[seat]), int(ACTION_SIZE)},
			torch::kFloat);
}

void Brain::evaluate(size_t seat)
{
	if (!_module)
	{
		if (personality != 'D')
		{
			std::cerr << "missing module"
				" for " << personality << serialNumber << ""
				"" << std::endl;
		}
		return;
	}

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

Brain Brain::makeMutation(float deviationFactor) const
{
	if (!_module)
	{
		return Brain(personality);
	}
	auto newModule = std::dynamic_pointer_cast<Module>(_module->clone());
	newModule->mutate(deviationFactor);
	return Brain(personality, serialNumber, 0, newModule);
}

Brain Brain::makeOffspringWith(const Brain& other) const
{
	if (!_module)
	{
		return Brain(personality);
	}
	auto newModule = std::dynamic_pointer_cast<Module>(_module->clone());
	newModule->spliceWith(*(other._module));
	return Brain(personality, serialNumber, other.serialNumber, newModule);
}
