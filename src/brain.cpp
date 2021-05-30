#include "brain.hpp"

#include "module.hpp"


Brain::Brain() :
	_module(new Module())
{
	if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

void Brain::reset(size_t seat)
{
	size_t n = numGamesPerSeat[seat] * NUM_VIEW_SETS * NUM_CARDS;
	viewBufferPerSeat[seat].resize(n, 0);
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
		viewTensorPerSeat[seat] = bufferTensor.to(
			torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
	}
}
