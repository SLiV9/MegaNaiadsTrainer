#pragma once

#include <queue>
#include <memory>

#include <torch/torch.h>

#include "const.hpp"

class Module;


class Brain
{
private:
	std::shared_ptr<Module> _module;

public:
	std::array<size_t, NUM_SEATS> numGamesPerSeat;
	std::array<torch::Tensor, NUM_SEATS> viewTensorPerSeat;
	std::array<torch::Tensor, NUM_SEATS> outputTensorPerSeat;
	std::array<std::vector<float>, NUM_SEATS> viewBufferPerSeat;
	int numLosses = 0;
	float totalScore = 0;

public:
	Brain();
	Brain(const Brain&) = delete;
	Brain(Brain&& other) = default;
	Brain& operator=(const Brain&) = delete;
	Brain& operator=(Brain&&) = default;
	~Brain() = default;

	void reset(size_t seat);
	void evaluate(size_t seat);
	void cycle(size_t seat);
};
