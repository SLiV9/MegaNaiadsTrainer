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

public:
	Brain();
	Brain(const Brain&) = delete;
	Brain(Brain&& other) = default;
	Brain& operator=(const Brain&) = delete;
	Brain& operator=(Brain&&) = default;
	~Brain() = default;

	void evaluate(size_t seat);
};