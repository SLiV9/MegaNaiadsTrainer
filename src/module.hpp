#pragma once

#include <torch/torch.h>


class Module : public torch::nn::Cloneable<Module>
{
private:
	friend class Brain;
	friend class TrainingBrain;

	torch::nn::Linear _fc1;
	torch::nn::Linear _fc2;
	torch::nn::Linear _fc3;
	torch::nn::Linear _fc4;
	torch::nn::Linear _fc5;

public:
	Module();
	Module(const Module&) = default;
	Module(Module&& other);
	Module& operator=(const Module&) = default;
	Module& operator=(Module&& other);
	~Module() = default;

	void reset() override
	{
		*this = Module();
	}

	torch::Tensor forward(const torch::Tensor& input) const;

	void mutate(double deviationFactor);
	void spliceWith(const Module& other);
};
