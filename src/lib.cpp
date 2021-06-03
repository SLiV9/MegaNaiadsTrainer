#include "lib.hpp"

#include "const.hpp"
#include "module.hpp"
#include "stateloader.hpp"


extern "C" Module* module_allocate()
{
	return new Module();
}

extern "C" void module_deallocate(Module* module)
{
	delete module;
}

extern "C" void module_load(Module* module, const char* filepath)
{
	load_state_dict(*module, filepath);
	module->to(torch::kCPU, torch::kFloat);
}

extern "C" void module_evaluate(Module* module, const float* input,
	int* wantsToPass, int* wantsToSwap, int* tableCard, int* ownCard)
{
	std::vector<size_t> tableCards;
	std::vector<size_t> ownCards;
	for (size_t i = 0; i < NUM_VIEW_SETS * NUM_CARDS; i++)
	{
		if (input[i] > 0)
		{
			tableCards.push_back(i);
		}
		else if (input[NUM_CARDS + i] > 0)
		{
			ownCards.push_back(i);
		}
	}

	torch::Tensor inputTensor = torch::from_blob(
		(void*) input,
		int(NUM_VIEW_SETS * NUM_CARDS),
		torch::kFloat);
	torch::Tensor outputTensor = torch::zeros(int(ACTION_SIZE), torch::kFloat);
	module->forward(inputTensor, outputTensor);

	const float* output = outputTensor.data_ptr<float>();
	float passWeight = output[2 * NUM_CARDS];
	float swapWeight = output[2 * NUM_CARDS + 1];
	float tableCardWeight = passWeight - 1;
	for (size_t c : tableCards)
	{
		if (output[c] > tableCardWeight)
		{
			*tableCard = (int) c;
			tableCardWeight = output[c];
		}
	}
	float ownCardWeight = passWeight - 1;
	for (size_t c : ownCards)
	{
		if (output[NUM_CARDS + c] > ownCardWeight)
		{
			*ownCard = (int) c;
			ownCardWeight = output[c];
		}
	}
	float playWeight = std::min(tableCardWeight, ownCardWeight);

	bool wantsToPlay = (playWeight > passWeight && playWeight > swapWeight);
	*wantsToSwap = (!wantsToPlay && swapWeight > passWeight);
	*wantsToPass = (!wantsToPlay && !wantsToSwap);
}
