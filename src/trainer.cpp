#include "trainer.hpp"

#include <algorithm>
#include <random>

#include <torch/torch.h>

#include "const.hpp"
#include "brain.hpp"


struct Game
{
	std::array<Trainer::Player, 4> players;
};

Trainer::Trainer() :
	_startTime(std::time(nullptr)),
	_round(0)
{
	torch::set_num_threads(4);
}

void Trainer::playRound()
{
	auto start = std::chrono::high_resolution_clock::now();
	size_t count = 0;
    std::random_device rd;
    std::mt19937 rng(rd());

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			_brainsPerPersonality[p][i].relativeGameOffset = 0;
		}
	}

	std::vector<Game> games;
	size_t numGamesPerBrain = 1000;
	games.resize(NUM_BRAINS_PER_PERSONALITY * numGamesPerBrain);

	for (Game& game : games)
	{
		assert(NUM_PERSONALITIES == 4);
		for (size_t p = 0; p < NUM_PERSONALITIES; p++)
		{
			size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
			game.players[p] = _brainsPerPersonality[p][i];
			_brainsPerPersonality[p][i].relativeGameOffset += 1;
		}
		std::shuffle(game.players.begin(), game.players.end(), rng);
	}

	auto gameStateTensor = torch::zeros(
		games.size() * NUM_STATE_SETS * NUM_CARDS,
		torch::TensorOptions().dtype(torch::kFloat));
	if (ENABLE_CUDA)
	{
		gameStateTensor = gameStateTensor.contiguous().to(torch::kCUDA,
			torch::kHalf);
	}

	std::array<std::array<torch::Tensor, NUM_BRAINS_PER_PERSONALITY>,
		NUM_PERSONALITIES> gameViewsPerPersonality;

	std::cout << "Playing " << games.size() << " games..." << std::endl;
	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			size_t n = _brainsPerPersonality[p][i].relativeGameOffset;
			std::cout << char('A' + p) << i << ""
				" plays " << n << " games" << std::endl;

			auto& viewTensor = gameViewsPerPersonality[p][i];
			viewTensor = torch::zeros(
				n * NUM_VIEW_SETS * NUM_CARDS,
				torch::TensorOptions().dtype(torch::kFloat));
			if (ENABLE_CUDA)
			{
				viewTensor = viewTensor.contiguous().to(torch::kCUDA,
					torch::kHalf);
			}
		}
	}

	// TODO play the games

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Playing round took " << elapsed << "ms"
			" (" << (elapsed / games.size()) << "ms per game)"
			"" << std::endl;
	}
}

void Trainer::train()
{
	auto start = std::chrono::high_resolution_clock::now();

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			_brainsPerPersonality[p][i].brain = std::make_shared<Brain>();
		}
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Initializing brains took " << elapsed << "ms"
			"" << std::endl;
	}

	size_t numRounds = 1;
	for (; _round < numRounds; _round++)
	{
		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;

		playRound();

		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;
	}
}
