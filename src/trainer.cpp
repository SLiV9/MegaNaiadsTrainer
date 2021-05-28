#include "trainer.hpp"

#include <algorithm>
#include <random>

#include <torch/torch.h>

#include "const.hpp"
#include "brain.hpp"


struct Player
{
	std::shared_ptr<Brain> brain;
	size_t relativeGameOffset;
};

struct Game
{
	std::array<Player, NUM_SEATS> players;
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
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				_brainsPerPersonality[p][i]->numGamesPerSeat[s] = 0;
			}
		}
	}

	std::vector<Game> games;
	size_t numGamesPerBrain = 1000;
	games.resize(NUM_BRAINS_PER_PERSONALITY * numGamesPerBrain);

	for (Game& game : games)
	{
		assert(NUM_PERSONALITIES == NUM_SEATS);
		for (size_t p = 0; p < NUM_PERSONALITIES; p++)
		{
			size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
			game.players[p].brain = _brainsPerPersonality[p][i];
		}
		std::shuffle(game.players.begin(), game.players.end(), rng);
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			game.players[s].relativeGameOffset =
				game.players[s].brain->numGamesPerSeat[s];
			game.players[s].brain->numGamesPerSeat[s] += 1;
		}
	}

	auto gameStateTensor = torch::zeros(
		games.size() * NUM_STATE_SETS * NUM_CARDS,
		torch::TensorOptions().dtype(torch::kFloat));
	if (ENABLE_CUDA)
	{
		gameStateTensor = gameStateTensor.contiguous().to(torch::kCUDA,
			torch::kHalf);
	}

	std::cout << "Playing " << games.size() << " games..." << std::endl;
	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				size_t n = brain->numGamesPerSeat[s];

				if (s == 0)
				{
					std::cout << char('A' + p) << i << ""
						" plays";
				}
				else
				{
					std::cout << ",";
				}
				std::cout << ""
					" " << n << " games"
					" from seat " << s << "";
				if (s + 1 == NUM_SEATS)
				{
					std::cout << std::endl;
				}

				auto& viewTensor = brain->viewTensorPerSeat[s];
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
	}

	size_t maxTurnsPerPlayer = 20;
	for (size_t t = 0; t < maxTurnsPerPlayer; t++)
	{
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			std::cout << "Evaluating turn " << (t * NUM_SEATS + s) << ""
				" (seat " << s << ")"
				"..." << std::endl;

			// Let all brains evaluate their positions.
			for (size_t p = 0; p < NUM_PERSONALITIES; p++)
			{
				for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
				{
					_brainsPerPersonality[p][i]->evaluate(s);
				}
			}

			// Use the results to change the game state.
			// TODO
		}
	}

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
			_brainsPerPersonality[p][i] = std::make_shared<Brain>();
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
