#include "trainer.hpp"

#include <algorithm>
#include <random>

#include <torch/torch.h>

#include "const.hpp"
#include "brain.hpp"


struct Game
{
	std::array<std::shared_ptr<Brain>, 4> brains;
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
	size_t skip = 100;

    std::random_device rd;
    std::mt19937 rng(rd());

	std::vector<Game> games;
	games.reserve(_brains.size() * _brains.size()
		* _brains.size() / (1 + skip) * _brains.size() / (1 + skip));

	for (size_t i = 0; i < _brains.size(); i++)
	{
		for (size_t j = i + 1; j < _brains.size(); j++)
		{
			for (size_t u = j + 1 + (rand() % skip);
				u < _brains.size();
				u += 1 + (rand() % skip))
			{
				for (size_t v = u + 1 + (rand() % skip);
					v < _brains.size();
					v += 1 + (rand() % skip))
				{
					games.emplace_back();
					Game& game = games.back();
					game.brains[0] = _brains[i];
					game.brains[1] = _brains[j];
					game.brains[2] = _brains[u];
					game.brains[3] = _brains[v];
					std::shuffle(game.brains.begin(), game.brains.end(), rng);
				}
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

	if (_brains.size() == 0)
	{
		for (size_t i = 0; i < 1000; i++)
		{
			_brains.push_back(std::make_shared<Brain>());
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
