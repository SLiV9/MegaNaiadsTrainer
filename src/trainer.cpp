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

inline void debugPrintCard(size_t card)
{
	const char* SUITS[NUM_SUITS] = {"C", "D", "H", "S"};
	const char* FACES[NUM_FACES_PER_SUIT] = {"7", "8", "9", "10",
		"J", "Q", "K", "A"};
	std::cout << SUITS[card / NUM_FACES_PER_SUIT]
		<< FACES[card % NUM_FACES_PER_SUIT];
}

inline void debugPrintGameState(const Game& /*game*/,
	const torch::Tensor& stateTensor)
{
	torch::Tensor stateTensorCPU = stateTensor.to(torch::kCPU,
		torch::kFloat);
	const float* state = stateTensorCPU.data_ptr<float>();

	std::cout << "----------------------" << std::endl;
	std::cout << "Table: ";
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		if (state[c] > 0.5)
		{
			debugPrintCard(c);
			std::cout << " ";
		}
	}
	std::cout << std::endl;
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		std::cout << "Seat " << s << ": ";
		for (size_t c = 0; c < NUM_CARDS; c++)
		{
			if (state[(1 + s) * NUM_CARDS + c] > 0.5)
			{
				debugPrintCard(c);
				if (state[(1 + NUM_SEATS + s) * NUM_CARDS + c] > 0.5)
				{
					std::cout << "*";
				}
				std::cout << " ";
			}
		}
		std::cout << std::endl;
	}
	for (size_t i = 0; i < NUM_STATE_SETS * NUM_CARDS; i++)
	{
		if (i > 0 && i % NUM_CARDS == 0)
		{
			std::cout << std::endl;
		}
		std::cout << (0.001 * int(1000 * state[i])) << " ";
	}
	std::cout << std::endl;
	std::cout << "----------------------" << std::endl;
}

inline void assertCorrectGameState(const Game& game,
	const torch::Tensor& stateTensor)
{
	torch::Tensor stateTensorCPU = stateTensor.to(torch::kCPU,
		torch::kFloat);
	const float* state = stateTensorCPU.data_ptr<float>();

	size_t numUsed = 0;
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		bool isUsed = false;
		for (size_t hand = 0; hand < NUM_SEATS + 1; hand++)
		{
			if (state[hand * NUM_CARDS + c] > 0.5)
			{
				assert(!isUsed);
				isUsed = true;
				numUsed += 1;
			}
		}
	}
	assert(numUsed == NUM_CARDS_PER_HAND * (NUM_SEATS + 1));
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		auto& brain = game.players[s].brain;
		size_t offset = game.players[s].relativeGameOffset;
		auto& viewTensor = brain->viewTensorPerSeat[s][offset];
		torch::Tensor viewTensorCPU = viewTensor.to(torch::kCPU,
			torch::kFloat);
		const float* view = viewTensorCPU.data_ptr<float>();
		for (size_t c = 0; c < NUM_CARDS; c++)
		{
			for (size_t hand = 0; hand < NUM_SEATS + 1; hand++)
			{
				if (state[hand * NUM_CARDS + c] < 0.5)
				{
					size_t relhand = (hand > 0)
						? (1 + ((hand - 1 + NUM_SEATS - s) % NUM_SEATS))
						: 0;
					assert(view[relhand * NUM_CARDS + c] < 0.1);
				}
				continue;
				size_t relhand = (hand > 0)
					? (1 + ((hand - 1 + NUM_SEATS - s) % NUM_SEATS))
					: 0;
				if (relhand == 0 || relhand == 1)
				{
					assert(view[relhand * NUM_CARDS + c] > 0.9);
				}
				float vis = state[(hand + NUM_SEATS) * NUM_CARDS + c];
				if (relhand > 0)
				{
					if (vis > 0.5)
					{
						assert(view[relhand * NUM_CARDS + c] > 0.9);
					}
					else
					{
						assert(view[relhand * NUM_CARDS + c] < 0.1);
					}
				}
				if (relhand == 1)
				{
					if (vis > 0.5)
					{
						assert(view[(1 + NUM_SEATS) * NUM_CARDS + c] > 0.9);
					}
					else
					{
						assert(view[(1 + NUM_SEATS) * NUM_CARDS + c] < 0.1);
					}
				}
			}
		}
	}
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

	// Zero-initialize the game state and the views.
	auto gameStateTensor = torch::zeros(
		{int(games.size()), int(NUM_STATE_SETS * NUM_CARDS)},
		torch::TensorOptions().dtype(torch::kFloat));

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
					{int(n), int(NUM_VIEW_SETS * NUM_CARDS)},
					torch::TensorOptions().dtype(torch::kFloat));
			}
		}
	}

	// Deal the cards.
	std::array<uint8_t, NUM_CARDS> deck;
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		deck[c] = c;
	}
	for (size_t g = 0; g < games.size(); g++)
	{
		std::shuffle(deck.begin(), deck.end(), rng);
		size_t deckoffset = 0;
		for (size_t hand = 0; hand < NUM_SEATS + 1; hand++)
		{
			for (int _z = 0; _z < NUM_CARDS_PER_HAND; _z++)
			{
				assert(deckoffset < NUM_CARDS);
				uint8_t card = deck[deckoffset++];
				gameStateTensor[g][hand * NUM_CARDS + card] = 1;
				for (size_t s = 0; s < NUM_SEATS; s++)
				{
					if (hand > 0 && hand - 1 != s) continue;
					auto& brain = games[g].players[s].brain;
					size_t offset = games[g].players[s].relativeGameOffset;
					auto& view = brain->viewTensorPerSeat[s][offset];
					view[hand * NUM_CARDS + card] = 1;
				}
			}
		}
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Initializing took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}

	for (size_t g = 0; g < games.size(); g += (1 + (rng() % numGamesPerBrain)))
	{
		if (g == 0)
		{
			debugPrintGameState(games[g], gameStateTensor[g]);
		}
		assertCorrectGameState(games[g], gameStateTensor[g]);
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Verifying took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}

	if (ENABLE_CUDA)
	{
		gameStateTensor = gameStateTensor.contiguous().to(torch::kCUDA,
			torch::kHalf);
		for (size_t p = 0; p < NUM_PERSONALITIES; p++)
		{
			for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
			{
				auto& brain = _brainsPerPersonality[p][i];
				for (size_t s = 0; s < NUM_SEATS; s++)
				{
					auto& viewTensor = brain->viewTensorPerSeat[s];
					viewTensor = viewTensor.contiguous().to(torch::kCUDA,
						torch::kHalf);
				}
			}
		}
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Contiguizing took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}

	std::cout << "Playing " << games.size() << " games..." << std::endl;
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

			// Verify some of the games.
			for (size_t g = 0; g < games.size();
				g += (1 + (rng() % numGamesPerBrain)))
			{
				if (g == 0)
				{
					debugPrintGameState(games[g], gameStateTensor[g]);
				}
				assertCorrectGameState(games[g], gameStateTensor[g]);
			}
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
		start = end;
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
