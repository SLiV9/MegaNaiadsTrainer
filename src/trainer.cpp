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
	bool hasPassed = false;
};

struct Game
{
	std::array<Player, NUM_SEATS> players;

	size_t numPassed() const
	{
		size_t count = 0;
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			if (players[s].hasPassed)
			{
				count += 1;
			}
		}
		return count;
	}
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

inline void debugPrintGameState(const Game& game,
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
		if (game.players[s].hasPassed)
		{
			std::cout << " <passed>";
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

inline void updateGameState(Game& game,
	torch::Tensor& stateTensor, size_t activeSeat)
{
	if (game.players[activeSeat].hasPassed)
	{
		return;
	}

	torch::Tensor stateTensorCPU = stateTensor.to(torch::kCPU,
		torch::kFloat);
	const float* state = stateTensorCPU.data_ptr<float>();

	auto& brain = game.players[activeSeat].brain;
	size_t offset = game.players[activeSeat].relativeGameOffset;
	auto& outputTensor = brain->outputTensorPerSeat[activeSeat][offset];
	torch::Tensor outputTensorCPU = outputTensor.to(torch::kCPU,
		torch::kFloat);
	const float* output = outputTensorCPU.data_ptr<float>();
	float passWeight = std::max(0.0f, output[2 * NUM_CARDS]);
	bool swapOnPass = false;
	if (output[2 * NUM_CARDS + 1] > passWeight)
	{
		swapOnPass = true;
		passWeight = output[2 * NUM_CARDS + 1];
	}
	size_t tableCard = 0;
	float tableCardWeight = 0.0f;
	size_t ownCard = 0;
	float ownCardWeight = 0.0f;
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		if (state[c] > 0.5
			&& output[c] > passWeight
			&& output[c] > tableCardWeight)
		{
			tableCard = c;
			tableCardWeight = output[c];
		}

		if (state[(1 + activeSeat) * NUM_CARDS + c] > 0.5
			&& output[NUM_CARDS + c] > passWeight
			&& output[NUM_CARDS + c] > ownCardWeight)
		{
			ownCard = c;
			ownCardWeight = output[NUM_CARDS + c];
		}
	}

	if (tableCardWeight > passWeight && ownCardWeight > passWeight)
	{
		// Normal move.
		stateTensor[tableCard] = 0;
		stateTensor[ownCard] = 1;
		stateTensor[(1 + activeSeat) * NUM_CARDS + tableCard] = 1;
		stateTensor[(1 + activeSeat) * NUM_CARDS + ownCard] = 0;
		stateTensor[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + tableCard] = 1;
		stateTensor[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + ownCard] = 0;
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			auto& viewTensor = game.players[s].brain->viewTensorPerSeat[s][
				game.players[s].relativeGameOffset];
			viewTensor[tableCard] = 0;
			viewTensor[ownCard] = 1;
			int relhand = (activeSeat - s + NUM_SEATS) % NUM_SEATS;
			viewTensor[relhand * NUM_CARDS + tableCard] = 1;
			viewTensor[relhand * NUM_CARDS + ownCard] = 0;
			if (s == activeSeat)
			{
				viewTensor[(1 + NUM_SEATS) * NUM_CARDS + tableCard] = 1;
				viewTensor[(1 + NUM_SEATS) * NUM_CARDS + ownCard] = 0;
			}
		}

		// If all players but one have passed, the game ends after
		// that player's next turn.
		if (game.numPassed() == NUM_SEATS - 1)
		{
			game.players[activeSeat].hasPassed = true;
		}
	}
	else
	{
		if (swapOnPass)
		{
			// Swap with the table.
			for (size_t c = 0; c < NUM_CARDS; c++)
			{
				stateTensor[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + c] =
					stateTensor[c];
				auto swap = stateTensor[c];
				stateTensor[c] = stateTensor[(1 + activeSeat) * NUM_CARDS + c];
				stateTensor[(1 + activeSeat) * NUM_CARDS + c] = swap;
			}
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				auto& viewTensor = game.players[s].brain->viewTensorPerSeat[s][
					game.players[s].relativeGameOffset];
				int relhand = (activeSeat - s + NUM_SEATS) % NUM_SEATS;
				for (size_t c = 0; c < NUM_CARDS; c++)
				{
					if (s == activeSeat)
					{
						viewTensor[(1 + NUM_SEATS) * NUM_CARDS + c] =
							viewTensor[c];
					}
					auto swap = viewTensor[c];
					viewTensor[ownCard] = viewTensor[relhand * NUM_CARDS + c];
					viewTensor[relhand * NUM_CARDS + c] = swap;
				}
			}
		}

		game.players[activeSeat].hasPassed = true;
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
		// I plan on moving updateGameState() to the GPU, but for now
		// it is more efficient to leave it here.
		//gameStateTensor = gameStateTensor.contiguous().to(torch::kCUDA,
		//	torch::kHalf);

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
	bool allFinished = false;
	for (size_t t = 0; t < maxTurnsPerPlayer && !allFinished; t++)
	{
		for (size_t s = 0; s < NUM_SEATS && !allFinished; s++)
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

			std::cout << "Updating turn " << (t * NUM_SEATS + s) << ""
				" (seat " << s << ")"
				"..." << std::endl;

			// Use the results to change the game state.
			allFinished = true;
			for (size_t g = 0; g < games.size(); g++)
			{
				updateGameState(games[g], gameStateTensor[g], s);
				if (games[g].numPassed() < NUM_SEATS)
				{
					allFinished = false;
				}
			}

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
