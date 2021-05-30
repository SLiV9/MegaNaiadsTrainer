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
	const uint8_t* state)
{
	std::cout << "----------------------" << std::endl;
	std::cout << "Table: ";
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		if (state[c] > 0)
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
			if (state[(1 + s) * NUM_CARDS + c] > 0)
			{
				debugPrintCard(c);
				if (state[(1 + NUM_SEATS + s) * NUM_CARDS + c] > 0)
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
		std::cout << "   discarded: ";
		for (size_t c = 0; c < NUM_CARDS; c++)
		{
			if (state[(1 + s) * NUM_CARDS + c] < 1
				&& state[(1 + NUM_SEATS + s) * NUM_CARDS + c] > 0)
			{
				debugPrintCard(c);
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
		else if (i > 0 && i % NUM_FACES_PER_SUIT == 0)
		{
			std::cout << "  ";
		}
		std::cout << int(state[i]) << " ";
	}
	std::cout << std::endl;
	std::cout << "----------------------" << std::endl;
}

inline void assertCorrectGameState(const Game& game,
	const uint8_t* state)
{
	size_t numUsed = 0;
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		bool isUsed = false;
		for (size_t hand = 0; hand < NUM_SEATS + 1; hand++)
		{
			if (state[hand * NUM_CARDS + c] > 0)
			{
				assert(!isUsed);
				isUsed = true;
				numUsed += 1;
			}
		}
	}
	assert(numUsed == NUM_CARDS_PER_HAND * (NUM_SEATS + 1));
}

inline void updateGameState(Game& game,
	uint8_t* state, size_t activeSeat)
{
	if (game.players[activeSeat].hasPassed)
	{
		return;
	}

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
		if (state[c] > 0
			&& output[c] > passWeight
			&& output[c] > tableCardWeight)
		{
			tableCard = c;
			tableCardWeight = output[c];
		}

		if (state[(1 + activeSeat) * NUM_CARDS + c] > 0
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
		state[tableCard] = 0;
		state[ownCard] = 1;
		state[(1 + activeSeat) * NUM_CARDS + tableCard] = 1;
		state[(1 + activeSeat) * NUM_CARDS + ownCard] = 0;
		state[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + tableCard] = 1;
		state[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + ownCard] = 1;

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
				if (state[c] > 0)
				{
					state[c] = 0;
					state[(1 + activeSeat) * NUM_CARDS + c] = 1;
					state[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + c] = 1;
				}
				else if (state[(1 + activeSeat) * NUM_CARDS + c] > 0)
				{
					state[c] = 1;
					state[(1 + activeSeat) * NUM_CARDS + c] = 0;
					state[(1 + NUM_SEATS + activeSeat) * NUM_CARDS + c] = 1;
				}
			}
		}

		game.players[activeSeat].hasPassed = true;
	}
}

inline void updateViewBuffers(const Game& game,
	const uint8_t* state, size_t activeSeat)
{
	auto& brain = game.players[activeSeat].brain;
	size_t offset = game.players[activeSeat].relativeGameOffset;
	float* rawbuffer = brain->viewBufferPerSeat[activeSeat].data();
	float* buffer = &rawbuffer[offset * NUM_VIEW_SETS * NUM_CARDS];
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		buffer[c] = state[c];
		buffer[NUM_CARDS + c] = state[(1 + activeSeat) * NUM_CARDS + c];
	}
	for (size_t t = 0; t < NUM_SEATS; t++)
	{
		int tt = (1 + ((t + NUM_SEATS - activeSeat) % NUM_SEATS));
		for (size_t c = 0; c < NUM_CARDS; c++)
		{
			buffer[(1 + NUM_SEATS + tt) * NUM_CARDS + c] =
				state[(1 + NUM_SEATS + t) * NUM_CARDS + c];
			if (t != activeSeat)
			{
				buffer[(1 + tt) * NUM_CARDS + c] =
					state[(1 + t) * NUM_CARDS + c]
						* state[(1 + NUM_SEATS + t) * NUM_CARDS + c];
			}
		}
		buffer[(1 + NUM_SEATS + NUM_SEATS) * NUM_CARDS + tt] =
			game.players[t].hasPassed;
	}
}

inline void scoreGame(const Game& game,
	const uint8_t* state)
{
	float leastScore = 100;
	std::array<float, NUM_SEATS> scores = { 0 };
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		std::array<uint8_t, NUM_CARDS_PER_HAND> hand;
		{
			size_t h = 0;
			for (size_t c = 0; c < NUM_CARDS; c++)
			{
				if (state[(1 + s) * NUM_CARDS + c] > 0)
				{
					hand[h++] = c;
				}
			}
		}
		bool hasMatch = true;
		uint8_t matchingFace = hand[0] % NUM_FACES_PER_SUIT;
		std::array<float, NUM_SUITS> scorePerSuit = { 0 };
		for (size_t h = 0; h < NUM_CARDS_PER_HAND; h++)
		{
			uint8_t suit = hand[h] / NUM_FACES_PER_SUIT;
			uint8_t face = hand[h] % NUM_FACES_PER_SUIT;
			constexpr int SCORE_PER_FACE[NUM_FACES_PER_SUIT] = {
				7, 8, 9, 10, 10, 10, 10, 11
			};
			scorePerSuit[suit] += SCORE_PER_FACE[face];
			if (h > 0)
			{
				hasMatch = hasMatch && (face == matchingFace);
			}
		}
		for (size_t suit = 0; suit < NUM_SUITS; suit++)
		{
			if (scores[s] < scorePerSuit[suit])
			{
				scores[s] = scorePerSuit[suit];
			}
		}
		if (hasMatch && scores[s] < 30.5)
		{
			scores[s] = 30.5;
		}
		if (scores[s] < leastScore)
		{
			leastScore = scores[s];
		}
	}
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		if (scores[s] == leastScore)
		{
			game.players[s].brain->numLosses += 1;
		}
		game.players[s].brain->totalScore += scores[s];
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
	std::vector<std::array<uint8_t, NUM_STATE_SETS * NUM_CARDS>> gameState;
	gameState.resize(games.size());
	for (size_t g = 0; g < games.size(); g++)
	{
		std::fill(gameState[g].begin(), gameState[g].end(), 0);
	}

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
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
					" " << brain->numGamesPerSeat[s] << " games"
					" from seat " << s << "";
				if (s + 1 == NUM_SEATS)
				{
					std::cout << std::endl;
				}

				brain->reset(s);
			}
			brain->numLosses = 0;
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
				gameState[g][hand * NUM_CARDS + card] = 1;
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

	std::cout << "Playing " << games.size() << " games..." << std::endl;
	size_t maxTurnsPerPlayer = 10;
	bool allFinished = false;
	for (size_t t = 0; t < maxTurnsPerPlayer && !allFinished; t++)
	{
		for (size_t s = 0; s < NUM_SEATS && !allFinished; s++)
		{
			std::cout << "Preparing turn " << (t * NUM_SEATS + s) << ""
				" (seat " << s << ")"
				"..." << std::endl;

			// Verify some of the games.
			for (size_t g = 0; g < games.size();
				g += (1 + (rng() % numGamesPerBrain)))
			{
				if (g == 0 && games[g].numPassed() < NUM_SEATS)
				{
					debugPrintGameState(games[g], gameState[g].data());
				}
				assertCorrectGameState(games[g], gameState[g].data());
			}

			// Prepare the views for this turn.
			for (size_t g = 0; g < games.size(); g++)
			{
				updateViewBuffers(games[g], gameState[g].data(), s);
			}
			for (size_t p = 0; p < NUM_PERSONALITIES; p++)
			{
				for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
				{
					_brainsPerPersonality[p][i]->cycle(s);
				}
			}

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
			size_t numUnfinished = 0;
			for (size_t g = 0; g < games.size(); g++)
			{
				updateGameState(games[g], gameState[g].data(), s);
				if (games[g].numPassed() < NUM_SEATS)
				{
					numUnfinished += 1;
				}
			}
			allFinished = (numUnfinished == 0);

			std::cout << "Still " << numUnfinished << " games"
				" left unfinished." << std::endl;
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

	// Verify and score all of the games.
	for (size_t g = 0; g < games.size(); g ++)
	{
		if (g == 0)
		{
			debugPrintGameState(games[g], gameState[g].data());
		}
		assertCorrectGameState(games[g], gameState[g].data());
		scoreGame(games[g], gameState[g].data());
	}

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		int pSurv = 0;
		int pNum = 0;
		float pScore = 0;
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			int num = 0;
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				num += brain->numGamesPerSeat[s];
			}
			int surv = num - brain->numLosses;
			float score = brain->totalScore;
			std::cout << char('A' + p) << i << " survived"
				" " << (0.001 * int(100 * 1000 * surv / num)) << "%"
				" (" << surv << "/" << num << ")"
				" and scored"
				" " << (0.001 * int(1000 * score / num)) << ""
				" (" << score << " total)"
				"" << std::endl;
			pSurv += surv;
			pNum += num;
			pScore += score;
		}
		std::cout << char('A' + p) << " survived"
			" " << (0.001 * int(100 * 1000 * pSurv / pNum)) << "%"
			" (" << pSurv << "/" << pNum << ")"
			" and scored"
			" " << (0.001 * int(1000 * pScore / pNum)) << ""
			" (" << pScore << " total)"
			"" << std::endl;
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Scoring took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}
}

void Trainer::train()
{
	auto start = std::chrono::high_resolution_clock::now();

	if (ENABLE_CUDA)
	{
		std::cout << "CUDA enabled" << std::endl;
	}
	else
	{
		std::cout << "No CUDA available, or not enabled" << std::endl;
	}

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
