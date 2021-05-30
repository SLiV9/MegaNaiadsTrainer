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
	int turnOfPass = -1;
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
		std::cout << "Seat " << s << ""
			" (" << game.players[s].brain->personality << ")"
			": ";
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
	if (false)
	{
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
	}
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
				if (isUsed)
				{
					throw std::runtime_error("assertion failed");
				}
				isUsed = true;
				numUsed += 1;
			}
		}
	}
	if (numUsed != NUM_CARDS_PER_HAND * (NUM_SEATS + 1))
	{
		throw std::runtime_error("assertion failed");
	}
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
		game.players[activeSeat].brain->totalConfidence +=
			std::min(std::min(tableCardWeight, ownCardWeight), 1.0f);

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
		game.players[activeSeat].brain->totalConfidence +=
			std::min(passWeight, 1.0f);

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

inline void tallyGameResult(const Game& game,
	const uint8_t* state)
{
	std::array<float, NUM_SEATS> handValues = { 0 };
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
		std::array<float, NUM_SUITS> suitValue = { 0 };
		for (size_t h = 0; h < NUM_CARDS_PER_HAND; h++)
		{
			uint8_t suit = hand[h] / NUM_FACES_PER_SUIT;
			uint8_t face = hand[h] % NUM_FACES_PER_SUIT;
			constexpr int VALUE_PER_FACE[NUM_FACES_PER_SUIT] = {
				7, 8, 9, 10, 10, 10, 10, 11
			};
			suitValue[suit] += VALUE_PER_FACE[face];
			if (h > 0)
			{
				hasMatch = hasMatch && (face == matchingFace);
			}
		}
		float v = 0;
		for (size_t suit = 0; suit < NUM_SUITS; suit++)
		{
			if (v < suitValue[suit])
			{
				v = suitValue[suit];
			}
		}
		if (hasMatch && v < 30.5)
		{
			v = 30.5;
		}
		handValues[s] = v;
	}
	float leastHandValue = 100;
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		if (handValues[s] < leastHandValue)
		{
			leastHandValue = handValues[s];
		}
	}
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		auto& brain = game.players[s].brain;
		if (handValues[s] == leastHandValue)
		{
			brain->numLosses += 1;
			brain->totalLosingHandValue += handValues[s];
		}
		else
		{
			brain->totalSurvivingHandValue += handValues[s];
		}
		brain->totalHandValue += handValues[s];
		brain->totalTurnsPlayed += game.players[s].turnOfPass + 1;
	}
}

void Trainer::playRound()
{
	auto start = std::chrono::high_resolution_clock::now();
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
				brain->reset(s);
			}
			brain->numLosses = 0;
			brain->totalTurnsPlayed = 0;
			brain->totalConfidence = 0;
			brain->totalHandValue = 0;
			brain->totalLosingHandValue = 0;
			brain->totalSurvivingHandValue = 0;
			brain->objectiveScore = 0;
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
				"...\t" << std::flush;

			// Verify some of the games.
			for (size_t g = 0; g < games.size();
				g += (1 + (rng() % numGamesPerBrain)))
			{
				if (g == 0 && games[g].numPassed() < NUM_SEATS)
				{
					std::cout << std::endl;
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

			std::cout << "Evaluating...\t" << std::flush;

			// Let all brains evaluate their positions.
			for (size_t p = 0; p < NUM_PERSONALITIES; p++)
			{
				for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
				{
					_brainsPerPersonality[p][i]->evaluate(s);
				}
			}

			std::cout << "Updating...\t" << std::flush;

			// Use the results to change the game state.
			size_t numUnfinished = 0;
			for (size_t g = 0; g < games.size(); g++)
			{
				updateGameState(games[g], gameState[g].data(), s);
				if (games[g].players[s].hasPassed
					&& games[g].players[s].turnOfPass < 0)
				{
					games[g].players[s].turnOfPass = t;
				}
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
	for (size_t g = 0; g < games.size(); g ++)
	{
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			if (games[g].players[s].turnOfPass < 0)
			{
				games[g].players[s].turnOfPass = maxTurnsPerPlayer;
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

	// Verify and tally all of the games.
	for (size_t g = 0; g < games.size(); g ++)
	{
		if (g == 0)
		{
			debugPrintGameState(games[g], gameState[g].data());
		}
		assertCorrectGameState(games[g], gameState[g].data());
		tallyGameResult(games[g], gameState[g].data());
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Tallying took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}
}

void Trainer::sortBrains()
{
	auto start = std::chrono::high_resolution_clock::now();

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			// Main objective: lose as few games as possible.
			int num = 0;
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				num += brain->numGamesPerSeat[s];
			}
			brain->objectiveScore = 1000.0 * (num - brain->numLosses) / num;
			// Bonus objective: get highest possible hand value.
			float handValue = (brain->totalHandValue / num);
			float maxHandValue = 31.0;
			brain->objectiveScore += 1000 * (handValue / maxHandValue);
		}
	}

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		// Sort the brains from least losses ot mos
		std::sort(_brainsPerPersonality[p].begin(),
			_brainsPerPersonality[p].end(),
			[](const auto& a, const auto& b) {
				return (a->objectiveScore > b->objectiveScore);
			});
	}

	std::cout << std::endl;
	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		int pSurv = 0;
		int pNum = 0;
		int pTotalTurnsPlayed = 0;
		float pTotalConfidence = 0;
		float pTotalHandValue = 0;
		float pTotalLossValue = 0;
		float pTotalWinValue = 0;
		float pTotalScore = 0;
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			int num = 0;
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				num += brain->numGamesPerSeat[s];
			}
			int surv = num - brain->numLosses;
			float averageTurnsBeforePass = 1.0
				* brain->totalTurnsPlayed / num;
			float averageConfidence = brain->totalConfidence
				/ std::max(1, brain->totalTurnsPlayed);
			float averageWinValue = brain->totalSurvivingHandValue
				/ std::max(1, surv);
			float averageLossValue = brain->totalLosingHandValue
				/ std::max(1, num - surv);
			std::cout << (i + 1) << ":\t"
				"" << brain->personality << brain->serialNumber << ""
				" (" << brain->motherNumber << "+" << brain->fatherNumber << ")"
				" scored " << (0.1 * int(10 * brain->objectiveScore)) << ""
				", survived"
				" " << (0.1 * int(100 * 10 * surv / num)) << "%"
				" of games"
				", played"
				" " << (0.1 * int(10 * averageTurnsBeforePass)) << " turns"
				" and had average hand value"
				" " << (0.1 * int(10 * brain->totalHandValue / num)) << ""
				" (win: " << (0.1 * int(10 * averageWinValue)) << ""
				", loss: " << (0.1 * int(10 * averageLossValue)) << ")"
				" with confidence"
				" " << (0.1 * int(100 * 10 * averageConfidence)) << "%"
				"" << std::endl;
			pSurv += surv;
			pNum += num;
			pTotalTurnsPlayed += brain->totalTurnsPlayed;
			pTotalConfidence += brain->totalConfidence;
			pTotalHandValue += brain->totalHandValue;
			pTotalWinValue += brain->totalSurvivingHandValue;
			pTotalLossValue += brain->totalLosingHandValue;
			pTotalScore += brain->objectiveScore;
		}
		float pAverageScore = pTotalScore / NUM_BRAINS_PER_PERSONALITY;
		float pAverageTurnsBeforePass = 1.0 * pTotalTurnsPlayed / pNum;
		float pAverageConfidence = pTotalConfidence
			/ std::max(1, pTotalTurnsPlayed);
		float pAverageWinValue = pTotalWinValue / std::max(1, pSurv);
		float pAverageLossValue = pTotalLossValue / std::max(1, pNum - pSurv);
		std::cout << "Overall, " << char('A' + p) << ""
			" scored " << (0.1 * int(10 * pAverageScore)) << ""
			", survived"
			" " << (0.1 * int(100 * 10 * pSurv / pNum)) << "%"
			" of games"
			", played"
			" " << (0.1 * int(10 * pAverageTurnsBeforePass)) << " turns"
			" and had average hand value"
			" " << (0.1 * int(10 * pTotalHandValue / pNum)) << ""
			" (win: " << (0.1 * int(10 * pAverageWinValue)) << ""
			", loss: " << (0.1 * int(10 * pAverageLossValue)) << ")"
			" with confidence"
			" " << (0.1 * int(100 * 10 * pAverageConfidence)) << "%"
			"" << std::endl;
		std::cout << std::endl;
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Sorting brains took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}
}

void Trainer::evolveBrains()
{
	auto start = std::chrono::high_resolution_clock::now();

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		auto& pool = _brainsPerPersonality[p];
		// The top 40% of brains (per pool) is kept as is.
		size_t chunkSize = NUM_BRAINS_PER_PERSONALITY / 5;
		// The bottom 60% of brains will be culled.
		size_t i = NUM_BRAINS_PER_PERSONALITY - 1;
		// A fifth of the new pool will be mutations of the best fifth.
		// The amount of mutation decreases over time.
		float deviationFactor = 0.5 / sqrtf(_round + 1);
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			Brain brain = pool[k]->makeMutation(deviationFactor);
			pool[i] = std::make_shared<Brain>(std::move(brain));
		}
		// A fifth of the new pool will consist of the offspring of
		// pairs of brains from the best fifth and second fifth.
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			Brain brain = pool[k]->makeOffspringWith(*pool[chunkSize + k]);
			pool[i] = std::make_shared<Brain>(std::move(brain));
		}
		// The middle fifth of the new pool is spliced with
		// brains from the best fifth.
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			Brain brain = pool[i]->makeOffspringWith(*pool[k]);
			pool[i] = std::make_shared<Brain>(std::move(brain));
		}
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Evolving brains took " << elapsed << "ms"
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
			char personality = char('A' + p);
			_brainsPerPersonality[p][i] = std::make_shared<Brain>(personality);
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

	size_t numRounds = 1000;
	for (; _round < numRounds; _round++)
	{
		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;

		playRound();
		sortBrains();
		evolveBrains();

		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;
	}
}
