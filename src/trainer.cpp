#include "trainer.hpp"

#include <algorithm>
#include <random>

#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <torch/torch.h>

#include "const.hpp"
#include "trainingbrain.hpp"


struct Player
{
	std::shared_ptr<TrainingBrain> brain;
	size_t relativeGameOffset;
	bool hasPassed = false;
	bool hasSwapped = false;
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

inline float determineHandValue(const Game& game,
	const uint8_t* state, size_t s);

inline void debugPrintCard(size_t card)
{
	if (card >= NUM_SUITS * NUM_FACES_PER_SUIT)
	{
		switch (card - NUM_SUITS * NUM_FACES_PER_SUIT)
		{
			case 0: std::cout << "CAf"; return; break;
			case 1: std::cout << "JKR"; return; break;
			case 2: std::cout << "H12"; return; break;
			case 3: std::cout << "SAf"; return; break;
		}
	}
	const char* SUITS[NUM_SUITS] = {"C", "D", "H", "S"};
	const char* FACES[NUM_FACES_PER_SUIT] = {"7", "8", "9", "10",
		"J", "Q", "K", "A"};
	std::cout << SUITS[card % NUM_SUITS] << FACES[card / NUM_SUITS];
}

inline void debugPrintGameState(const Game& game,
	const uint8_t* state, bool full = false)
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
		if (game.players[s].brain->personality == Personality::EMPTY)
		{
			continue;
		}
		std::cout << "Seat " << s << ""
			" (" << TrainingBrain::personalityName(
				game.players[s].brain->personality) << ")"
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
		std::cout << "   " << determineHandValue(game, state, s);
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
	if (full)
	{
		for (size_t i = 0; i < NUM_STATE_SETS * NUM_CARDS; i++)
		{
			if (i > 0 && i % NUM_CARDS == 0)
			{
				std::cout << std::endl;
			}
			else if (i > 0 && (i % NUM_CARDS) % NUM_SUITS == 0)
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
					debugPrintGameState(game, state, /*full=*/true);
					std::cerr << "card " << c << " used twice" << std::endl;
					throw std::runtime_error("assertion failed");
				}
				isUsed = true;
				numUsed += 1;
			}
		}
	}
	if (numUsed != NUM_CARDS_PER_HAND * (NUM_SEATS + 1))
	{
		debugPrintGameState(game, state, /*full=*/true);
		std::cerr << "cards missing" << std::endl;
		throw std::runtime_error("assertion failed");
	}
}

inline void fakeMove(uint8_t* state, size_t activeSeat, uint8_t a, uint8_t b)
{
	// Make it so a is the tableCard and b is the ownCard.
	if (state[b] > 0)
	{
		std::swap(a, b);
	}
	// Swap the cards.
	state[a] = 0;
	state[b] = 1;
	state[(1 + activeSeat) * NUM_CARDS + a] = 1;
	state[(1 + activeSeat) * NUM_CARDS + b] = 0;
	// But do not update vision because that is irreversible.
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
	float passWeight = output[2 * NUM_CARDS];
	bool swapOnPass = false;
	if (output[2 * NUM_CARDS + 1] > passWeight)
	{
		swapOnPass = true;
		passWeight = output[2 * NUM_CARDS + 1];
	}
	size_t tableCard = 0;
	float tableCardWeight = passWeight - 1;
	size_t ownCard = 0;
	float ownCardWeight = passWeight - 1;
	for (size_t c = 0; c < NUM_CARDS; c++)
	{
		if (state[c] > 0
			&& output[c] > tableCardWeight)
		{
			tableCard = c;
			tableCardWeight = output[c];
		}

		if (state[(1 + activeSeat) * NUM_CARDS + c] > 0
			&& output[NUM_CARDS + c] > ownCardWeight)
		{
			ownCard = c;
			ownCardWeight = output[NUM_CARDS + c];
		}
	}

	if (brain->personality == Personality::GREEDY)
	{
		passWeight = determineHandValue(game, state, activeSeat);
		tableCardWeight = 0;
		std::vector<uint8_t> tableCards;
		std::vector<uint8_t> ownCards;
		for (uint8_t c = 0; c < NUM_CARDS; c++)
		{
			if (state[c] > 0)
			{
				tableCards.push_back(c);
				for (uint8_t x : ownCards)
				{
					fakeMove(state, activeSeat, c, x);
					float value = determineHandValue(game, state, activeSeat);
					fakeMove(state, activeSeat, c, x);
					if (value > tableCardWeight)
					{
						tableCard = c;
						ownCard = x;
						tableCardWeight = value;
					}
				}
			}
			else if (state[(1 + activeSeat) * NUM_CARDS + c] > 0)
			{
				ownCards.push_back(c);
				for (uint8_t x : tableCards)
				{
					fakeMove(state, activeSeat, c, x);
					float value = determineHandValue(game, state, activeSeat);
					fakeMove(state, activeSeat, c, x);
					if (value > passWeight)
					{
						ownCard = c;
						tableCard = x;
						tableCardWeight = value;
					}
				}
			}
		}
		{
			for (size_t i = 0; i < ownCards.size(); i++)
			{
				fakeMove(state, activeSeat, ownCards[i], tableCards[i]);
			}
			float value = determineHandValue(game, state, activeSeat);
			if (value >= 25 && value > passWeight && value > tableCardWeight)
			{
				swapOnPass = true;
				passWeight = value;
			}
			for (size_t i = 0; i < ownCards.size(); i++)
			{
				fakeMove(state, activeSeat, ownCards[i], tableCards[i]);
			}
		}
		// If we can make a move without losing much value, keep playing.
		if (passWeight < 14 || tableCardWeight + 1 > passWeight)
		{
			passWeight = -1;
		}
		ownCardWeight = tableCardWeight;
	}

	if (tableCardWeight > passWeight && ownCardWeight > passWeight)
	{
		game.players[activeSeat].brain->totalConfidence +=
			std::max(0.0f,
				std::min(std::min(tableCardWeight, ownCardWeight), 1.0f));

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
			std::max(0.0f, std::min(passWeight, 1.0f));

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
		game.players[activeSeat].hasSwapped = swapOnPass;
	}

	// If a player makes 31, the game ends immediately.
	if (determineHandValue(game, state, activeSeat) >= 31.0f)
	{
		for (size_t s = 0; s < NUM_SEATS; s++)
		{
			game.players[s].hasPassed = true;
		}
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
	}
	for (size_t t = 0; t < NUM_SEATS; t++)
	{
		Personality otherPersonality = game.players[t].brain->personality;
		int tt = ((t + NUM_SEATS - activeSeat) % NUM_SEATS);
		for (size_t c = 0; c < NUM_CARDS; c++)
		{
			buffer[(1 + NUM_SEATS + tt) * NUM_CARDS + c] =
				state[(1 + NUM_SEATS + t) * NUM_CARDS + c];
			if (t == activeSeat
				|| brain->personality == Personality::SPY
				|| (brain->personality == Personality::GOON
					&& otherPersonality == Personality::BOSS))
			{
				buffer[(1 + tt) * NUM_CARDS + c] =
					state[(1 + t) * NUM_CARDS + c];
			}
			else
			{
				buffer[(1 + tt) * NUM_CARDS + c] =
					state[(1 + t) * NUM_CARDS + c]
						* state[(1 + NUM_SEATS + t) * NUM_CARDS + c];
			}
		}
		size_t offset = (1 + NUM_SEATS + NUM_SEATS) * NUM_CARDS;
		buffer[offset + tt] = (otherPersonality == Personality::EMPTY);
		offset += NUM_SEATS;
		buffer[offset + tt] = game.players[t].hasPassed;
		offset += NUM_SEATS;
		buffer[offset + tt] = (otherPersonality == Personality::PLAYER
				|| otherPersonality == Personality::GREEDY
				|| otherPersonality == Personality::DUMMY);
		offset += NUM_SEATS;
		buffer[offset + tt] = (otherPersonality == Personality::BOSS);
	}
}

inline float determineHandValue(const Game& game,
	const uint8_t* state, size_t s)
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
	uint8_t matchingFace = 0;
	std::array<float, NUM_SUITS> suitValue = { 0 };
	for (size_t h = 0; h < NUM_CARDS_PER_HAND; h++)
	{
		uint8_t suit = hand[h] % NUM_SUITS;
		uint8_t face = hand[h] / NUM_SUITS;
		if (face < NUM_FACES_PER_SUIT)
		{
			constexpr int VALUE_PER_FACE[NUM_FACES_PER_SUIT] = {
				7, 8, 9, 10, 10, 10, 10, 11
			};
			suitValue[suit] += VALUE_PER_FACE[face];
		}
		else
		{
			switch (hand[h] - NUM_SUITS * NUM_FACES_PER_SUIT)
			{
				case 0:
				case 3:
				{
					// suit = suit;
					face = NUM_FACES_PER_SUIT - 1; // ace
					suitValue[suit] += 11;
				}
				break;
				case 1:
				{
					face = 255;
					// joker has no value
				}
				break;
				case 2:
				{
					// suit = suit;
					face = NUM_FACES_PER_SUIT;
					suitValue[suit] += 12;
				}
				break;
			}
		}
		if (h > 0)
		{
			hasMatch = hasMatch && (face == matchingFace);
		}
		else
		{
			matchingFace = face;
		}
	}

	if (game.players[s].brain->personality == Personality::ILLUSIONIST
		&& !game.players[s].hasSwapped)
	{
		// Up to one non-Ace clubs becomes a spade.
		// Up to one non-Ace diamond becomes a heart.
		if (suitValue[0] <= 10 && suitValue[3] >= 14)
		{
			suitValue[3] += suitValue[0];
		}
		else if (suitValue[1] <= 10 && suitValue[2] >= 14)
		{
			suitValue[2] += suitValue[1];
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

	if (hasMatch && matchingFace == NUM_FACES_PER_SUIT - 1)
	{
		return 31.0f;
	}
	else if (hasMatch && v < 30.5f)
	{
		return 30.5f;
	}
	else if (game.players[s].brain->personality == Personality::FOOL)
	{
		// The Fool is trained to only receive points for sets,
		// so that when playing in the real game it will only try
		// to collect sets and foolishly discard aces and trumps.
		return 0;
	}
	else
	{
		return v;
	}
}

inline void tallyGameResult(const Game& game,
	const uint8_t* state)
{
	std::array<float, NUM_SEATS> handValues = { 0 };
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		handValues[s] = determineHandValue(game, state, s);
	}
	float leastHandValue = 100;
	for (size_t s = 0; s < NUM_SEATS; s++)
	{
		if (handValues[s] < leastHandValue
			&& game.players[s].brain->personality != Personality::EMPTY)
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
			if (brain->personality == Personality::BOSS)
			{
				for (size_t t = 0; t < NUM_SEATS; t++)
				{
					game.players[t].brain->numBossLosses += 1;
				}
			}
			else if (brain->personality == Personality::PLAYER
				|| brain->personality == Personality::GREEDY
				|| brain->personality == Personality::DUMMY)
			{
				for (size_t t = 0; t < NUM_SEATS; t++)
				{
					game.players[t].brain->numPlayerLosses += 1;
				}
			}
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

	std::array<uint8_t, NUM_NORMAL_PERSONALITIES> normies;
	for (size_t p = 0; p < NUM_NORMAL_PERSONALITIES; p++)
	{
		normies[p] = p;
	}

	std::vector<Game> games;
	size_t numGamesPerBrain = 1000;
	size_t numNormalGames = NUM_BRAINS_PER_PERSONALITY * numGamesPerBrain
		* normies.size() / NUM_SEATS;
	size_t numGoonGames = numGamesPerBrain;
	games.resize(numNormalGames + numGoonGames);

	size_t remGoonGames = numGoonGames;
	for (Game& game : games)
	{
		// Each game includes a stand in for the player.
		if ((rng() % 3) > 0)
		{
			// 33% chance of greedy, 33% chance of dummy
			size_t p = (size_t) ((rng() % 2)
				? Personality::GREEDY
				: Personality::DUMMY);
			size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
			game.players[0].brain = _brainsPerPersonality[p][i];
		}
		else
		{
			// 33% chance of player
			size_t p = (size_t) Personality::PLAYER;
			size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
			game.players[0].brain = _brainsPerPersonality[p][i];
		}

		// The other players are the actual AIs we are training.
		if (remGoonGames > 0)
		{
			remGoonGames--;
			size_t p = (size_t) Personality::BOSS;
			size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
			game.players[1].brain = _brainsPerPersonality[p][i];
			for (size_t s = 2; s < NUM_SEATS; s++)
			{
				p = (size_t) Personality::GOON;
				i = rng() % NUM_BRAINS_PER_PERSONALITY;
				game.players[s].brain = _brainsPerPersonality[p][i];
			}
		}
		else
		{
			std::shuffle(normies.begin(), normies.end(), rng);
			for (size_t s = 1; s < NUM_SEATS; s++)
			{
				size_t p = normies[s];
				size_t i = rng() % NUM_BRAINS_PER_PERSONALITY;
				game.players[s].brain = _brainsPerPersonality[p][i];
			}
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
			brain->calculateCorrelation(_round % 100 == 0);
			brain->numLosses = 0;
			brain->numBossLosses = 0;
			brain->numPlayerLosses = 0;
			brain->totalTurnsPlayed = 0;
			brain->totalConfidence = 0;
			brain->totalHandValue = 0;
			brain->totalLosingHandValue = 0;
			brain->totalSurvivingHandValue = 0;
			brain->objectiveScore = 0;
		}
	}

	// Deal the cards from a normal deck of playing cards.
	std::array<uint8_t, NUM_SUITS * NUM_FACES_PER_SUIT> deck;
	for (size_t c = 0; c < NUM_SUITS * NUM_FACES_PER_SUIT; c++)
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
				if (_z == 0 && hand > 0)
				{
					size_t s = hand - 1;
					switch (games[g].players[s].brain->personality)
					{
						case Personality::FORGER:
						{
							card = NUM_FACES_PER_SUIT * NUM_SUITS
								+ ((rng() % 2 == 0) ? 0 : 3);
						}
						break;
						case Personality::ARTIST:
						{
							card = NUM_FACES_PER_SUIT * NUM_SUITS + 2;
						}
						break;
						case Personality::TRICKSTER:
						{
							card = NUM_FACES_PER_SUIT * NUM_SUITS + 1;
						}
						break;
					}
				}
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
	size_t shownGameIndex = rng() % games.size();
	std::cout << "(Showing game #" << shownGameIndex << ".)" << std::endl;
	size_t maxTurnsPerPlayer = 5;
	bool allFinished = false;
	for (size_t t = 0; t < maxTurnsPerPlayer && !allFinished; t++)
	{
		for (size_t s = 0; s < NUM_SEATS && !allFinished; s++)
		{
			std::cout << "Preparing"
				" round " << _round << ""
				" turn " << (t * NUM_SEATS + s) << ""
				" (seat " << s << ")"
				"...\t" << std::flush;

			// Verify some of the games.
			if (games[shownGameIndex].numPassed() < NUM_SEATS)
			{
				std::cout << std::endl;
				debugPrintGameState(games[shownGameIndex],
					gameState[shownGameIndex].data());
			}
			for (size_t g = 0; g < games.size();
				g += (1 + (rng() % (games.size() / 100))))
			{
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
		if (g == shownGameIndex)
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
			int num = 0;
			for (size_t s = 0; s < NUM_SEATS; s++)
			{
				num += brain->numGamesPerSeat[s];
			}
			if (num == 0)
			{
				brain->objectiveScore = 0;
				continue;
			}
			// Main objective: get highest possible hand value.
			float handValue = (brain->totalHandValue / num);
			// Set the baseline to the value of the starting hand (DUMMY).
			float averageHandValue = 15.0f;
			float goodHandValue = 30.0f;
			brain->objectiveScore = 1000
				* (handValue - averageHandValue)
				/ (goodHandValue - averageHandValue);
			// Bonus objective: lose as few games as possible.
			float averageLosses = 0.25 * num;
			brain->objectiveScore += 100.0
				* (averageLosses - brain->numLosses)
				 / averageLosses;
			switch (brain->personality)
			{
				case Personality::GOON:
				{
					// Alternative objective: make sure the Boss does not lose.
					brain->objectiveScore = 1000.0
						* (averageLosses - brain->numBossLosses)
						/ averageLosses;
					// Bonus objective: make sure the Player loses.
					brain->objectiveScore += 100.0
						* (brain->numPlayerLosses - averageLosses)
						/ averageLosses;
				}
				break;
				default:
				break;
			}
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
		if (((Personality) p) == Personality::EMPTY)
		{
			continue;
		}
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
			if (num == 0)
			{
				std::cout << (i + 1) << ":\t"
					"" << TrainingBrain::personalityName(
						brain->personality) << ""
					"" << brain->serialNumber << ""
					" (" << brain->motherNumber << ""
					"+" << brain->fatherNumber << ")"
					" did not play any games!" << std::endl;
				continue;
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
				"" << TrainingBrain::personalityName(brain->personality) << ""
				"" << brain->serialNumber << ""
				" (" << brain->motherNumber << ""
				"+" << brain->fatherNumber << ")"
				" scored " << (0.1 * int(10 * brain->objectiveScore)) << ""
				", survived"
				" " << (0.1 * int(100 * 10 * surv / num)) << "%"
				", played"
				" " << (0.1 * int(10 * averageTurnsBeforePass)) << " turns"
				", hand value"
				" " << (0.1 * int(10 * brain->totalHandValue / num)) << ""
				" (win: " << (0.1 * int(10 * averageWinValue)) << ""
				", loss: " << (0.1 * int(10 * averageLossValue)) << ")"
				", confidence"
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
		if (pNum == 0)
		{
			continue;
		}
		float pAverageScore = pTotalScore / NUM_BRAINS_PER_PERSONALITY;
		float pAverageTurnsBeforePass = 1.0 * pTotalTurnsPlayed / pNum;
		float pAverageConfidence = pTotalConfidence
			/ std::max(1, pTotalTurnsPlayed);
		float pAverageWinValue = pTotalWinValue / std::max(1, pSurv);
		float pAverageLossValue = pTotalLossValue / std::max(1, pNum - pSurv);
		std::cout << "Overall,"
			" " << TrainingBrain::personalityName((Personality) p) << ""
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
		if (!TrainingBrain::isNeural((Personality) p))
		{
			continue;
		}
		auto& pool = _brainsPerPersonality[p];
		// The top 40% of brains (per pool) is kept as is.
		size_t chunkSize = NUM_BRAINS_PER_PERSONALITY / 5;
		// The bottom 60% of brains will be culled.
		size_t i = NUM_BRAINS_PER_PERSONALITY - 1;
		// A fifth of the new pool will be mutations of the best fifth.
		// The amount of mutation decreases over time.
		double deviationFactor = 0.05 / sqrt(_round + 1);
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			auto brain = pool[k]->makeMutation(deviationFactor);
			pool[i] = std::make_shared<TrainingBrain>(std::move(brain));
		}
		// A fifth of the new pool will consist of the offspring of
		// pairs of brains from the best fifth and second fifth.
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			auto brain = pool[k]->makeOffspringWith(*pool[chunkSize + k]);
			pool[i] = std::make_shared<TrainingBrain>(std::move(brain));
		}
		// The middle fifth of the new pool is spliced with
		// brains from the best fifth.
		for (size_t k = 0; k < chunkSize && i > 2 * chunkSize; k++, i--)
		{
			auto brain = pool[i]->makeOffspringWith(*pool[k]);
			pool[i] = std::make_shared<TrainingBrain>(std::move(brain));
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

void Trainer::saveBrains()
{
	auto start = std::chrono::high_resolution_clock::now();
	std::random_device rd;
	std::mt19937 rng(rd());

	std::string folder = BRAIN_OUTPUT_FOLDER "/" + std::to_string(_startTime);
	{
		struct stat buffer;
		if (stat(folder.c_str(), &buffer) != 0)
		{
			std::cout << "Making directory " << folder << std::endl;
#ifdef __unix__
			mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
			mkdir(folder.c_str());
#endif
		}
	}

	std::ofstream list;
	list.open(folder + "/round" + std::to_string(_round) + ".txt");

	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		for (size_t i = 0; i < NUM_BRAINS_PER_PERSONALITY; i++)
		{
			auto& brain = _brainsPerPersonality[p][i];
			{
				std::string name;
				name += TrainingBrain::personalityName(brain->personality);
				name += "_" + std::to_string(brain->serialNumber);
				name += "_" + std::to_string(brain->motherNumber);
				name += "_" + std::to_string(brain->fatherNumber);

				brain->save(folder + "/" + name + ".pth.tar");
				if (i == 0)
				{
					brain->save(folder + "/" + name + "_cpu.pth.tar",
						/*forceCPU=*/true);
				}
				brain->saveScan(folder + "/" + name + ".png");
				brain->saveCorrelationScan(
					folder + "/" + name + "_correlation.png");

				list << name << std::endl;
			}
		}
	}

	// Timing:
	{
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			end - start).count();
		std::cout << "Saving brains took " << elapsed << "ms"
			"" << std::endl;
		start = end;
	}
}

void Trainer::resume(std::string session, size_t round)
{
	std::string folder = BRAIN_OUTPUT_FOLDER "/" + session;
	std::string filename = folder + "/round" + std::to_string(round) + ".txt";

	std::ifstream file(filename);
	if (!file)
	{
		std::cerr << "Failed to open " << filename << std::endl;
		throw std::runtime_error("Failed to open " + filename);
	}
	std::string line;
	std::array<size_t, NUM_PERSONALITIES> countPerPersonality = { 0 };
	while (std::getline(file, line))
	{
		if (line.empty())
		{
			continue;
		}
		std::string name = line;
		std::stringstream strm = std::stringstream(line);
		std::string personality;
		if (!std::getline(strm, personality, '_')
			|| personality.empty())
		{
			std::cerr << "Ignoring '" << line << "'" << std::endl;
			continue;
		}
		size_t p = NUM_PERSONALITIES + 1000;
		for (size_t pp = 0; pp < NUM_PERSONALITIES; pp++)
		{
			if (personality
				== TrainingBrain::personalityName((Personality) pp))
			{
				p = pp;
			}
		}
		if (p >= NUM_PERSONALITIES)
		{
			std::cerr << "Ignoring unknown '" << line << "'" << std::endl;
			continue;
		}
		size_t i = countPerPersonality[p];
		if (i >= NUM_BRAINS_PER_PERSONALITY)
		{
			std::cerr << "Ignoring excess for " << personality << std::endl;
			continue;
		}
		_brainsPerPersonality[p][i] =
			std::make_shared<TrainingBrain>((Personality) p);
		_brainsPerPersonality[p][i]->load(folder + "/" + name + ".pth.tar");
		countPerPersonality[p] += 1;
	}
	for (size_t p = 0; p < NUM_PERSONALITIES; p++)
	{
		if (countPerPersonality[p] < NUM_BRAINS_PER_PERSONALITY)
		{
			std::cerr << "Adding brains for "
				<< TrainingBrain::personalityName((Personality) p)
				<< std::endl;
		}
	}

	_round = round + 1;
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
			if (_brainsPerPersonality[p][i]) continue;
			Personality personality = (Personality) p;
			_brainsPerPersonality[p][i] =
				std::make_shared<TrainingBrain>(personality);
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

	size_t numRounds = 10000;
	for (; _round <= numRounds; _round++)
	{
		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;

		playRound();
		sortBrains();
		if (_round % 100 == 0)
		{
			saveBrains();
		}
		evolveBrains();

		std::cout << "########################################" << std::endl;
		std::cout << "ROUND " << _round << std::endl;
		std::cout << "########################################" << std::endl;
	}
}
