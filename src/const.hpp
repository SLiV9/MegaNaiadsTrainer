#pragma once

constexpr size_t NUM_SEATS = 4;
constexpr size_t NUM_STATE_SETS = 1 + NUM_SEATS + NUM_SEATS;
constexpr size_t NUM_VIEW_SETS = 1 + NUM_SEATS + NUM_SEATS + 1;
constexpr size_t NUM_SUITS = 4;
constexpr size_t NUM_FACES_PER_SUIT = 8;
constexpr size_t NUM_FAKE_CARDS = 4;
constexpr size_t NUM_CARDS = NUM_SEATS * NUM_FACES_PER_SUIT + NUM_FAKE_CARDS;
constexpr size_t NUM_CARDS_PER_HAND = 3;
constexpr size_t ACTION_SIZE = 2 * NUM_CARDS + 2;

#define ENABLE_CHEATING false

enum class Personality
{
	// Knight, Brute, Prince and Swindler all play the same.
	// We need three, because three of these can be playing against
	// the player in Left, Mid and Right position,
	// and we want the AI to learn to play in those situations.
	// (And we'll allow like three different strategies to develop.)
	NORMAL1,
	NORMAL2,
	NORMAL3,
	NORMAL4,
	NORMAL5,
	// In-game use:
	FOOL,
	ARTIST,
	TRICKSTER,
	FORGER,
	ILLUSIONIST,
	SPY,
	DRUNK,
	// Boss-battle:
	GOON,
	BOSS,
	// Training only:
	PLAYER,
	GREEDY,
	DUMMY,
	EMPTY,
	// No longer used:
	DUELIST,
};
#if ENABLE_CHEATING
constexpr size_t NUM_NORMAL_PERSONALITIES = size_t(Personality::DRUNK) + 1;
#else
constexpr size_t NUM_NORMAL_PERSONALITIES = size_t(Personality::NORMAL5) + 1;
#endif
constexpr size_t NUM_PERSONALITIES = size_t(Personality::EMPTY) + 1;
constexpr size_t NUM_BRAINS_PER_PERSONALITY = 25;

#define ENABLE_CUDA torch::cuda::is_available()

#define BRAIN_OUTPUT_FOLDER "C:\\Users\\sande\\Documents\\Cpp\\MegaNaiadsTrainer\\brains"
