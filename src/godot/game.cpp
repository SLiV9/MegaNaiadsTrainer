#include "game.hpp"

#include <memory>
#include <array>

#include "const.hpp"

using namespace godot;


void Game::_register_methods()
{
	register_method("_ready", &Game::_ready);
	register_method("evaluate_hand", &Game::evaluate_hand);
}

void Game::_init()
{
	// Nothing to do.
}

void Game::_ready()
{
	// Nothing to do.
}

float Game::evaluate_hand(int card1, int card2, int card3)
{
	std::array<int, NUM_CARDS_PER_HAND> hand = { card1, card2, card3 };
	bool hasMatch = true;
	uint8_t matchingFace = 0;
	std::array<float, NUM_SUITS> suitValue = { 0 };
	for (size_t h = 0; h < NUM_CARDS_PER_HAND; h++)
	{
		uint8_t card = (uint8_t) hand[(int) h];
		uint8_t suit = (uint8_t) (card % NUM_SUITS);
		uint8_t face = (uint8_t) (card / NUM_SUITS);
		if (face < NUM_FACES_PER_SUIT)
		{
			constexpr int VALUE_PER_FACE[NUM_FACES_PER_SUIT] = {
				7, 8, 9, 10, 10, 10, 10, 11
			};
			suitValue[suit] += VALUE_PER_FACE[face];
		}
		else
		{
			switch (card - NUM_SUITS * NUM_FACES_PER_SUIT)
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
	else
	{
		return v;
	}
}
