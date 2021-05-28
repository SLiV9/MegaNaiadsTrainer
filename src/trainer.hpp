#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <ctime>
#include <array>

#include "const.hpp"

class Brain;


class Trainer
{
public:
	struct Player
	{
		std::shared_ptr<Brain> brain;
		size_t relativeGameOffset;
	};

private:
	std::time_t _startTime;
	std::array<
		std::array<Player, NUM_BRAINS_PER_PERSONALITY>,
		NUM_PERSONALITIES> _brainsPerPersonality;
	size_t _round;

public:
	Trainer();
	Trainer(const Trainer&) = delete;
	Trainer(Trainer&& other) = delete;
	Trainer& operator=(const Trainer&) = delete;
	Trainer& operator=(Trainer&&) = delete;
	~Trainer() = default;

private:
	void playRound();

public:
	void train();
};
