#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <ctime>
#include <array>

#include "const.hpp"

class TrainingBrain;


class Trainer
{
private:
	std::time_t _startTime;
	std::array<
		std::array<std::shared_ptr<TrainingBrain>, NUM_BRAINS_PER_PERSONALITY>,
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
	void sortBrains();
	void evolveBrains();
	void saveBrains();

public:
	void resume(std::string session, size_t round);
	void train();
};
