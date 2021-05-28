#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <ctime>

class Brain;


class Trainer
{
private:
	std::time_t _startTime;
	std::vector<std::shared_ptr<Brain>> _brains;
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
