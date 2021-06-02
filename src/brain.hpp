#pragma once

#include <queue>
#include <memory>

#include <torch/torch.h>

#include "const.hpp"

class Module;


class Brain
{
private:
	std::shared_ptr<Module> _module;

public:
	std::array<size_t, NUM_SEATS> numGamesPerSeat;
	std::array<torch::Tensor, NUM_SEATS> viewTensorPerSeat;
	std::array<torch::Tensor, NUM_SEATS> outputTensorPerSeat;
	std::array<std::vector<float>, NUM_SEATS> viewBufferPerSeat;
	const Personality personality;
	const size_t serialNumber;
	const size_t motherNumber;
	const size_t fatherNumber;
	int numLosses = 0;
	int numBossLosses = 0;
	int totalTurnsPlayed = 0;
	float totalConfidence = 0;
	float totalHandValue = 0;
	float totalLosingHandValue = 0;
	float totalSurvivingHandValue = 0;
	float objectiveScore = 0;

private:
	explicit Brain(Personality personality,
		size_t motherNumber, size_t fatherNumber,
		std::shared_ptr<Module> module);

public:
	explicit Brain(Personality personality);
	Brain(const Brain&) = delete;
	Brain(Brain&& other) = default;
	Brain& operator=(const Brain&) = delete;
	Brain& operator=(Brain&&) = default;
	~Brain() = default;

	static bool isNeural(Personality personality);
	static const char* personalityName(Personality personality);

	void reset(size_t seat);
	void evaluate(size_t seat);
	void cycle(size_t seat);

	Brain makeMutation(double deviationFactor) const;
	Brain makeOffspringWith(const Brain& other) const;

	void save(const std::string& filepath);
	void load(const std::string& filepath);

	void saveScan(const std::string& filepath);
};
