#pragma once

#include <queue>
#include <memory>

#include "const.hpp"

class Module;


class Brain
{
private:
	std::shared_ptr<Module> _module;

public:
	Brain();
	Brain(const Brain&) = delete;
	Brain(Brain&& other) = default;
	Brain& operator=(const Brain&) = delete;
	Brain& operator=(Brain&&) = default;
	~Brain() = default;
};
