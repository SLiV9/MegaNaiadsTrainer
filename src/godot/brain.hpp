#pragma once

#include "Godot.hpp"
#include "Node.hpp"

//#include "lib.hpp"


class Module;

namespace godot
{

class Brain : public Node
{
	GODOT_CLASS(Brain, Node)

private:
	//std::unique_ptr<Module, decltype(&module_deallocate)> _module;

public:
	Array input;
	bool wantsToPass;
	bool wantsToSwap;
	int tableCard;
	int ownCard;

public:
	Brain();

	Brain(const Brain&) = delete;
	Brain(Brain&& other) = default;
	Brain& operator=(const Brain&) = delete;
	Brain& operator=(Brain&&) = default;
	~Brain();

	static void _register_methods();

	void _init();

	void _ready();

	void load(const String& filepath);
	void evaluate();
};

}
