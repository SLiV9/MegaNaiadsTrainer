#pragma once

#include "Godot.hpp"
#include "Node.hpp"


namespace godot
{

class Game : public Node
{
	GODOT_CLASS(Game, Node)

public:
	static void _register_methods();

	void _init();

	void _ready();
};

}
