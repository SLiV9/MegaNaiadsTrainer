#include "game.hpp"

using namespace godot;


void Game::_register_methods()
{
	register_method("_ready", &Game::_ready);
}

void Game::_init()
{
	// Nothing to do.
}

void Game::_ready()
{
	// TODO
}
