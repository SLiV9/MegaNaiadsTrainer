#include "brain.hpp"

#include <memory>
#include <vector>

#include "lib.hpp"


using namespace godot;

Brain::Brain() :
	_module(module_allocate(), module_deallocate)
{}

Brain::~Brain() = default;

void Brain::_register_methods()
{
	register_method("_ready", &Brain::_ready);
	register_method("load", &Brain::load);
	register_method("evaluate", &Brain::evaluate);
    register_property<Brain, Array>("input", &Brain::input, Array());
    register_property<Brain, bool>("wantsToPass", &Brain::wantsToPass, false);
    register_property<Brain, bool>("wantsToSwap", &Brain::wantsToSwap, false);
    register_property<Brain, int>("tableCard", &Brain::tableCard, 0);
    register_property<Brain, int>("ownCard", &Brain::ownCard, 0);
}

void Brain::_init()
{
	// Nothing to do.
}

void Brain::_ready()
{
	// Nothing to do.
}

void Brain::load(String filepath)
{
	{
		struct stat buffer;
		if (stat(filepath.utf8().get_data(), &buffer) != 0)
		{
			std::cerr << "Missing module file"
				" '" << filepath.utf8().get_data() << "'"
				"" << std::endl;
			return;
		}
	}

	module_load(_module.get(), filepath.utf8().get_data());
}

void Brain::evaluate()
{
	std::vector<float> buffer;
	buffer.resize(input.size(), 0);
	for (int i = 0; i < input.size(); i++)
	{
		buffer[i] = (float) input[i];
	}
	int pass = 0;
	int swap = 0;
	int tc = 0;
	int oc = 0;
	module_evaluate(_module.get(), buffer.data(),
		&pass, &swap, &tc, &oc);
	wantsToPass = (pass > 0);
	wantsToSwap = (swap > 0);
	tableCard = tc;
	ownCard = oc;
}
