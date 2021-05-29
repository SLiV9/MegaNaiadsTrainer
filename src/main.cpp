#include <iostream>
#include <chrono>
#ifdef _MSC_VER
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "trainer.hpp"

static uint64_t currentMilliseconds()
{
	auto now = std::chrono::system_clock::now().time_since_epoch();
	return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

void run(int argc, char* argv[])
{
	std::cout << "Press Enter to start:" << std::endl;
	std::cin.ignore();

	srand(currentMilliseconds());

	Trainer trainer;
	trainer.train();

	std::cout << std::endl << "Done!" << std::endl;
}

#define TRY true

int main(int argc, char* argv[])
{
#if TRY
	try
	{
#endif
		run(argc, argv);
#if TRY
	}
	catch (std::exception& e)
	{
		std::cerr << "Exception: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "Unknown error" << std::endl;
	}
	return 0;
#endif
}
