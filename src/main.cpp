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
	std::string session;
	int round = 0;

	std::cout << "Session to resume:" << std::endl;
	std::cin >> session;

	if (!session.empty())
	{
		std::cout << "Round to resume:" << std::endl;
		std::cin >> round;
	}

	srand(currentMilliseconds());

	Trainer trainer;
	if (!session.empty())
	{
		trainer.resume(session, round);
	}
	trainer.train();

	std::cout << std::endl << "Done!" << std::endl;
}

#define TRY false

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
