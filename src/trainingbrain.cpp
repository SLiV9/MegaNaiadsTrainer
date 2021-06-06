#include "trainingbrain.hpp"

#include "libs/lodepng/lodepng.h"

#include "module.hpp"
#include "stateloader.hpp"


// We are not backpropagating, so no need for gradient calculation.
static torch::NoGradGuard no_grad;

static size_t _brainSerialNumber = 0;

const char* TrainingBrain::personalityName(Personality p)
{
	switch (p)
	{
		case Personality::NORMAL1: return "A";
		case Personality::NORMAL2: return "B";
		case Personality::NORMAL3: return "C";
		case Personality::PLAYER: return "X";
		case Personality::FOOL: return "fool";
		case Personality::ARTIST: return "artist";
		case Personality::TRICKSTER: return "trickster";
		case Personality::FORGER: return "forger";
		case Personality::ILLUSIONIST: return "illusionist";
		case Personality::SPY: return "spy";
		case Personality::DRUNK: return "drunk";
		case Personality::GOON: return "goon";
		case Personality::BOSS: return "boss";
		case Personality::DUELIST: return "duelist";
		case Personality::GREEDY: return "greedy";
		case Personality::DUMMY: return "dummy";
		case Personality::EMPTY: return "empty";
	}
}

bool TrainingBrain::isNeural(Personality p)
{
	switch (p)
	{
		case Personality::DRUNK: return false;
		case Personality::GREEDY: return false;
		case Personality::DUMMY: return false;
		case Personality::EMPTY: return false;
		default: return true;
	}
}

inline std::shared_ptr<Module> createModuleBasedOnPersonality(Personality p)
{
	if (TrainingBrain::isNeural(p))
	{
		return std::make_shared<Module>();
	}
	else
	{
		return nullptr;
	}
}

TrainingBrain::TrainingBrain(Personality p, size_t mNum, size_t fNum,
		std::shared_ptr<Module> module) :
	_module(module),
	personality(p),
	serialNumber(++_brainSerialNumber),
	motherNumber(mNum),
	fatherNumber(fNum)
{
	if (!_module) {}
	else if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kFloat);
}

TrainingBrain::TrainingBrain(Personality personality) :
	TrainingBrain(personality, 0, 0,
		createModuleBasedOnPersonality(personality)
	)
{}

void TrainingBrain::reset(size_t seat)
{
	size_t n = numGamesPerSeat[seat] * NUM_VIEW_SETS * NUM_CARDS;
	viewBufferPerSeat[seat].resize(n, 0);
	viewTensorPerSeat[seat] = torch::zeros(
			{int(numGamesPerSeat[seat]), int(NUM_VIEW_SETS * NUM_CARDS)},
			torch::kFloat);
	if (ENABLE_CUDA)
	{
		viewTensorPerSeat[seat] = viewTensorPerSeat[seat].contiguous().to(
			torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
	}
	outputTensorPerSeat[seat] = torch::zeros(
			{int(numGamesPerSeat[seat]), int(ACTION_SIZE)},
			torch::kFloat);
}

void TrainingBrain::calculateCorrelation(bool on)
{
	if (on)
	{
		correlationTensor = torch::zeros(
			{int(ACTION_SIZE), int(NUM_VIEW_SETS * NUM_CARDS)},
			torch::kFloat);
		if (ENABLE_CUDA)
		{
			correlationTensor = correlationTensor.contiguous().to(
				torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
		}
	}
	else
	{
		correlationTensor = torch::empty(0);
	}
}

void TrainingBrain::evaluate(size_t seat)
{
	if (!TrainingBrain::isNeural(personality))
	{
		switch (personality)
		{
			case Personality::DRUNK:
			{
				outputTensorPerSeat[seat] = torch::rand(
					{int(numGamesPerSeat[seat]), int(ACTION_SIZE)},
					torch::kFloat);
				return;
			}
			break;
			default:
			{
				// Keep output as all zeros.
				return;
			}
			break;
		}
	}

	if (!_module)
	{
		std::cerr << "missing module"
			" for " << personalityName(personality) << serialNumber << ""
			"" << std::endl;
		return;
	}

	torch::Tensor outputTensor = _module->forward(viewTensorPerSeat[seat]);

	if (correlationTensor.size(0) > 0)
	{
		// Scale [0, 1] to [-1, 1].
		//torch::Tensor weight = torch::add(
		//	torch::mul(viewTensorPerSeat[seat], 2.0f),
		//	-1.0f);
		// Calculate outer product and add that to the existing correlation.
		for (size_t i = 0; i < numGamesPerSeat[seat]; i++)
		{
			const torch::Tensor& weight = viewTensorPerSeat[seat];
			correlationTensor.add_(outputTensor[i].ger(weight[i]));
		}
	}

	outputTensorPerSeat[seat] = outputTensor.to(torch::kCPU, torch::kFloat,
		/*non_blocking=*/true);
}

void TrainingBrain::cycle(size_t seat)
{
	torch::Tensor bufferTensor = torch::from_blob(
		viewBufferPerSeat[seat].data(),
		{
			int(viewBufferPerSeat[seat].size() / (NUM_VIEW_SETS * NUM_CARDS)),
			int(NUM_VIEW_SETS * NUM_CARDS)
		},
		torch::kFloat);
	if (ENABLE_CUDA)
	{
		viewTensorPerSeat[seat] = bufferTensor.contiguous().to(
			torch::kCUDA, torch::kHalf, /*non_blocking=*/true);
	}
	else
	{
		viewTensorPerSeat[seat] = bufferTensor;
	}
}

TrainingBrain TrainingBrain::makeMutation(double deviationFactor) const
{
	if (!_module)
	{
		return TrainingBrain(personality);
	}
	auto newModule = std::dynamic_pointer_cast<Module>(_module->clone());
	newModule->mutate(deviationFactor);
	return TrainingBrain(personality, serialNumber, 0, newModule);
}

TrainingBrain TrainingBrain::makeOffspringWith(const TrainingBrain& other) const
{
	if (!_module)
	{
		return TrainingBrain(personality);
	}
	auto newModule = std::dynamic_pointer_cast<Module>(_module->clone());
	newModule->spliceWith(*(other._module));
	return TrainingBrain(personality, serialNumber, other.serialNumber,
		newModule);
}

void TrainingBrain::save(const std::string& filepath, bool forceCPU)
{
	if (!_module)
	{
		if (TrainingBrain::isNeural(personality))
		{
			std::cerr << "missing module"
				" for " << personalityName(personality) << serialNumber << ""
				"" << std::endl;
		}
		return;
	}

	{
		struct stat buffer;
		if (stat(filepath.c_str(), &buffer) == 0)
		{
			std::cout << "Kept " << filepath << std::endl;
			return;
		}
	}

	if (forceCPU) _module->to(torch::kCPU, torch::kFloat);
	save_state_dict(*_module, filepath);
	if (forceCPU && ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	std::cout << "Saved " << filepath << std::endl;
}

void TrainingBrain::load(const std::string& filepath)
{
	if (!_module)
	{
		if (TrainingBrain::isNeural(personality))
		{
			std::cerr << "missing module"
				" for " << personalityName(personality) << serialNumber << ""
				"" << std::endl;
		}
		return;
	}

	{
		struct stat buffer;
		if (stat(filepath.c_str(), &buffer) != 0)
		{
			std::cerr << "no model in path " << filepath << std::endl;
			throw std::runtime_error("no model in path " + filepath);
		}
	}

	load_state_dict(*_module, filepath);
	if (ENABLE_CUDA) _module->to(torch::kCUDA, torch::kHalf);
	else _module->to(torch::kCPU, torch::kFloat);
	std::cout << "Loaded " << filepath << std::endl;
}

inline uint8_t paletteIndexFromValue(float value, float multiplier)
{
	// Scale [-X, X] to [0, 1].
	float rel = ((value * multiplier) + 1.0f) * 0.5f;
	// Scale [0, 1] to [0, 12], rounding to the nearest integer.
	// (Adding and subtracting 1 makes sure -0.75 is rounded to -1).
	int step = int(rel * 12 + 1.499999f) - 1;
	// Convert {0, ..., 12} to {2, ..., 14}, clamping the edges to 1 and 15.
	if (step < 0)
	{
		return 1;
	}
	else if (step > 12)
	{
		return 15;
	}
	return 2 + step;
}

constexpr int SCAN_BITDEPTH = 4;

inline void writeScan(const std::string& filepath,
	int imagew, int imageh,
	const std::vector<uint8_t>& data);

void TrainingBrain::saveScan(const std::string& filepath)
{
	if (!_module)
	{
		if (TrainingBrain::isNeural(personality))
		{
			std::cerr << "missing module"
				" for " << personalityName(personality) << serialNumber << ""
				"" << std::endl;
		}
		return;
	}

	{
		struct stat buffer;
		if (stat(filepath.c_str(), &buffer) == 0)
		{
			std::cout << "Keeping scan " << filepath << std::endl;
			return;
		}
	}

	int margin = 10;
	int padding = 10;
	int widthOfBias = 4;
	int heightOfGradient = 50;

	// Image dimensions:
	int imagew = 2 * margin - padding;
	int imageh = 2 * margin + heightOfGradient;
	for (const auto& layer : { _module->_fc1, _module->_fc2,
			_module->_fc3, _module->_fc4, _module->_fc5 })
	{
		int w = layer->options.in_features();
		int h = layer->options.out_features();
		imagew += padding + w + 1 + widthOfBias;
		imageh = std::max(2 * margin + h + padding + heightOfGradient, imageh);
	}

	std::cout << "Saving scan"
		" of size " << imagew << "x" << imageh << ""
		" to " << filepath << ""
		"" << std::endl;

	// Palette size can be 2, 4, 16 or 256 colors.
	const int paletteSize = 1 << SCAN_BITDEPTH;
	const int pixelsPerByte = 8 / SCAN_BITDEPTH;
	const uint8_t mask = (1 << SCAN_BITDEPTH) - 1;
	std::vector<size_t> histogram(paletteSize, 0);

	size_t numBytes = (imagew * imageh + pixelsPerByte - 1) / pixelsPerByte;
	std::vector<uint8_t> data(numBytes, 0);
	int xOfBlock = margin;
	int yOfBlock = margin;
	float weightMultiplier = 15.0f;

	for (const auto& layer : { _module->_fc1, _module->_fc2,
			_module->_fc3, _module->_fc4, _module->_fc5 })
	{
		const auto& tt = layer->weight.to(torch::kCPU, torch::kFloat);
		float* weight = tt.data_ptr<float>();
		int w = layer->options.in_features();
		int h = layer->options.out_features();
		float _boundsCheck = *(tt[h - 1][w - 1].data_ptr<float>());
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				int xx = xOfBlock + x;
				int yy = yOfBlock + y;
				float v = weight[y * w + x];
				uint8_t pix = paletteIndexFromValue(v, weightMultiplier);
				histogram[pix] += 1;
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
		}
		xOfBlock += w;
		if (!layer->options.bias())
		{
			xOfBlock += padding;
			continue;
		}
		xOfBlock += 1;
		const auto& bb = layer->bias.to(torch::kCPU, torch::kFloat);
		float _bbCheck = *(bb[h - 1].data_ptr<float>());
		float *bias = bb.data_ptr<float>();
		for (int y = 0; y < h; y += 1)
		{
			float v = bias[y];
			uint8_t pix = paletteIndexFromValue(v, weightMultiplier);
			histogram[pix] += 1;
			for (int x = 0; x < widthOfBias; x++)
			{
				int xx = xOfBlock + x;
				int yy = yOfBlock + y;
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
		}
		xOfBlock += widthOfBias + padding;
	}

	if (heightOfGradient > 3)
	{
		xOfBlock = margin;
		yOfBlock = imageh - margin - heightOfGradient;
		int w = imagew - 2 * margin;
		int h = heightOfGradient / 2 - 1;
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				int xx = xOfBlock + x;
				int yy = yOfBlock + y;
				int d = y - heightOfGradient / 2;
				if (2 * x < w) d *= -1;
				float v = -1.2f + 2.4f * (x + d) / w;
				uint8_t pix = paletteIndexFromValue(v, 1.0f);
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
		}
		yOfBlock += h + 1;

		xOfBlock = margin;
		size_t total = 0;
		for (int i = 0; i < paletteSize; i++)
		{
			total += histogram[i];
		}
		for (int pix = 0; pix < paletteSize; pix++)
		{
			int wOfPix = ((w - paletteSize) * histogram[pix] + total - 1)
				/ total;
			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < wOfPix; x++)
				{
					int xx = xOfBlock + x;
					int yy = yOfBlock + y;
					int i = yy * imagew + xx;
					int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
					data[i / pixelsPerByte] |=
						(pix & mask) << (part * SCAN_BITDEPTH);
				}
			}
			xOfBlock += wOfPix;
		}
	}

	writeScan(filepath, imagew, imageh, data);
}

void TrainingBrain::saveCorrelationScan(const std::string& filepath)
{
	if (!_module)
	{
		if (TrainingBrain::isNeural(personality))
		{
			std::cerr << "missing module"
				" for " << personalityName(personality) << serialNumber << ""
				"" << std::endl;
		}
		return;
	}

	{
		struct stat buffer;
		if (stat(filepath.c_str(), &buffer) == 0)
		{
			std::cout << "Keeping c-scan " << filepath << std::endl;
			return;
		}
	}

	int margin = 10;
	int separation = 2;
	int padding = 10;
	int widthOfBias = 5;
	int heightOfGradient = 50;

	// Image dimensions:
	int imagew = 2 * margin
		+ NUM_VIEW_SETS * (NUM_CARDS + separation)
		+ widthOfBias;
	int imageh = 2 * margin
		+ ((ACTION_SIZE + NUM_CARDS - 1) / NUM_CARDS) * (NUM_CARDS + separation)
		+ padding + heightOfGradient;

	std::cout << "Saving c-scan"
		" of size " << imagew << "x" << imageh << ""
		" to " << filepath << ""
		"" << std::endl;

	// Palette size can be 2, 4, 16 or 256 colors.
	const int paletteSize = 1 << SCAN_BITDEPTH;
	const int pixelsPerByte = 8 / SCAN_BITDEPTH;
	const uint8_t mask = (1 << SCAN_BITDEPTH) - 1;
	std::vector<size_t> histogram(paletteSize, 0);

	size_t numBytes = (imagew * imageh + pixelsPerByte - 1) / pixelsPerByte;
	std::vector<uint8_t> data(numBytes, 0);
	int xOfBlock = margin;
	int yOfBlock = margin;
	float weightMultiplier = 5.00f;

	std::array<float, ACTION_SIZE> bias;

	{
		int w = NUM_VIEW_SETS * NUM_CARDS;
		int h = ACTION_SIZE;
		const auto& tt = correlationTensor.to(torch::kCPU, torch::kFloat);
		float* weight = tt.data_ptr<float>();
		float _boundsCheck = *(tt[h - 1][w - 1].data_ptr<float>());
		for (int y = 0; y < h; y++)
		{
			bias[y] = 0;
			for (int x = 0; x < w; x++)
			{
				int xx = xOfBlock + (x / NUM_CARDS) * (NUM_CARDS + separation)
					+ (x % NUM_CARDS);
				int yy = yOfBlock + (y / NUM_CARDS) * (NUM_CARDS + separation)
					+ (y % NUM_CARDS);
				float v = weight[y * w + x] / std::max(1, totalTurnsPlayed);
				bias[y] += v;
				uint8_t pix = paletteIndexFromValue(v, weightMultiplier);
				histogram[pix] += 1;
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
			bias[y] = bias[y] / w;
		}
		xOfBlock += (w / NUM_CARDS) * (NUM_CARDS + separation);
	}

	{
		int h = ACTION_SIZE;
		for (int y = 0; y < h; y += 1)
		{
			float v = bias[y];
			uint8_t pix = paletteIndexFromValue(v, weightMultiplier);
			histogram[pix] += 1;
			for (int x = 0; x < widthOfBias; x++)
			{
				int xx = xOfBlock + x;
				int yy = yOfBlock + (y / NUM_CARDS) * (NUM_CARDS + separation)
					+ (y % NUM_CARDS);
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
		}
		xOfBlock += widthOfBias + padding;
	}

	if (heightOfGradient > 3)
	{
		xOfBlock = margin;
		yOfBlock = imageh - margin - heightOfGradient;
		int w = imagew - 2 * margin;
		int h = heightOfGradient / 2 - 1;
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				int xx = xOfBlock + x;
				int yy = yOfBlock + y;
				int d = y - heightOfGradient / 2;
				if (2 * x < w) d *= -1;
				float v = -1.2f + 2.4f * (x + d) / w;
				uint8_t pix = paletteIndexFromValue(v, 1.0f);
				int i = yy * imagew + xx;
				int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
				data[i / pixelsPerByte] |=
					(pix & mask) << (part * SCAN_BITDEPTH);
			}
		}
		yOfBlock += h + 1;

		xOfBlock = margin;
		size_t total = 0;
		for (int i = 0; i < paletteSize; i++)
		{
			total += histogram[i];
		}
		for (int pix = 0; pix < paletteSize; pix++)
		{
			int wOfPix = ((w - paletteSize) * histogram[pix] + total - 1)
				/ total;
			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < wOfPix; x++)
				{
					int xx = xOfBlock + x;
					int yy = yOfBlock + y;
					int i = yy * imagew + xx;
					int part = (pixelsPerByte - 1) - (i % pixelsPerByte);
					data[i / pixelsPerByte] |=
						(pix & mask) << (part * SCAN_BITDEPTH);
				}
			}
			xOfBlock += wOfPix;
		}
	}

	writeScan(filepath, imagew, imageh, data);
}

inline void writeScan(const std::string& filepath,
	int imagew, int imageh,
	const std::vector<uint8_t>& data)
{
	const int paletteSize = 1 << SCAN_BITDEPTH;

	lodepng::State state;
	const uint32_t palette[paletteSize] = {
		0x000000,
		0xffffff, 0xaef8db,
		0x2fedb7, 0x00d1c8, 0x00a2b8, 0x007495, 0x244966,
		0x202433,
		0x4a3659, 0x89416d, 0xc74e68, 0xf26d4d, 0xffa01c,
		0xfacf00, 0xe1ff00,
	};
	for (int i = 0; i < paletteSize; i++)
	{
		uint8_t r = palette[i] >> 16;
		uint8_t g = (palette[i] >> 8) & 0xFF;
		uint8_t b = palette[i] & 0xFF;
		lodepng_palette_add(&state.info_png.color, r, g, b, 0xFF);
		lodepng_palette_add(&state.info_raw, r, g, b, 0xFF);
	}

	state.info_png.color.colortype = LCT_PALETTE;
	state.info_png.color.bitdepth = SCAN_BITDEPTH;
	state.info_raw.colortype = LCT_PALETTE;
	state.info_raw.bitdepth = SCAN_BITDEPTH;
	state.encoder.auto_convert = false;

	{
		std::vector<uint8_t> buffer;
		auto error = lodepng::encode(buffer, data.data(),
			imagew, imageh, state);
		if (error)
		{
			std::cerr << "encoder error " << error << ": "
				<< lodepng_error_text(error) << std::endl;
		}

		error = lodepng::save_file(buffer, filepath.c_str());
		if (error)
		{
			std::cerr << "encoder error " << error << ": "
				<< lodepng_error_text(error) << std::endl;
		}
	}
}
