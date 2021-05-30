#pragma once

#include <regex>

#include <torch/torch.h>

// Source:
// https://github.com/Kolkir/mlcpp/blob/master/mask_rcnn_pytorch/stateloader.cpp

inline void save_state_dict(const torch::nn::Module& module,
	const std::string& filename)
{
	torch::serialize::OutputArchive archive;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	for (const auto& val : params)
	{
		archive.write(val.key(), val.value());
	}
	for (const auto& val : buffers)
	{
		archive.write(val.key(), val.value(), /*is_buffer*/ true);
	}
	archive.save_to(filename);
}

inline void load_state_dict(torch::nn::Module& module,
	const std::string& filename, const std::string& ignore_name_regex = "")
{
	torch::serialize::InputArchive archive;
	archive.load_from(filename, torch::kCPU);
	torch::NoGradGuard no_grad;
	std::regex re(ignore_name_regex);
	std::smatch m;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	bool typeDiscrepancyDetected = false;
	for (auto& val : params)
	{
		if (!std::regex_match(val.key(), m, re))
		{
			try
			{
				archive.read(val.key(), val.value());
			}
			catch (const torch::Error&)
			{
				typeDiscrepancyDetected = true;
				module.to(torch::kHalf);
				archive.read(val.key(), val.value());
			}
		}
	}
	for (auto& val : buffers)
	{
		if (!std::regex_match(val.key(), m, re))
		{
			archive.read(val.key(), val.value(), /*is_buffer*/ true);
		}
	}
	if (typeDiscrepancyDetected)
	{
		module.to(torch::kFloat);
	}
}
