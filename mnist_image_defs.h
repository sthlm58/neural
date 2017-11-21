#pragma once

#include <vector>

namespace mnist
{

	constexpr int ImageWidth = 28;
	constexpr int ImageHeight = 28;
	constexpr int ImagePixelCount = ImageWidth * ImageHeight;

	using Label = int;
	using Labels = std::vector<Label>;

	using ImageData = std::vector<double>;
	using ImagesData = std::vector<ImageData>;

	struct Data
	{
		static const std::size_t Inputs = ImagePixelCount;
		static const std::size_t Outputs = 10;

		ImagesData images;
		Labels labels;
	};

}
