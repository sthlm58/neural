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
		static const int Inputs = ImagePixelCount;
		static const int Output = 10;

		ImagesData images;
		Labels labels;
	};

	Data readTrainingData(const std::string& directory);
	Data readTestData(const std::string& directory);

}

