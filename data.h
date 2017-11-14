#pragma once

#include <array>
#include <vector>

using Label = int;
using Labels = std::vector<Label>;



constexpr int ImageWidth = 28;
constexpr int ImageHeight = 28;
constexpr int ImagePixelCount = ImageWidth * ImageHeight;

using ImageData = std::vector<double>;
using ImagesData = std::vector<ImageData>;

Labels labelsFromFile(const std::string& file);
ImagesData imagesDataFromFile(const std::string& file);
