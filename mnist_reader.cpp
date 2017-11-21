#include "mnist_reader.h"

#include <array>
#include <fstream>
#include <iostream>

// hello MSVC
#include <intrin.h>

namespace
{
	mnist::Labels labelsFromFile(const std::string& file)
	{
		mnist::Labels labels {};
		std::ifstream labels_file(file, std::ios::binary);
		if (!labels_file) {
			std::cerr << "no such file" << std::endl;
			return labels;
		}

		int magic {};
		if (!labels_file.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
			std::cerr << "cannot read header" << std::endl;
			return labels;
		}

		magic = _byteswap_ulong(magic);

		if (magic != 0x00000801) {
			std::cerr << "not a valid file" << std::endl;
			return labels;
		}

		int count {};
		if (!labels_file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
			std::cerr << "cannot read size" << std::endl;
			return labels;
		}

		count = _byteswap_ulong(count);

		for (int i {}; i < count; i++)
		{
			char label {};
			if (!labels_file.read(reinterpret_cast<char*>(&label), sizeof(label))) {
				std::cerr << "cannot read label" << std::endl;
				return labels;
			}

			labels.push_back(label);
		}

		return labels;
	}

	mnist::ImagesData imagesDataFromFile(const std::string& file)
	{
		mnist::ImagesData images;
		std::ifstream images_file(file, std::ios::binary);
		if (!images_file) {
			std::cerr << "no such file" << std::endl;
			return images;
		}

		int magic {};
		if (!images_file.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
			std::cerr << "cannot read header" << std::endl;
			return images;
		}

		magic = _byteswap_ulong(magic);

		if (magic != 0x00000803) {
			std::cerr << "not a valid file" << std::endl;
			return images;
		}

		int count {};
		if (!images_file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
			std::cerr << "cannot read size" << std::endl;
			return images;
		}

		count = _byteswap_ulong(count);

		int width {};
		if (!images_file.read(reinterpret_cast<char*>(&width), sizeof(width))) {
			std::cerr << "cannot read width" << std::endl;
			return images;
		}
		width = _byteswap_ulong(width);
		if (width != mnist::ImageWidth) {
			std::cerr << "invalid width" << std::endl;
			return images;
		}

		int height {};
		if (!images_file.read(reinterpret_cast<char*>(&height), sizeof(height))) {
			std::cerr << "cannot read height" << std::endl;
			return images;
		}
		height = _byteswap_ulong(height);
		if (height != mnist::ImageHeight) {
			std::cerr << "invalid height" << std::endl;
			return images;
		}

		for (int i {}; i < count; i++)
		{
			std::array<unsigned char, mnist::ImagePixelCount> image_data;
			if (!images_file.read(reinterpret_cast<char*>(image_data.data()), image_data.size())) {
				std::cerr << "cannot read label" << std::endl;
				return images;
			}

			mnist::ImageData normalized;
			normalized.reserve(mnist::ImagePixelCount);
			std::transform(image_data.begin(), image_data.end(), std::back_inserter(normalized), [](auto pix){ return pix / 255.; });
			images.push_back(normalized);
		}

		return images;
	}
}

mnist::Data mnist::readTrainingData(const std::string& directory)
{
	const auto labels = labelsFromFile(directory + "train-labels.idx1-ubyte");
	const auto images = imagesDataFromFile(directory + "train-images.idx3-ubyte");

	return { images, labels };
}
