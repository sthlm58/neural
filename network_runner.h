#pragma once

#include "network.h"
#include "static_network.h"

#include "mnist_reader.h"
#include "mnist_custom_reader.h"
#include "util.h"

#include <chrono>
#include <iostream>
#include <iomanip>

template <typename NetworkType>
std::pair<std::size_t, std::size_t> results(NetworkType& n, const std::vector<std::vector<double>>& input, const std::vector<int>& labels)
{
	std::size_t correct {};
	for (std::size_t d{}; d < input.size(); ++d)
	{
		const auto& image = input[d];
		const auto result = n.feedForward(image);

		if (util::argmax(result) == labels[d])
		{
			correct++;
		}
	}

	return { correct, input.size() };
}



template <typename NetworkType>
void run_network(NetworkType& n)
{
	auto data = mnist::readTrainingData("d:/dev/cpp/handreco-data/");


	static const std::size_t LEARNING_SAMPLES = 50000;

	const auto learning_data = mnist::ImagesData(data.images.begin(), data.images.begin() + LEARNING_SAMPLES);
	const auto learning_labels = mnist::Labels(data.labels.begin(), data.labels.begin() + LEARNING_SAMPLES);

	const auto verification_data = mnist::ImagesData(data.images.begin() + LEARNING_SAMPLES, data.images.end());
	const auto verification_labels = mnist::Labels(data.labels.begin() + LEARNING_SAMPLES, data.labels.end());

	std::cout << "before: " << results(n, verification_data, verification_labels).first << std::endl;

	for (int epoch {}; epoch < 3; epoch++)
	{
		auto before = std::chrono::high_resolution_clock::now();
		for (std::size_t i {}; i < learning_data.size(); i++)
		{
			const auto& image = learning_data[i];
			const auto& label = util::vectorized<mnist::Data::Outputs>(learning_labels[i]);
			n.learnOnce(image, label);
		}
		auto after = std::chrono::high_resolution_clock::now();

		std::cout << "epoch " << epoch + 1 << ": " << results(n, verification_data, verification_labels).first
				  << " after " << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << " ms" << std::endl;
	}

	auto own = mnist::custom::readImagesMatching("d:/dev/cpp/handreco-data", "?__*.*");
	auto own_results = results(n, own.images, own.labels);
	std::cout << "own images: " << own_results.first << "/" << own_results.second << std::endl;
}

void run_dynamic_network()
{
	static const std::size_t HIDDEN_UNITS = 50;
	static const double LEARNING_FACTOR = 0.003;
	Network n({mnist::Data::Inputs, HIDDEN_UNITS, mnist::Data::Outputs},
			  util::leakyRelu, util::leakyReluPrime, LEARNING_FACTOR);

	run_network(n);
}

void run_static_network()
{
	StaticNetwork<784, 50, 10, &util::leakyRelu, &util::leakyReluPrime, 3> n;

	run_network(n);
}
