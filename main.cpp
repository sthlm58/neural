#include "network.h"

#include "mnist_reader.h"
#include "misc.h"

#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef DOCTEST_CONFIG_DISABLE

int verify(const Network& n, const std::vector<std::vector<double>>& input, const std::vector<int>& labels)
{
	int correct {};
	for (std::size_t d{}; d < input.size(); ++d)
	{
		const auto& image = input[d];
		const auto result = n.feedForward(image);

		if (misc::argmax(result) == labels[d])
		{
			correct++;
		}
	}

	return correct;
}

int main()
{
	auto data = mnist::readTrainingData("d:/dev/cpp/handreco-data/");

	static const std::size_t HIDDEN_UNITS = 10;
	Network n({mnist::Data::Inputs, HIDDEN_UNITS, mnist::Data::Outputs},
			  misc::leakyRelu, misc::leakyReluPrime, 0.003);

	static const std::size_t LEARNING_SAMPLES = 50000;

	const auto test_data = mnist::ImagesData(data.images.begin(), data.images.begin() + LEARNING_SAMPLES);
	const auto test_labels = mnist::Labels(data.labels.begin(), data.labels.begin() + LEARNING_SAMPLES);

	const auto verification_data = mnist::ImagesData(data.images.begin() + LEARNING_SAMPLES, data.images.end());
	const auto verification_labels = mnist::Labels(data.labels.begin() + LEARNING_SAMPLES, data.labels.end());

	std::cout << "before: " << verify(n, verification_data, verification_labels) << std::endl;

	for (int epoch {}; epoch < 1000; epoch++)
	{
		auto before = std::chrono::high_resolution_clock::now();
		for (std::size_t d{}; d < test_data.size(); ++d)
		{
			const auto& image = test_data[d];
			const auto& label = misc::vectorized<10>(test_labels[d]);
			n.learnOnce(image, label);
		}
		auto after = std::chrono::high_resolution_clock::now();

		std::cout << "epoch " << epoch + 1 << ": " << verify(n, verification_data, verification_labels)
				  << " after " << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() << " ms" << std::endl;
	}
}

#endif
