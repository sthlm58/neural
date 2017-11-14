#include "network.h"

#include "dataLoad.h"
#include "misc.h"

#include <iostream>
#include <iomanip>

#ifdef DOCTEST_CONFIG_DISABLE

int main(
		)
{

	Network n({784, 30, 10}, 0.003);

	auto data = dataLoad();

	auto test_data = ImagesData(data.first.begin(), data.first.begin() + 50000);
	auto test_labels = Labels(data.second.begin(), data.second.begin() + 50000);

	auto verification_data = ImagesData(data.first.begin() + 50000, data.first.end());
	auto verification_labels = Labels(data.second.begin() + 50000, data.second.end());

	int correct1 {};
	for (std::size_t d{}; d < verification_data.size(); ++d)
	{
		const auto& image = verification_data[d];
		const auto result = n.feedForward(image);

		if (misc::argmax(result) == verification_labels[d])
		{
			correct1++;
		}

	}

//		std::cout << "Result: " << correct << " / " << verification_data.size() << "\n\n";
	std::cout << correct1 << "," ;

	for (int epoch = 0; epoch < 1000; epoch++)
	{
		std::cout << std::setprecision(2);
		for (std::size_t d{}; d < test_data.size(); ++d)
		{
			const auto& image = test_data[d];
			const auto& label = misc::vectorized<10>(test_labels[d]);
			n.learnOnce(image, label);

			if (d % 1000 == 0)
			{
//				std::cout << "\r" << "                                 ";
//				std::cout << "\r" << "Epoch " << epoch << " [" << d * 100. / test_data.size() << "%]";
			}


		//		std::cout << "Result: " << correct << " / " << verification_data.size() << "\n\n";

		}
		int correct {};
		for (std::size_t d{}; d < verification_data.size(); ++d)
		{
			const auto& image = verification_data[d];
			const auto result = n.feedForward(image);

			if (misc::argmax(result) == verification_labels[d])
			{
				correct++;
			}

		}
		std::cout << correct << "," << std::flush;

//		std::cout << "\r" << "                                 " << "\r";
//		std::cout << "Epoch " << epoch << " done.\n";

	}


//	const auto result = n.feedForward(image);
}

#endif
