#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "network.h"
#include "util.h"
#include "mnist_reader.h"

TEST_CASE("random")
{
	auto r1 = util::randomNormal(), r2 = util::randomNormal();
	CHECK(r1 != r2);

}

TEST_CASE("pairwise")
{
	std::vector<int> v { 1, 2, 3 };
	int cnt = 0;
	util::pairwise(v, [&cnt](auto prev, auto curr)
	{
		cnt++;
		CHECK(prev < curr);
	});

	CHECK(cnt == v.size());
}

TEST_CASE("architecture")
{
	Network n({2, 3, 4});
	auto arch = n.architecture();
	CHECK(arch.size() == 3);
	CHECK(arch == std::vector<std::size_t>{2, 3, 4});
	CHECK(n.layers[1].neurons[0].weights[0] != n.layers[1].neurons[1].weights[0]);

	CHECK_THROWS(Network{{}});
}

TEST_CASE("sigmoid")
{
	CHECK(util::sigmoid(-5.) < 0.1);
	CHECK(util::sigmoid(5) > 0.9);
	CHECK(util::sigmoid(0) == 0.5);
	CHECK(util::sigmoidPrime(0) > util::sigmoidPrime(-1));

}

TEST_CASE("basic feed forward")
{
	Network n({2, 2, 1});
	n.layers[1].neurons[0].weights[0] = 0.5;
	n.layers[1].neurons[0].weights[1] = 0.5;
	n.layers[1].neurons[0].bias = -1;
	n.layers[1].neurons[1].weights[0] = 0.5;
	n.layers[1].neurons[1].weights[1] = 0.5;
	n.layers[1].neurons[1].bias = 0.0;
	n.layers[2].neurons[0].weights[0] = 0.5;
	n.layers[2].neurons[0].weights[1] = 0.5;
	n.layers[2].neurons[0].bias = -0.5;

	auto result = n.feedForward({1.0, 1.0});
	CHECK(result[0] == 0);
}

TEST_CASE("learn only one sample")
{
	Network n({4, 2, 2}, &util::sigmoid, &util::sigmoidPrime, 3.);

	std::vector<double> in { 0, 1, 0, 1 };
	std::vector<double> out { 1, 1 };

	for (int epoch {}; epoch < 10; epoch++)
	{
		n.learnOnce(in, out);
	}

	const auto result = n.feedForward(in);
	CHECK(result[0] > 0.8);
	CHECK(result[1] > 0.8);
}

#include <iostream>
TEST_CASE("images")
{
	Network n({784, 30, 10}, &util::leakyRelu, &util::leakyReluPrime, 0.003);

	auto data = mnist::readTrainingData("d:/dev/cpp/handreco-data/");

	const auto& image = data.images.front();
	const auto& label = util::vectorized<10>(data.labels.front());

	for (int epoch {}; epoch < 300; epoch++)
	{
		n.learnOnce(image, label);
		const auto result = n.feedForward(image);
		std::cout << util::argmax(result) << std::endl;
	}

	const auto result = n.feedForward(image);
	CHECK(util::argmax(result) == data.labels.front());
}
