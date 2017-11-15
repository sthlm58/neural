#pragma once

#include <algorithm>
#include <random>

namespace misc
{

	template <typename T>
	std::vector<double> asDoubles(T&& container)
	{
		std::vector<double> doubles;
		doubles.reserve(container.size());
		for (const auto& element : container)
		{
			doubles.push_back(element);
		}
		return doubles;
	}

	template <int N>
	std::vector<double> vectorized(int i)
	{
		std::vector<double> v(N, 0.);
		v[i] = 1.0;
		return v;
	}

	inline int argmax(const std::vector<double>& vector)
	{
		return std::distance(vector.begin(), std::max_element(vector.begin(), vector.end()));
	}

	inline double randomNormal()
	{
		static std::random_device randomizer;
		static std::default_random_engine engine{ randomizer() };
		static std::normal_distribution<double> distribution { 0, 0.01 };

		return distribution(engine);
	}

	template <typename T, typename Generator>
	std::vector<T> randomVector(std::size_t size, Generator&& generator)
	{
		std::vector<T> values(size);
		std::generate(values.begin(), values.end(), generator);

		return values;
	}

	const auto randomNormalVector = [](std::size_t size)
	{
		return randomVector<double>(size, misc::randomNormal);
	};



	inline double sigmoid(double input)
	{
		return 1. / (1. + std::exp(-input));
	}
	inline double sigmoidPrime(double input)
	{
		return sigmoid(input) * (1 - sigmoid(input));
	}

	inline double relu(double input)
	{
		return input < 0 ? 0.0*input : input;
	}
	inline double reluPrime(double input)
	{
		return input < 0 ? 0.0 : 1;
	}

	inline double leakyRelu(double input)
	{
		return input < 0 ? 0.01*input : input;
	}
	inline double leakyReluPrime(double input)
	{
		return input < 0 ? 0.01 : 1;
	}

	inline double identity(double input)
	{
		return input;
	}
	inline double identityPrime(double /*input*/)
	{
		return 1;
	}

}
