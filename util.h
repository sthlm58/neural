#pragma once

#include <algorithm>
#include <numeric>
#include <random>

namespace util
{

template <typename T1, typename T2>
struct zip2_iterator
{
	std::pair<const typename T1::value_type&, const typename T2::value_type&> operator*()
	{
		return { *it1, *it2 };
	}

	bool operator==(zip2_iterator other)
	{
		return it1 == other.it1 && it2 == other.it2;
	}

	bool operator!=(zip2_iterator other)
	{
		return !(*this == other);
	}

	zip2_iterator& operator++()
	{
		it1++;
		it2++;

		return *this;
	}

	zip2_iterator operator++(int)
	{
		zip2_iterator copy = *this;

		it1++;
		it2++;

		return copy;
	}


	typename T1::const_iterator it1;
	typename T2::const_iterator it2;
};
template <typename T1, typename T2>
struct zipper2
{
	zipper2(T1& t1, T2& t2) : t1(t1), t2(t2) { }

	auto begin()
	{
		return zip2_iterator<T1, T2>{t1.begin(), t2.begin()};
	}

	auto end()
	{
		return zip2_iterator<T1, T2>{t1.end(), t2.end()};
	}

	T1& t1;
	T2& t2;
};

template <typename T1, typename T2>
auto zip(T1& t1, T2& t2)
{
	return zipper2<T1, T2>(t1, t2);
}

template <typename Container, typename Functor>
void pairwise(const Container& cont, Functor func)
{
	std::accumulate(cont.begin(), cont.end(), typename Container::value_type{}, [func](auto acc, auto elem)
	{
		func(acc, elem);
		return elem;
	});
}

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

template <std::size_t N>
std::vector<double> vectorized(int i)
{
	std::vector<double> v(N, 0.);
	v[i] = 1.0;
	return v;
}

template <std::size_t N>
std::array<double, N> arrayized(int i)
{
	std::array<double, N> v {};
	v[i] = 1.0;
	return v;
}

template <typename Container>
inline int argmax(const Container& container)
{
	return std::distance(container.begin(), std::max_element(container.begin(), container.end()));
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
	return randomVector<double>(size, util::randomNormal);
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
	return input < 0 ? 0.02*input : input;
}
inline double leakyReluPrime(double input)
{
	return input < 0 ? 0.02 : 1;
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
