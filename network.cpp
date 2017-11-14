#include "network.h"

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include <sstream>

#include "dataLoad.h"

#include "doctest.h"

namespace doctest {
template <typename T>
String toString(const std::vector<T>& v)
{
	std::stringstream interp;
	interp << "[";
	for (std::size_t i = 0; i < v.size(); i++)
	{
		if (i > 0) interp << ",";
		interp << v[i];
	}
	interp << "]";

	return interp.str().c_str();
}
}




using DoublesVec = std::vector<double>;


TEST_CASE("random")
{
	auto r1 = misc::randomNormal(), r2 = misc::randomNormal();
	CHECK(r1 != r2);

}

TEST_CASE("architecture")
{
	Network n({2, 3, 4});
	auto arch = n.architecture();
	CHECK(arch.size() == 3);
	CHECK(arch == std::vector<std::size_t>{2, 3, 4});
	CHECK(n.layers[1].neurons[0].weights[0] != n.layers[1].neurons[1].weights[0]);
}

TEST_CASE("sigmoid")
{
	CHECK(misc::sigmoid(-5.) < 0.1);
	CHECK(misc::sigmoid(5) > 0.9);
	CHECK(misc::sigmoid(0) == 0.5);
	CHECK(misc::sigmoidPrime(0) > misc::sigmoidPrime(-1));

}

TEST_CASE("basic feed forward")
{
	Network n({2, 2, 1});
	n.layers[1].neurons[0].weights[0] = 0.5;
	n.layers[1].neurons[0].weights[1] = 0.5;
	n.layers[1].neurons[0].bias = -1;
	n.layers[1].neurons[1].weights[0] = 0.5;
	n.layers[1].neurons[1].weights[1] = 0.5;
	n.layers[1].neurons[1].bias = -1;
	n.layers[2].neurons[0].weights[0] = 0.5;
	n.layers[2].neurons[0].weights[1] = 0.5;
	n.layers[2].neurons[0].bias = -0.5;

	auto result = n.feedForward({1.0, 1.0});
	CHECK(result[0] == 0.5);
}

TEST_CASE("same in")
{
	Network n({4, 2, 2});

	std::vector<double> in { 0, 1, 0, 1 };
	std::vector<double> out { 1, 1 };

	for (int epoch = 0; epoch < 10; epoch++)
	{
		n.learnOnce(in, out);
	}

	const auto result = n.feedForward(in);
	CHECK(result[0] > 0.8);
	CHECK(result[1] > 0.8);
}

TEST_CASE("images")
{
	Network n({784, 30, 10});

	auto data = dataLoad();

	const auto& image = data.first.front();
	const auto& label = misc::vectorized<10>(data.second.front());

	for (int epoch = 0; epoch < 100; epoch++)
	{
		n.learnOnce(image, label);


	}

	const auto result = n.feedForward(image);
	CHECK(misc::argmax(result) == data.second.front());


}


Network::Network(const std::vector<size_t>& architecture, double learningRate)
	: learningRate(learningRate)
{
	for (std::size_t i = 0; i < architecture.size(); i++)
	{
		const std::size_t& layer_size = architecture[i];
		if (i == 0)
		{
			layers.push_back(Layer{layer_size});
		}
		else
		{
			const std::size_t& previous_layer_size = architecture[i - 1];
			layers.push_back(Layer{layer_size, previous_layer_size});
		}
	}
}

std::vector<size_t> Network::architecture() const
{
	std::vector<std::size_t> result;
	std::transform(layers.begin(), layers.end(), std::back_inserter(result), [](const auto& layer) { return layer.neurons.size(); });
	return result;
}

double Network::error(const std::vector<double>& input, const std::vector<double>& output)
{
	const auto result = feedForward(input);
	double sum_squared_error {};
	for (std::size_t i {}; i < output.size(); i++)
	{
		const double error = output[i] - result[i];
		sum_squared_error += error * error;
	}

	return std::sqrt(sum_squared_error / output.size());
}

std::vector<double> Network::feedForward(const std::vector<double>& input)
{
	layers[0].assignActivations(input);

	for (std::size_t l = 1; l < layers.size(); l++)
	{
		auto& current_layer_neurons = layers[l].neurons;
		auto& previous_layer_neurons = layers[l - 1].neurons;
		for (std::size_t n = 0; n < current_layer_neurons.size(); n++)
		{
			current_layer_neurons[n].z = {};
			for (std::size_t pn = 0; pn < previous_layer_neurons.size(); pn++)
			{
				current_layer_neurons[n].z += current_layer_neurons[n].weights[pn] * previous_layer_neurons[pn].activation;
			}
			current_layer_neurons[n].z += current_layer_neurons[n].bias;
			current_layer_neurons[n].activation = activationFunction(current_layer_neurons[n].z);
		}
	}

	return layers.back().activations();
}

void Network::learnOnce(const std::vector<double>& input, const std::vector<double>& output)
{
	feedForward(input);

	std::vector<DoublesVec> errors(layers.size());

	const auto& last_layer_neurons = layers.back().neurons;
	auto& last_layer_errors = errors.back();
	last_layer_errors.resize(layers.back().neurons.size());
	for (std::size_t n {}; n < last_layer_neurons.size(); n++)
	{
		last_layer_errors[n] = (last_layer_neurons[n].activation - output[n]) * activationFunctionDerivative(last_layer_neurons[n].z);
	}

	for (int l = layers.size() - 2; l >= 1; --l)
	{
		const auto& current_layer_neurons = layers[l].neurons;
		auto& current_layer_errors = errors[l];
		current_layer_errors.resize(current_layer_neurons.size());

		const auto& next_layer_neurons = layers[l + 1].neurons;
		const auto& next_layer_errors = errors[l + 1];

		for (std::size_t n {}; n < current_layer_neurons.size(); n++)
		{
			current_layer_errors[n] = {};
			for (std::size_t nn {}; nn < next_layer_neurons.size(); nn++)
			{
				current_layer_errors[n] += next_layer_neurons[nn].weights[n] * next_layer_errors[nn];
			}
			current_layer_errors[n] *= activationFunctionDerivative(current_layer_neurons[n].z);
		}
	}

	for (int l = layers.size() - 1; l >= 1; --l)
	{
		auto& current_layer_neurons = layers[l].neurons;
		const auto& current_layer_errors = errors[l];
		const auto& previous_layer_neurons = layers[l - 1].neurons;

		for (std::size_t n {}; n < current_layer_neurons.size(); n++)
		{
			auto& current_neuron = current_layer_neurons[n];


			for (std::size_t pn {}; pn < current_neuron.weights.size(); pn++)
			{
				const auto weight_correction = current_layer_errors[n] * previous_layer_neurons[pn].activation * learningRate;
				current_neuron.weights[pn] -= weight_correction;
			}

			const auto bias_correction = current_layer_errors[n] * learningRate;
			current_neuron.bias -= bias_correction;
		}
	}
}

Layer::Layer(size_t size, size_t previousLayerSize)
	: neurons{ misc::randomVector<Neuron>(size, [=]{ return Neuron(previousLayerSize); })}
{
}

void Layer::assignActivations(const std::vector<double>& activations)
{
	for (size_t i{}; i < activations.size(); ++i)
	{
		neurons[i].activation = activations[i];
	}
}

std::vector<double> Layer::activations() const
{
	std::vector<double> activations;
	activations.reserve(neurons.size());
	for (const auto& n : neurons)
	{
		activations.push_back(n.activation);
	}
	return activations;
}

Neuron::Neuron(size_t incomingWeights)
	: weights{ misc::randomNormalVector(incomingWeights) }
	, bias { misc::randomNormal() }
{
}
