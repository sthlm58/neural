#include "network.h"

#include <iterator>
#include <algorithm>
#include <numeric>


Network::Network(const Architecture& architecture, ActivationFunction activation, ActivationFunction activationDerivative, double learningRate)
	: activationFunction(std::move(activation))
	, activationFunctionDerivative(std::move(activationDerivative))
	, learningRate(learningRate)
{
	if (architecture.size() <= 1) throw std::runtime_error("Not enough layers");
	for (std::size_t i {}; i < architecture.size(); i++)
	{
		const std::size_t& previous_layer_size = (i > 0) ? architecture[i - 1] : 0;
		const std::size_t& layer_size = architecture[i];

		layers.push_back(Layer{layer_size, previous_layer_size});
	}
}
Architecture Network::architecture() const
{
	std::vector<std::size_t> result;
	std::transform(layers.begin(), layers.end(), std::back_inserter(result), [](const auto& layer) { return layer.neurons.size(); });
	return result;
}

double Network::error(const std::vector<double>& input, const std::vector<double>& output)
{
	const auto result = feedForward(input);

	double sum_squared_error = 0.;
	for (std::size_t i {}; i < result.size(); i++)
	{
		const double error = output[i] - result[i];
		sum_squared_error += error * error;
	};

	return std::sqrt(sum_squared_error / output.size());
}

std::vector<double> Network::feedForward(const std::vector<double>& input) const
{
	layers[0].applyActivations(input);

	for (std::size_t layer = 1; layer < layers.size(); layer++)
	{
		auto& current_layer_neurons = layers[layer].neurons;
		auto& previous_layer_neurons = layers[layer - 1].neurons;
		for (std::size_t n {}; n < current_layer_neurons.size(); n++)
		{
			current_layer_neurons[n].z = {};
			for (std::size_t pn {}; pn < previous_layer_neurons.size(); pn++)
			{
				current_layer_neurons[n].z += current_layer_neurons[n].weights[pn] * previous_layer_neurons[pn].activation;
			}
			current_layer_neurons[n].z += current_layer_neurons[n].bias;
			current_layer_neurons[n].activation = activationFunction(current_layer_neurons[n].z);
		}
	}

	return layers.back().activations();
}

void Network::calculateLastLayerError(const std::vector<double>& expected)
{
	const auto& last_layer_neurons = layers.back().neurons;
	auto& last_layer_errors = layers.back().errors;

	for (std::size_t n {}; n < last_layer_neurons.size(); n++)
	{
		last_layer_errors[n] = (last_layer_neurons[n].activation - expected[n]) * activationFunctionDerivative(last_layer_neurons[n].z);
	}
}

void Network::calculateInnerLayersError()
{
	for (int layer = layers.size() - 2; layer >= 1; --layer)
	{
		const auto& current_layer_neurons = layers[layer].neurons;
		auto& current_layer_errors = layers[layer].errors;
		current_layer_errors.resize(current_layer_neurons.size());

		const auto& next_layer_neurons = layers[layer + 1].neurons;
		const auto& next_layer_errors = layers[layer + 1].errors;

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
}

void Network::correctWeightAndBiases()
{
	for (int layer = layers.size() - 1; layer >= 1; --layer)
	{
		auto& current_layer_neurons = layers[layer].neurons;
		const auto& current_layer_errors = layers[layer].errors;
		const auto& previous_layer_neurons = layers[layer - 1].neurons;

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

void Network::learnOnce(const std::vector<double>& input, const std::vector<double>& output)
{
	feedForward(input);

	calculateLastLayerError(output);

	calculateInnerLayersError();

	correctWeightAndBiases();
}

Layer::Layer(size_t size, size_t previousLayerSize)
	: neurons( util::randomVector<Neuron>(size, [=]{ return Neuron(previousLayerSize); }) )
	, errors( size )
{
}

void Layer::applyActivations(const std::vector<double>& activations) const
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

Neuron::Neuron(size_t inputs)
	: weights{ util::randomNormalVector(inputs) }
	, bias { util::randomNormal() }
{
}

