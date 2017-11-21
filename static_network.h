#pragma once

#include "util.h"

#include <array>


template <std::size_t Inputs>
struct StaticNeuron
{
	static constexpr std::size_t Inputs = Inputs;

	double activation {};
	double z {};

	std::array<double, Inputs> weights {};
	double bias {};
};

template <std::size_t Size, std::size_t PreviousSize>
struct StaticLayer
{
	static const std::size_t Size = Size;
	static const std::size_t PreviousSize = PreviousSize;

	std::array<StaticNeuron<PreviousSize>, Size> neurons;
	std::array<double, Size> errors;

	template <typename ActivationsType>
	void applyActivations(const ActivationsType& activations)
	{
		for (size_t i{}; i < activations.size(); ++i)
		{
			neurons[i].activation = activations[i];
		}
	}
	std::array<double, Size> activations()
	{
		std::array<double, Size> activations;
		for (size_t i{}; i < activations.size(); ++i)
		{
			activations[i] = neurons[i].activation;
		}
		return activations;
	}
};

using ActivationFunctionType = double (*)(double);
template <std::size_t Inputs, std::size_t Hidden, std::size_t Outputs, ActivationFunctionType ActivationFunc, ActivationFunctionType ActivationFuncDerivative, int LearningRate>
class StaticNetwork
{
public:
	enum LayerIds { INPUT_LAYER = 0, HIDDEN_LAYER = 1, OUTPUT_LAYER = 2 };


	StaticNetwork()
	{

		const auto randomize = [](auto& layer)
		{
			for (std::size_t i {}; i < layer.neurons.size(); i++)
			{
				auto& neuron = layer.neurons[i];
				for (std::size_t j {}; j < neuron.weights.size(); j++)
				{
					neuron.weights[j] = util::randomNormal();
				}
				neuron.bias = util::randomNormal();
			}
		};
		apply_on_layer<INPUT_LAYER>(randomize);
		apply_on_layer<HIDDEN_LAYER>(randomize);
		apply_on_layer<OUTPUT_LAYER>(randomize);
	}


	std::array<double, Outputs> feedForward(const std::vector<double>& input)
	{
		std::get<0>(layers).applyActivations(input);

		const auto feedForwardLayer = [](auto& layer, auto& next_layer)
		{
			auto& current_layer_neurons = next_layer.neurons;
			auto& previous_layer_neurons = layer.neurons;
			for (std::size_t n {}; n < current_layer_neurons.size(); n++)
			{
				current_layer_neurons[n].z = {};
				for (std::size_t pn {}; pn < previous_layer_neurons.size(); pn++)
				{
					current_layer_neurons[n].z += current_layer_neurons[n].weights[pn] * previous_layer_neurons[pn].activation;
				}
				current_layer_neurons[n].z += current_layer_neurons[n].bias;
				auto z = current_layer_neurons[n].z;
				current_layer_neurons[n].activation = ActivationFunc(current_layer_neurons[n].z);
			}
		};

		apply_on_layer_and_next<0>(feedForwardLayer);
		apply_on_layer_and_next<1>(feedForwardLayer);

		return std::get<OUTPUT_LAYER>(layers).activations();

	}

	template <typename InputContainer, typename ExpectedContainer>
	void learnOnce(const InputContainer& input, const ExpectedContainer& expected)
	{
		feedForward(input);

		apply_on_layer<OUTPUT_LAYER>([&expected](auto& layer)
		{
			const auto& last_layer_neurons = layer.neurons;
			auto& last_layer_errors = layer.errors;

			for (std::size_t n {}; n < last_layer_neurons.size(); n++)
			{
				last_layer_errors[n] = (last_layer_neurons[n].activation - expected[n]) * ActivationFuncDerivative(last_layer_neurons[n].z);
			}
		});

		const auto calculateInnerError = [](auto& layer, auto& next_layer)
		{
			const auto& current_layer_neurons = layer.neurons;
			auto& current_layer_errors = layer.errors;

			const auto& next_layer_neurons = next_layer.neurons;
			const auto& next_layer_errors = next_layer.errors;

			for (std::size_t n {}; n < current_layer_neurons.size(); n++)
			{
				current_layer_errors[n] = {};
				for (std::size_t nn {}; nn < next_layer_neurons.size(); nn++)
				{
					current_layer_errors[n] += next_layer_neurons[nn].weights[n] * next_layer_errors[nn];
				}
				current_layer_errors[n] *= ActivationFuncDerivative(current_layer_neurons[n].z);
			}
		};

		apply_on_layer_and_next<HIDDEN_LAYER>(calculateInnerError);
		apply_on_layer_and_next<INPUT_LAYER>(calculateInnerError);

		const auto correctWeights = [](auto& layer, auto& next_layer)
		{
			auto& current_layer_neurons = next_layer.neurons;
			const auto& current_layer_errors = next_layer.errors;
			const auto& previous_layer_neurons = layer.neurons;

			for (std::size_t n {}; n < current_layer_neurons.size(); n++)
			{
				auto& current_neuron = current_layer_neurons[n];

				for (std::size_t pn {}; pn < current_neuron.weights.size(); pn++)
				{
					const auto weight_correction = current_layer_errors[n] * previous_layer_neurons[pn].activation * LearningRate / 1000.;
					current_neuron.weights[pn] -= weight_correction;
				}

				const auto bias_correction = current_layer_errors[n] * LearningRate / 1000.;
				current_neuron.bias -= bias_correction;
			}
		};

		apply_on_layer_and_next<HIDDEN_LAYER>(correctWeights);
		apply_on_layer_and_next<INPUT_LAYER>(correctWeights);


	}

	std::tuple<
		StaticLayer<Inputs, 0>,
		StaticLayer<Hidden, Inputs>,
		StaticLayer<Outputs, Hidden>> layers;

	template <std::size_t I, typename Function>
	void apply_on_layer(Function func)
	{
		func(std::get<I>(layers));
	}
	template <std::size_t I, typename Function>
	void apply_on_layer_and_next(Function func)
	{
		func(std::get<I>(layers), std::get<I + 1>(layers));
	}

};


