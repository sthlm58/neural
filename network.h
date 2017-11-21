#pragma once

#include "util.h"

#include <vector>
#include <functional>

using ActivationFunction = std::function<double(double)>;
using Architecture = std::vector<std::size_t>;


struct Neuron
{
	Neuron(std::size_t inputs = 0);

	double activation {};
	double z {};

	std::vector<double> weights {};
	double bias {};

};


struct Layer
{
	Layer(std::size_t size = 0, std::size_t previousLayerSize = 0);

	void applyActivations(const std::vector<double>& activations);
	std::vector<double> activations() const;

	std::vector<Neuron> neurons {};
	std::vector<double> errors {};
};


class Network
{
public:

	Network(const Architecture& architecture,
			ActivationFunction activation = &util::identity,
			ActivationFunction activationDerivative = &util::identityPrime,
			double learningRate = 0.3);

	Architecture architecture() const;
	double error(const std::vector<double>& input, const std::vector<double>& output);

	std::vector<double> feedForward(const std::vector<double>& input);
	void learnOnce(const std::vector<double>& input, const std::vector<double>& expected);

	std::vector<Layer> layers {};

protected:
	void calculateLastLayerError(const std::vector<double>& expected);
	void calculateInnerLayersError();
	void correctWeightsAndBiases();

private:

	ActivationFunction activationFunction {};
	ActivationFunction activationFunctionDerivative {};
	double learningRate {};

};
