#pragma once

#include "mnist_reader.h"
#include "misc.h"

#include <vector>
#include <functional>

using ActivationFunction = std::function<double(double)>;
using Architecture = std::vector<std::size_t>;


struct Neuron
{
	Neuron(std::size_t inputs = 0);

	mutable double activation {};
	mutable double z {};

	std::vector<double> weights {};
	double bias {};

};


struct Layer
{
	Layer(std::size_t size = 0, std::size_t previousLayerSize = 0);

	void applyActivations(const std::vector<double>& activations) const;
	std::vector<double> activations() const;

	std::vector<Neuron> neurons {};
};


class Network
{
public:

	Network(const Architecture& architecture,
			ActivationFunction activation = &misc::identity,
			ActivationFunction activationDerivative = &misc::identityPrime,
			double learningRate = 0.3);

	Architecture architecture() const;
	double error(const std::vector<double>& input, const std::vector<double>& output);

	std::vector<double> feedForward(const std::vector<double>& input) const;
	void learnOnce(const std::vector<double>& input, const std::vector<double>& output);

	std::vector<Layer> layers {};



protected:
	void correctWeightAndBiases(std::vector<std::vector<double>>& errors);
	void calculateInnerLayersError(std::vector<std::vector<double> >& errors);
	void calculateLastLayerError(std::vector<std::vector<double> >& errors, const std::vector<double>& output);

private:

	ActivationFunction activationFunction {};
	ActivationFunction activationFunctionDerivative {};
	double learningRate {};

};
