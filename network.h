#pragma once

#include "data.h"
#include "misc.h"

#include <vector>
#include <functional>


class Neuron
{
public:
	double activation {};
	double z {};
	std::vector<double> weights {};
	double bias {};

	Neuron(std::size_t incomingWeights = 0);


};


struct Layer
{
	std::vector<Neuron> neurons {};

	Layer(std::size_t size = 0, std::size_t previousLayerSize = 0);

	void assignActivations(const std::vector<double>& activations);

	std::vector<double> activations() const;
};


class Network
{
public:
	/**
	 * @brief Network
	 * @param architecture { 784, 30, 10 }
	 */
	Network(const std::vector<std::size_t>& architecture, double learningRate = 0.3);

	std::vector<std::size_t> architecture() const;

	double learningRate {0.3};
	std::vector<Layer> layers {};
	std::function<double(double)> activationFunction {misc::leakyRelu};
	std::function<double(double)> activationFunctionDerivative { misc::leakyReluPrime };

	double error(const std::vector<double>& input, const std::vector<double>& output);

	/** input has to be same size as first layer */
	std::vector<double> feedForward(const std::vector<double>& input);

	void learnOnce(const std::vector<double>& input, const std::vector<double>& output);



};
