#pragma once

#include "mnist_image_defs.h"

namespace mnist
{
	mnist::Data readTrainingData(const std::string& directory);
	mnist::Data readTestData(const std::string& directory);
}
