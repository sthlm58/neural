#pragma once

#include "mnist_image_defs.h"

namespace mnist { namespace custom
{
	mnist::Data readImagesMatching(const std::string& dir, const std::string& pattern);
} }
