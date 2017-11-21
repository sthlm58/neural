#include "mnist_custom_reader.h"

#include <QDir>
#include <QImage>

#include <iostream>

namespace
{
	std::pair<mnist::ImageData, bool> readSingleImage(const QString& path)
	{
		mnist::ImageData pixels;

		QImage image(path);
		if (image.width() != mnist::ImageWidth || image.height() != mnist::ImageHeight)
		{
			return { pixels, false };
		}

		pixels.reserve(mnist::Data::Inputs);
		for (int y = 0; y < image.height(); y++)
		{
			for (int x = 0; x < image.width(); x++)
			{
				pixels.push_back(qGray(image.pixel(x, y)) / 255.);
			}
		}

		return { pixels, true };

	}
}

namespace mnist { namespace custom
{
	mnist::Data readImagesMatching(const std::string& dir, const std::string& pattern)
	{
		mnist::Data result;

		auto matches = QDir(QString::fromStdString(dir)).entryInfoList({ QString::fromStdString(pattern) });
		for (const auto& entry : matches)
		{
			auto data = readSingleImage(entry.absoluteFilePath());
			if (auto ok = data.second)
			{
				std::cout << "Image read: " << entry.absoluteFilePath().toStdString() << std::endl;;
				result.images.push_back(data.first);
				result.labels.push_back(QFileInfo(entry).fileName().left(1).toInt());
			}
		}


		return result;
	}

} }
