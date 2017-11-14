#include "data.h"

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QShortcut>

#include <iostream>


std::pair<ImagesData, Labels> dataLoad()
{
	std::string data_directory = "d:/dev/cpp/handreco-data/";

	const auto labels = labelsFromFile(data_directory + "train-labels.idx1-ubyte");
	const auto images = imagesDataFromFile(data_directory + "train-images.idx3-ubyte");

	return { images, labels };

//	QApplication app(argc, argv);

//	auto label = new QLabel;
//	label->resize(ImageWidth, ImageHeight);

//	const auto pixmapFromImageData = [](const ImageData& image_data) {
//		QImage digit(image_data.data(), ImageWidth, ImageHeight, QImage::Format_Grayscale8);
//		return QPixmap::fromImage(digit);
//	};

//	auto s = new QShortcut(Qt::Key_Space, label);
//	QObject::connect(s, &QShortcut::activated, [=]{
//		static int i = 0;
//		label->setPixmap(pixmapFromImageData(images[i++]));
//	});


//	label->show();

//	return app.exec();
}
