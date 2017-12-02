#include "static_network.h"
#include "network.h"
#include "network_runner.h"

#include <QApplication>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QShortcut>
#include "mnist_reader.h"

int main(int argc, char* argv[])
{
	run_dynamic_network();
}

//	QApplication app(argc, argv);

//	auto data = mnist::readTrainingData("d:/dev/cpp/handreco-data/");


//	auto label = new QLabel;
//	label->resize(mnist::ImageWidth, mnist::ImageHeight);

//	const auto pixmapFromImageData = [](const mnist::ImageData& image_data) {
//		std::vector<unsigned char> data(image_data.size(), 0);
//		std::transform(image_data.begin(), image_data.end(), data.begin(), [](double val) { return val * 255; });
//		QImage digit(data.data(), mnist::ImageWidth, mnist::ImageHeight, QImage::Format_Grayscale8);
//		return QPixmap::fromImage(digit);
//	};

//	auto s = new QShortcut(Qt::Key_Space, label);
//	QObject::connect(s, &QShortcut::activated, [=]{
//		static int i = 0;
//		auto pix = pixmapFromImageData(data.images[i++]);
//		label->setPixmap(pix);
//		pix.save(QString("d:/%1.png").arg(i, 2, 10, QChar('0')));
//	});


//	label->show();

//	return app.exec();
//}
