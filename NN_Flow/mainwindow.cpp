#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "netlayer.h"
#include <QImage>
#include <QRgb>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <QFileDialog>
#include <QMessageBox>
#include "opencv2/opencv.hpp"

#include "SignDetect.h"
#include "imgDetectHelper.h"


using namespace cv;
using namespace cv::ml;
using namespace std;


convLayer *layer1_X24_Y24_NX12_NY12;
convLayer *layer2_X12_Y12_NX5_NY5;
convLayer *layer3_X5_Y5_NX5_NY11;
NetNNT *nnDetect;
NetNNT *nn_Detect;

float dataImg[24 * 24 * 3]; // 3 канала изображения



static int writeDataToNet(Mat &imgSrc)
{
	int countPix = 0;
	int sizeY = 24, sizeX = 24;
	Vec3b intensity;
	uchar blue, green,red;

	// копирование данных в массив и нормировка к 1.0f
	for (int y = 0; y < sizeY; y++)
	{
		for (int x = 0; x < sizeX; x++)
		{
			intensity = imgSrc.at<Vec3b>(y, x);
			blue = intensity.val[0];
			green = intensity.val[1];
			red = intensity.val[2];

			dataImg[countPix] = (red - 127) / 128.0;
			dataImg[24*24 + countPix] = (green - 127) / 128.0;
			dataImg[24*24*2 + countPix] = (blue - 127) / 128.0;
			++countPix;
		}
	}
	return countPix;
}


static void initInputLayer(void)
{
	int tmp = 0;

	// запись указателей на даные в первый слой
	for (int i = 0; i < nn_Detect->layers[0]->neuronNum; i++)
	{
		for (int x = 0; x < nn_Detect->layers[0]->dendrNumX; x++)
		{
			for (int y = 0; y < nn_Detect->layers[0]->dendrNumY; y++)
			{
				tmp = x * nn_Detect->layers[0]->dendrNumX + y;
				nn_Detect->layers[0]->nn[i].ptrDendrDataInput[tmp] = &dataImg[tmp];
			}
		}
	}
}


static int netCalc(Mat *imgSrc, Rect *location)
{
	Mat imgSign = (*imgSrc)(*location);

	// запись данных в первый слой
	writeDataToNet(imgSign);

	// прямой проход
	nn_Detect->calc();

	return 1;
}


MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	//omp_set_num_threads(omp_get_num_procs()*2);

	ui->setupUi(this);

	nnDetect = new NetNNT();


	QFile openFile("C:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/NNTraningRoadSign/NNTraning/target10RoadSign.txt");
	openFile.open(QIODevice::ReadOnly | QIODevice::Text);
	if(!openFile.isOpen())
	{
		QMessageBox::warning(NULL, "df", "The file is not open, select the file from the training sample");
		on_actionLoad_traning_sample_triggered();
	}

	// обучающая выборка 10 символов по 500 образцов, всего N изображений
	// размерность изображения - 24*24, количество образцов - N, целей - 10
	inputDataBase = new traningSampleImg(NetNNT::DATA_IN_W,NetNNT::DATA_IN_H, NetNNT::NUM_OUT);

	if(inputDataBase->loadDataFromFile(&openFile) == -1)
	{
		QMessageBox::warning(this, "df", "Check the size of the sample. /n Parameter file, line # 2");
		return;
	}
	openFile.close();

	// прямой проход, тест
	// предъявление примера
	writeDataToLayer(0);

	nnDetect->calc();

	ui->spinBox->setMaximum(inputDataBase->getNumSample() - 1);
	ui->spinBox_2->setMaximum(nnDetect->layers[0]->neuronNum - 1);
	ui->spinBox_4->setMaximum(nnDetect->layers[1]->neuronNum - 1);
	ui->spinBox_5->setMaximum(nnDetect->layers[2]->neuronNum - 1);

	refresh_ui();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::on_pushButton_pressed()
{
	calculated = true;
	// целевые значения
	float *targetArr;
	float randFloat;

	int numSampleRand = ui->spinBox->value();
	int refreshRate = ui->spinBox_6->value();

	// установка скорости обучения
	for(int i = 0; i < nnDetect->NUM_LAYERS; i++)
	{
		nnDetect->layers[i]->setSpeedTrain(ui->doubleSpinBox->value());
	}

	srand((unsigned int)clock());

	int numEpoch = ui->spinBox_3->value();
	float meanMSE = 0;

	for(int k = ui->spinBox_3->value() * inputDataBase->getNumSample(); k > 0; k--)
	{
		// счетчик итераций (эпох)
		ui->spinBox_3->setValue(k / inputDataBase->getNumSample());

		//if(k%1 == 0)
		{	// генерация большого случайного числа
			qint64 randd = rand();
			randd = randd << 15;
			randd += rand();
			randFloat = (float)randd / (float)(0x3FFFFFFF);
			numSampleRand = (int)(randFloat * inputDataBase->getNumSample());
			//numSampleRand++;
		}
		if(numSampleRand >= inputDataBase->getNumSample())
		{
			numSampleRand = 0;
		}

		// загрузка целевых значений
		targetArr = inputDataBase->getTargetNum(numSampleRand);
		// запись данных в первый слой
		writeDataToLayer(numSampleRand);

		// прямой проход
		nnDetect->calc();
		//layer1_X28_Y28_NX14_NY14->directCalc();
		//layer2_X14_Y14_NX5_NY5->directCalc();
		//layer3_X5_Y5_NX5_NY1->directCalc();

		// вычисление ошибки на выходе сети
		for(int i = 0; i < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->neuronNum; i++)
		{
			//ui->textEdit->append("targrt: " + QString::number(targetArr[i]));
			nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].bkErr = (targetArr[i] - nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out);
		}

		nnDetect->tune();

/*
		for(int i = NetNNT::NUM_LAYERS - 2; i >= 0; i--)
		{
			// вычисление ошибок для весов элементов
			nnDetect->layers[i]->errWeightCalc();
			// вычисление ошибок сети
			nnDetect->layers[i]->errCalc();
		}
*/
		/*
		// вычисление ошибок для весов элементов
		layer3_X5_Y5_NX5_NY1->errWeightCalc();

		// вычисление ошибок сети
		layer2_X14_Y14_NX5_NY5->errCalc();

		// вычисление ошибок весов
		layer2_X14_Y14_NX5_NY5->errWeightCalc();

		// вычисление ошибок сети
		layer1_X28_Y28_NX14_NY14->errCalc();

		// вычисление ошибок весов
		layer1_X28_Y28_NX14_NY14->errWeightCalc();
		*/

		// коррекция (тут порядок не имеет значения)
		/*layer3_X5_Y5_NX5_NY1->tuneWeightBP(true);
		layer2_X14_Y14_NX5_NY5->tuneWeightBP(true);
		layer1_X28_Y28_NX14_NY14->tuneWeightBP(true);
*/
		// для отображения результата
		if(k%refreshRate == 0)
		{
			ui->spinBox->setValue(numSampleRand);
			on_spinBox_valueChanged(numSampleRand);
			refresh_ui();
			if(stopTraning)
			{
				stopTraning = false;
				break;
			}
		}

		// условие останова на каждой эпохе
		meanMSE += nnDetect->layers[NetNNT::NUM_LAYERS - 1]->RMSECalc(inputDataBase->getTargetNum(numSampleRand));
		if(k % inputDataBase->getNumSample() == 1)
		{
			meanMSE = meanMSE / inputDataBase->getNumSample();
			ui->lineEdit_16->setText(QString::number(meanMSE));
			if(meanMSE < (float)ui->doubleSpinBox_2->value())
			{
				break;
			}
		}
		numEpoch = ui->spinBox_3->value();

	}
	// прямой проход
	nnDetect->calc();
	/*
	layer1_X28_Y28_NX14_NY14->directCalc();
	layer2_X14_Y14_NX5_NY5->directCalc();
	layer3_X5_Y5_NX5_NY1->directCalc();
*/
	refresh_ui();

	calculated = false;
}

void MainWindow::writeDataToLayer(int numImg)
{
	// входные данные
    int tmp;
	for(int i = 0; i < nnDetect->layers[0]->neuronNum; i++)
    {
		for(int x = 0; x < nnDetect->layers[0]->dendrNumX; x++)
        {
			for(int y = 0; y < nnDetect->layers[0]->dendrNumY; y++)
            {
				tmp = x * nnDetect->layers[0]->dendrNumX + y;
				nnDetect->layers[0]->nn[i].ptrDendrDataInput[tmp] = &((inputDataBase->getImgNum(numImg))[tmp]);
            }
        }
    }
}

void MainWindow::on_spinBox_valueChanged_inputImage(int arg1)
{
	QImage *img = new QImage(inputDataBase->getSizeX(),inputDataBase->getSizeY(), QImage::Format_RGB32);
	float* dataImg = inputDataBase->getImgNum(arg1);
	int pixelR = 0;
	int pixelG = 0;
	int pixelB = 0;

	// max / min
	float minR = std::numeric_limits<float>::max();
	float maxR = std::numeric_limits<float>::min();
	float minG = std::numeric_limits<float>::max();
	float maxG = std::numeric_limits<float>::min();
	float minB = std::numeric_limits<float>::max();
	float maxB = std::numeric_limits<float>::min();

	int numWeight = inputDataBase->getSizeX() * inputDataBase->getSizeY();
	for(int i = 0; i < numWeight; i++)
	{
		if(minR > *dataImg) {minR = *dataImg;}
		if(maxR < *dataImg) {maxR = *dataImg;}
		dataImg++;
	}
	ui->lineEdit_4->setText(QString::number(maxR,'e',2));
	ui->lineEdit_3->setText(QString::number(minR,'e',2));

	numWeight = inputDataBase->getSizeX() * inputDataBase->getSizeY();
	for(int i = 0; i < numWeight; i++)
	{
		if(minG > *dataImg) {minG = *dataImg;}
		if(maxG < *dataImg) {maxG = *dataImg;}
		dataImg++;
	}

	numWeight = inputDataBase->getSizeX() * inputDataBase->getSizeY();
	for(int i = 0; i < numWeight; i++)
	{
		if(minB > *dataImg) {minB = *dataImg;}
		if(maxB < *dataImg) {maxB = *dataImg;}
		dataImg++;
	}

	int sizeColorLayer = inputDataBase->getSizeX()*inputDataBase->getSizeX();
	// отображение
	dataImg = inputDataBase->getImgNum(arg1);
	for(int i = 0; i < inputDataBase->getSizeX(); i++)
	{
		for(int j = 0; j < inputDataBase->getSizeY(); j++)
		{
			// данные для цветовых каналов расположены в памяти
			//последовательно по слоям, слой1 24*24 RedLayer,слой2 24*24 GreenLayer,слой3 24*24 BlueLayer,
			pixelR = (*dataImg + (-minR)) * 510 * (1 / (maxR - minR));
			pixelG = (*(dataImg + sizeColorLayer) + (-minG)) * 510 * (1 / (maxG - minG));
			pixelB = (*(dataImg + sizeColorLayer * 2) + (-minB)) * 510 * (1 / (maxB - minB));
			//pixelB = (*(dataImg + 24*24*2) + (-min)) * 510 * (1 / (max - min));
			//img->setPixel(j,i, grayToColorTemperature(pixel));
			img->setPixel(j,i, qRgb(pixelR, pixelG, pixelB));
			dataImg++;
		}
	}
	ui->label->setPixmap(QPixmap::fromImage(*img,Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	delete img;
}

void MainWindow::on_spinBox_valueChanged(int arg1)
{
	if(!calculated)
	{
		writeDataToLayer(arg1);
		nnDetect->calc();
	/*	layer1_X28_Y28_NX14_NY14->directCalc();
		layer2_X14_Y14_NX5_NY5->directCalc();
		layer3_X5_Y5_NX5_NY1->directCalc();
*/
		// обновление интерфейса
		refresh_ui();
	}
}

// обновление интерфейса
void MainWindow::refresh_ui()
{
	ui->textEdit->clear();
	on_spinBox_valueChanged_inputImage(ui->spinBox->value());
	on_spinBox_2_valueChanged(ui->spinBox_2->value());
	on_spinBox_3_valueChanged(ui->spinBox_3->value());
	on_spinBox_4_valueChanged(ui->spinBox_4->value());
	on_spinBox_5_valueChanged(ui->spinBox_5->value());
    show_L2_out();
    show_L3_out();

	for(int i = 0; i < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->neuronNum; i++)
	{
		ui->textEdit->append("out" + QString::number(i) + ": " + QString::number(nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out));
		ui->textEdit->append("err" + QString::number(i) + ": " + QString::number(nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].bkErr) + "/n");
		//ui->textEdit->append("gradErr:" + QString::number(layer3_X5_Y5_NX5_NY1->nn[i].bkErr) + "/n");
	}

	// отображение результата
	float result = NetNNT::NUM_OUT;
	for(int i = 0; i < NetNNT::NUM_OUT; i++)
	{
		result = (result > nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out) ? nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out : result;
	}
	for(int i = 0; i < NetNNT::NUM_OUT; i++)
	{
		if(result == nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out)
		{
			ui->lineEdit_15->setText(QString::number(i));
		}
	}
	//float MSE = layer3_X5_Y5_NX5_NY1->MSECalc(inputDataBase->getTargetNum(ui->spinBox->value()));
	//ui->lineEdit_16->setText(QString::number(MSE));

	qApp->processEvents();
}

// отображение весовых коэффициентов
void MainWindow::on_spinBox_2_valueChanged(int arg1)
{
	arg1 = (arg1 < nnDetect->layers[0]->neuronNum) ? arg1 : nnDetect->layers[0]->neuronNum;
	arg1 = arg1 * 3;

	//QImage *img = new QImage(nnDetect->layers[0]->dendrNumX, nnDetect->layers[0]->dendrNumY, QImage::Format_RGB32);
	QImage *imgR = new QImage(NetNNT::DATA_IN_H, NetNNT::DATA_IN_W, QImage::Format_RGB32);
	QImage *imgG = new QImage(NetNNT::DATA_IN_H, NetNNT::DATA_IN_W, QImage::Format_RGB32);
	QImage *imgB = new QImage(NetNNT::DATA_IN_H, NetNNT::DATA_IN_W, QImage::Format_RGB32);

	weight_Coeff* dataImgR = nnDetect->layers[0]->nn[arg1].weightCoeff;
	weight_Coeff* dataImgG = nnDetect->layers[0]->nn[arg1 + 1].weightCoeff;
	weight_Coeff* dataImgB = nnDetect->layers[0]->nn[arg1 + 2].weightCoeff;

	int pixel;

	// max / min
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	int numWeight = nnDetect->layers[0]->dendrNumX * nnDetect->layers[0]->dendrNumY;
	for(int i = 0; i < numWeight; i++)
	{
		min = (min > dataImgR->weight) ? dataImgR->weight : min;
		max = (max < dataImgR->weight) ? dataImgR->weight : max;
		dataImgR++;
	}
	ui->lineEdit->setText(QString::number(max));
	ui->lineEdit_2->setText(QString::number(min));

	// отображение
	dataImgR = nnDetect->layers[0]->nn[arg1].weightCoeff;
	dataImgG = nnDetect->layers[0]->nn[arg1 + 1].weightCoeff;
	dataImgB = nnDetect->layers[0]->nn[arg1 + 2].weightCoeff;

	// red
	for(int i = 0; i < NetNNT::DATA_IN_H; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_W; j++)
		{
			// нормализация
			pixel = (dataImgR->weight + (-min)) * 510 * (1 / (max - min));
			imgR->setPixel(j,i, grayToColorTemperature(pixel));
			dataImgR++;
		}
	}

	//green
	for(int i = 0; i < NetNNT::DATA_IN_H; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_W; j++)
		{
			// нормализация
			pixel = (dataImgG->weight + (-min)) * 510 * (1 / (max - min));
			imgG->setPixel(j,i, grayToColorTemperature(pixel));
			dataImgG++;
		}
	}

	//blue
	for(int i = 0; i < NetNNT::DATA_IN_H; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_W; j++)
		{
			// нормализация
			pixel = (dataImgB->weight + (-min)) * 510 * (1 / (max - min));
			imgB->setPixel(j,i, grayToColorTemperature(pixel));
			dataImgB++;
		}
	}

	ui->label_20->setPixmap(QPixmap::fromImage(*imgR,Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_19->setPixmap(QPixmap::fromImage(*imgG,Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_3->setPixmap( QPixmap::fromImage(*imgB,Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));

	delete imgR;
	delete imgG;
	delete imgB;
}

// отображение выхода сети
void MainWindow::on_spinBox_3_valueChanged(int arg1)
{
	QImage *imgR = new QImage(NetNNT::DATA_IN_W/2, NetNNT::DATA_IN_H/2, QImage::Format_RGB32);
	QImage *imgG = new QImage(NetNNT::DATA_IN_W/2, NetNNT::DATA_IN_H/2, QImage::Format_RGB32);
	QImage *imgB = new QImage(NetNNT::DATA_IN_W/2, NetNNT::DATA_IN_H/2, QImage::Format_RGB32);
	int pixel;

	// max / min
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	for(int i = 0; i < nnDetect->layers[0]->neuronNum; i++)
	{
		if(min > nnDetect->layers[0]->nn[i].out) {min = nnDetect->layers[0]->nn[i].out;}
		if(max < nnDetect->layers[0]->nn[i].out) {max = nnDetect->layers[0]->nn[i].out;}
	}
	ui->lineEdit_5->setText(QString::number(max));
	ui->lineEdit_6->setText(QString::number(min));

	// отображение
	// red
	for(int i = 0; i < NetNNT::DATA_IN_W/2; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_H/2; j++)
		{
			// нормализация
			pixel = (nnDetect->layers[0]->nn[i * (NetNNT::DATA_IN_W / 2) + j].out + (-min)) * 510 * (1 / (max - min));
			imgR->setPixel(j,i, grayToColorTemperature(pixel));
		}
	}

	int shiftDataColor = nnDetect->layers[1]->dendrNumY * nnDetect->layers[1]->dendrNumY;
	// green
	for(int i = 0; i < NetNNT::DATA_IN_W/2; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_H/2; j++)
		{
			// нормализация
			pixel = (nnDetect->layers[0]->nn[i * (NetNNT::DATA_IN_W / 2) + j + shiftDataColor].out + (-min)) * 510 * (1 / (max - min));
			imgG->setPixel(j,i, grayToColorTemperature(pixel));
		}
	}

	shiftDataColor = nnDetect->layers[1]->dendrNumY * nnDetect->layers[1]->dendrNumY * 2;
	// blue
	for(int i = 0; i < NetNNT::DATA_IN_W/2; i++)
	{
		for(int j = 0; j < NetNNT::DATA_IN_H/2; j++)
		{
			// нормализация
			pixel = (nnDetect->layers[0]->nn[i * (NetNNT::DATA_IN_W / 2) + j + shiftDataColor].out + (-min)) * 510 * (1 / (max - min));
			imgB->setPixel(j,i, grayToColorTemperature(pixel));
		}
	}

	ui->label_22->setPixmap(QPixmap::fromImage(*imgR, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_21->setPixmap(QPixmap::fromImage(*imgG, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_5->setPixmap( QPixmap::fromImage(*imgB, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));

	delete imgR;
	delete imgG;
	delete imgB;
}

// отображение весовых коэффициентов
void MainWindow::on_spinBox_4_valueChanged(int arg1)
{
	QImage *imgR = new QImage(NetNNT::DATA_IN_H/2,NetNNT::DATA_IN_W/2, QImage::Format_RGB32);
	QImage *imgG = new QImage(NetNNT::DATA_IN_H/2,NetNNT::DATA_IN_W/2, QImage::Format_RGB32);
	QImage *imgB = new QImage(NetNNT::DATA_IN_H/2,NetNNT::DATA_IN_W/2, QImage::Format_RGB32);

	int pixel;
	weight_Coeff* dataImg = nnDetect->layers[1]->nn[arg1].weightCoeff;
	// max / min
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	int numWeight = nnDetect->layers[1]->dendrNumX * nnDetect->layers[1]->dendrNumY;
	for(int i = 0; i < numWeight; i++)
	{
		min = (min > dataImg->weight) ? dataImg->weight : min;
		max = (max < dataImg->weight) ? dataImg->weight : max;
		dataImg++;
	}
	ui->lineEdit_12->setText(QString::number(max));
	ui->lineEdit_8->setText(QString::number(min));

	// отображение
	dataImg = nnDetect->layers[1]->nn[arg1].weightCoeff;

	// для 3х каналов RGB
	// red
	for(int i = 0; i < nnDetect->layers[1]->dendrNumX / 3; i++)
	{
		// отображение данных по цветам
		for(int j = 0; j < nnDetect->layers[1]->dendrNumY; j++)
		{
			// нормализация Red
			pixel = (dataImg->weight + (-min)) * 510 * (1 / (max - min));
			imgR->setPixel(j, i, grayToColorTemperature(pixel));
			dataImg++;
		}
	}

	// green
	for(int i = 0; i < nnDetect->layers[1]->dendrNumX / 3; i++)
	{
		// отображение данных по цветам
		for(int j = 0; j < nnDetect->layers[1]->dendrNumY; j++)
		{
			// Green
			pixel = (dataImg->weight + (-min)) * 510 * (1 / (max - min));
			imgG->setPixel(j, i, grayToColorTemperature(pixel));
			dataImg++;
		}
	}

	// blue
	for(int i = 0; i < nnDetect->layers[1]->dendrNumX / 3; i++)
	{
		// отображение данных по цветам
		for(int j = 0; j < nnDetect->layers[1]->dendrNumY; j++)
		{
			// Green
			pixel = (dataImg->weight + (-min)) * 510 * (1 / (max - min));
			imgB->setPixel(j, i, grayToColorTemperature(pixel));
			dataImg++;
		}
	}


	// копирование данных в UI
	ui->label_24->setPixmap(QPixmap::fromImage(*imgR, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_23->setPixmap(QPixmap::fromImage(*imgG, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	ui->label_6->setPixmap( QPixmap::fromImage(*imgB, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));

	delete imgR;
	delete imgG;
	delete imgB;
}

// отображение выхода сети
void MainWindow::show_L2_out()
{
	QImage *img = new QImage(5,5, QImage::Format_RGB32);
	int pixel;

	// max / min
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	for(int i = 0; i < nnDetect->layers[1]->neuronNum; i++)
	{
		if(min > nnDetect->layers[1]->nn[i].out) {min = nnDetect->layers[1]->nn[i].out;}
		if(max < nnDetect->layers[1]->nn[i].out) {max = nnDetect->layers[1]->nn[i].out;}
	}
	ui->lineEdit_10->setText(QString::number(max));
	ui->lineEdit_7->setText(QString::number(min));

	// отображение
	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 5; j++)
		{
			// нормализация
			pixel = (nnDetect->layers[1]->nn[i*5 + j].out + (-min)) * 510 * (1 / (max - min));
			img->setPixel(j,i, grayToColorTemperature(pixel));
		}
	}

	ui->label_9->setPixmap(QPixmap::fromImage(*img, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
	delete img;
}

// отображение выхода сети
void MainWindow::show_L3_out()
{
	int numOut = NetNNT::NUM_OUT;
	QImage *img = new QImage(numOut, 1, QImage::Format_RGB32);
    int pixel;

    // max / min
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
	for(int i = 0; i < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->neuronNum; i++)
    {
		if(min > nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out) {min = nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out;}
		if(max < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out) {max = nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i].out;}
    }
	ui->lineEdit_11->setText(QString::number(max));
	ui->lineEdit_9->setText(QString::number(min));

    // отображение
	for(int i = 0; i < 1; i++)
    {
		for(int j = 0; j < numOut; j++)
        {
            // нормализация
			pixel = (nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[i*numOut + j].out + (-min)) * 510 * (1 / (max - min));
			img->setPixel(j,i, grayToColorTemperature(pixel));
        }
    }

	ui->label_10->setPixmap(QPixmap::fromImage(*img, Qt::AutoColor).scaled(56, 56, Qt::IgnoreAspectRatio, Qt::FastTransformation));
    delete img;
}

// отображение весовых коэффициентов
void MainWindow::on_spinBox_5_valueChanged(int arg1)
{
	QImage *img = new QImage(5,5, QImage::Format_RGB32);
    int pixel;
	weight_Coeff* dataImg = nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[arg1].weightCoeff;
	// max / min
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
	int numWeight = nnDetect->layers[NetNNT::NUM_LAYERS - 1]->dendrNumX * nnDetect->layers[NetNNT::NUM_LAYERS - 1]->dendrNumY;
	for(int i = 0; i < numWeight; i++)
	{
		min = (min > dataImg->weight) ? dataImg->weight : min;
		max = (max < dataImg->weight) ? dataImg->weight : max;
		dataImg++;
	}
	ui->lineEdit_14->setText(QString::number(max));
	ui->lineEdit_13->setText(QString::number(min));

	// отображение
	dataImg = nnDetect->layers[NetNNT::NUM_LAYERS - 1]->nn[arg1].weightCoeff;
	for(int i = 0; i < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->dendrNumX; i++)
	{
		for(int j = 0; j < nnDetect->layers[NetNNT::NUM_LAYERS - 1]->dendrNumY; j++)
		{
			// нормализация
			pixel = (dataImg->weight + (-min)) * 510 * (1 / (max - min));
			img->setPixel(j,i, grayToColorTemperature(pixel));
			dataImg++;
		}
	}

    ui->label_8->setPixmap(QPixmap::fromImage(*img, Qt::AutoColor).scaled(56,56,Qt::IgnoreAspectRatio, Qt::FastTransformation));
    delete img;
}

// остановка обучения
void MainWindow::on_pushButton_2_pressed()
{
	stopTraning = true;

	refresh_ui();
}

int MainWindow::grayToColorTemperature(int color)
{
	if(color < 256)
	{
		return (int)qRgb(color, 0, 255 - color);
	}
	else if(color > 255)
	{
		return (int)qRgb(255, color - 255, 0);
	}
}


void MainWindow::on_actionSave_network_configuration_triggered()
{
	QFileDialog dialogOpenF(this, "Save file", QApplication::applicationDirPath(), "bin files (*.bin)");
	dialogOpenF.setAcceptMode(QFileDialog::AcceptSave);
	if(!dialogOpenF.exec()){ return; }

	QFile openFile(dialogOpenF.selectedFiles().back());
	openFile.open(QIODevice::WriteOnly);
	if(!openFile.isOpen())
	{
		QMessageBox::warning(NULL, "df", "Файл не открыт");
	}

	nnDetect->saveToFile(&openFile);

	openFile.close();
}

void MainWindow::on_actionLoad_network_configuration_triggered()
{
	QFileDialog dialogOpenF(this, "Open file", QApplication::applicationDirPath(), "bin files (*.bin)");
	dialogOpenF.setAcceptMode(QFileDialog::AcceptOpen);
	if(!dialogOpenF.exec()){ return; }

	QFile openFile(dialogOpenF.selectedFiles().back());
	openFile.open(QIODevice::ReadOnly);
	if(!openFile.isOpen())
	{
		QMessageBox::warning(NULL, "df", "Файл не открыт");
	}

	nnDetect->loadFromFile(&openFile);

	openFile.close();

	refresh_ui();
}

void MainWindow::on_actionLoad_traning_sample_triggered()
{
	QFileDialog dialogOpenF(this, "Open file", QApplication::applicationDirPath(), "text files (*.txt)");
	dialogOpenF.setAcceptMode(QFileDialog::AcceptOpen);
	if(!dialogOpenF.exec()){ return; }

	QFile openFile(dialogOpenF.selectedFiles().back());
	openFile.open(QIODevice::ReadOnly | QIODevice::Text);
	if(!openFile.isOpen())
	{
		QMessageBox::warning(NULL, "df", "Файл не открыт");
	}
	inputDataBase->loadDataFromFile(&openFile);
	openFile.close();
}





// проверка работы классификатора
void MainWindow::on_pushButton_3_clicked()
{
	// HOG детектор
	SignDetect detectorSign;
	// Дополнительный класс для HOG детектора
	imgDetectHelper detectorHelper(&detectorSign);

	nn_Detect = new NetNNT();
	initInputLayer();

	// загрузка конфигурации сети из файла
/*	QFileDialog dialogOpenF(this, "Open file", QApplication::applicationDirPath(), "bin files (*.bin)");
	dialogOpenF.setAcceptMode(QFileDialog::AcceptOpen);
	if(!dialogOpenF.exec()){ return; }

	QFile openFile(dialogOpenF.selectedFiles().back());
*/
	QFile openFile("c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/NNTraningRoadSign/NNTraning/release/net_conf_24_24_sign_recognition_v6_700E_01_1000E_005_500E_01.bin");
	openFile.open(QIODevice::ReadOnly);
	if(!openFile.isOpen())
	{
		QMessageBox::warning(NULL, "df", "Файл не открыт");
	}

	// загрузка данных из файла
	nn_Detect->loadFromFile(&openFile);

	nnDetect->loadFromFile(&openFile);

	openFile.close();

	// проверка правильности детектирования
	Mat imgTest = imread("c:/Users/Ilya/Documents/Projects/DATA/road_sign/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/00012/00000_00029_1.bmp");

	namedWindow("inputImage", WINDOW_FREERATIO);
	imshow("inputImage", imgTest);

	cout << "dataImg:" << endl;
	for(int k = 1; k < 24*24*3+1; k++)
	{
		cout << setprecision(3) << dataImg[k] << " ";
		if((k%24) == 0){cout << endl;}
		if((k%(24*24)) == 0){cout << endl;}
	}
	cout << endl;

	netCalc(&imgTest, &Rect(0,0,imgTest.size().width, imgTest.size().height));

	writeDataToLayer(0);
	nnDetect->calc();

	int countTst = 0;
	// проверка весов
	for(int ll = 0; ll < 3; ll++)
	{
		for(int k = 0; k < nnDetect->layers[ll]->neuronNum; k++)
		{
			for(int m = 0; m < nnDetect->layers[ll]->neuronNum; m++)
			{
				float originNN = nnDetect->layers[ll]->nn[m].out;
				float testnetNN = nn_Detect->layers[ll]->nn[m].out;

				if(originNN != testnetNN)
				{
					cout << "err\n";
				}

				for(int l = 0; l < nnDetect->layers[ll]->dendrNumX * nnDetect->layers[ll]->dendrNumY + 1; l++)
				{
					float origin = *(nnDetect->layers[ll]->nn[m].ptrDendrDataInput[l]);
					float testnet = *(nn_Detect->layers[ll]->nn[m].ptrDendrDataInput[l]);

					float originW = nnDetect->layers[ll]->nn[m].weightCoeff[l].weight;
					float testnetW = nn_Detect->layers[ll]->nn[m].weightCoeff[l].weight;

					countTst++;
					if((origin != testnet) || (originW != testnetW))
					{
						cout << "err\n";
					}
				}
			}
		}
	}




	for(int k = 1; k < 24*24*3+1; k++)
	{
		cout << setprecision(3) << dataImg[k] << " ";
		if((k%24) == 0){cout << endl;}
		if((k%(24*24)) == 0){cout << endl;}
	}


	cout << "sample data \'input database\'" << endl;


	for(int k = 1; k < 24*24*3+1; k++)
	{
		cout << setprecision(3) << inputDataBase->sampleData[k] << " ";
		if((k%24) == 0){cout << endl;}
		if((k%(24*24)) == 0){cout << endl;}
	}








	for(int k = 0; k < 11; k++)
	{
		cout << "out" << k+1 << ": " << nn_Detect->layers[2]->nn[k].out << endl;
	}

	nnDetect->calc();

	for(int k = 0; k < 11; k++)
	{
		cout << "out" << k+1 << ": " << nnDetect->layers[2]->nn[k].out << endl;
	}

	cout << endl << "compare weight" << endl;
	for(int k = 0; k < 24*24*3; k++)
	{
		cout << setprecision(3) << nnDetect->layers[0]->nn[k].ptrDendrDataInput[0][k] << " ";
		cout << setprecision(3) << nn_Detect->layers[0]->nn[k].ptrDendrDataInput[0][k] << "   ";
		if((k%24) == 0){cout << endl;}
		if((k%(24*24)) == 0){cout << endl;}
	}





	// загрузка изображения
	Mat inputImage = imread("C:/Users/Ilya/Documents/Projects/DATA/road_sign/8ss97PaUOLA.jpg");
	if (inputImage.empty())
	{
		cerr << "load image is corrupt" << endl;
		getchar();
	}

	cv::resize(inputImage, inputImage, Size(), 0.5, 0.5);

	// передача изображения в детектор
	detectorSign.setImage(&inputImage, false);
getchar();
	int key = 0;
	while(1)
	{
		qApp->processEvents();
		detectorSign.compute();
		detectorHelper.drawLocationsWeight(inputImage);
		detectorHelper.printStatus();
		cout << endl;

		imshow("show", inputImage);
		key = waitKey(1);
		if (key == 27)
		{
			cout << "/nclose";
			return;
		}
	}

	return;





	cv::Mat tstImg = cv::imread("c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/NET_SVM_HOG_SIGN_RECOGNITION/lena.png");
	cv::namedWindow ("Image processing", cv::WINDOW_FREERATIO);
	cv::imshow("Image processing", tstImg);
	cv::waitKey();
}
