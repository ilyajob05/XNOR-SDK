#include "netlayer.h"
#include <qmath.h>
#include <QImage>
#include <QRgb>
#include <QStringList>
#include <time.h>
#include <QList>
#include <dialogprogressbar.h>
#include <QApplication>


int traningSampleImg::loadDataFromFile(QFile *file)
{
	DialogProgressBar dialogStatus;

	QImage img;

	float *dataPtr = NULL;
	float *dataTarget = NULL;

	int countSample = 0;

	QStringList listTarget;

	// чтение файла
	while (!file->atEnd())
	{
		QString line = file->readLine();

		if(line == "\n")
		{
			continue;
		}
		else if(line.contains("target"))
		{
			// загрузка целей
			line = file->readLine();
			listTarget = line.split(' ');

            if(listTarget.size() != numTarget)
			{
				return -1;
			}
			continue;
		}
		else if(line.contains("number of samples"))
		{	// количество образцов
			line = file->readLine();
			numSample = line.toInt();
			dialogStatus.setNumItem(numSample);
			dialogStatus.putStr("Memory allocate...");
			dialogStatus.show();
			qApp->processEvents();

			// выделение памяти для обучающей выборки
			sampleData = new float[sizeX * sizeY * 3 * numSample]; //TODO: сдеать очистку памяти после неудачной загрузки данных
			targetData = new float[numTarget * numSample];
			dataPtr = sampleData;
			dataTarget = targetData;
			continue;
		}

		// запись целевых значений
		for(int i = 0; i < listTarget.size(); i++)
		{
			*dataTarget = listTarget.at(i).toFloat();
			dataTarget++;
		}

		line.resize(line.size() - 1); // удаление перевода строки для строки с именем файла
		// загрузка изображения
		img = QImage(line);

		img = img.scaled(sizeY, sizeX);

		// проверка размера
		if((img.size().height() != sizeY) || (img.size().width() != sizeX))
		{
			return -1;
		}	// ERROR

		// копирование данных и нормировка к 1.0f
		for(int y = 0; y < sizeY; y++)
		{
			for(int x = 0; x < sizeX; x++)
			{
				*dataPtr = (qRed(img.pixel(x,y))   - 127) / 128.0;
				dataPtr++;
			}
		}

		for(int y = 0; y < sizeY; y++)
		{
			for(int x = 0; x < sizeX; x++)
			{
				*dataPtr = (qGreen(img.pixel(x,y)) - 127) / 128.0;
				dataPtr++;
			}
		}

		for(int y = 0; y < sizeY; y++)
		{
			for(int x = 0; x < sizeX; x++)
			{
				*dataPtr = (qBlue(img.pixel(x,y))  - 127) / 128.0;
				dataPtr++;
			}
		}

		countSample++;

		dialogStatus.setCurrentNumItem(countSample);
		dialogStatus.putStr(line);
		qApp->processEvents();
	}

	dialogStatus.close();

	if(countSample != numSample)
		return -1;
	else
		return 1;
}


float* traningSampleImg::getImgNum(int num)
{
	return &(sampleData[num * sizeX * sizeY * 3]);
}

float* traningSampleImg::getTargetNum(int num)
{
	return &(targetData[num * numTarget]);
}

float* traningSampleImg::getSampleDataPtr()
{
	return sampleData;
}

float* traningSampleImg::getTargetDataPtr()
{
	return targetData;
}

int traningSampleImg::getNumTarget()
{
	return numTarget;
}

int traningSampleImg::getSizeX()
{
	return sizeX;
}

int traningSampleImg::getSizeY()
{
	return sizeY;
}

int traningSampleImg::getNumSample()
{
	return numSample;
}

traningSampleImg::traningSampleImg(int size_X, int size_Y, int num_Target)
{
    sizeX = size_X;
    sizeY = size_Y;
    numTarget = num_Target;
}

traningSampleImg::~traningSampleImg()
{
    delete[] sampleData;
    delete[] targetData;
}


convLayer::convLayer(int nNum, int dendrNum_X, int dendrNum_Y, int numOut_Weight):
dendrNumX(dendrNum_X), dendrNumY(dendrNum_Y), neuronNum(nNum), numOutWeight(numOut_Weight)
{
	previousLayer = nullptr;
	nextLayer = nullptr;

	// добавление нейронов
	nn = new neurStruct[neuronNum];

	//добавление весов и указателей к нейронам
	int numDendr = dendrNumX * dendrNumY;

	for(int i = 0; i < neuronNum; i++)
	{
		nn[i].out = i;
		nn[i].sum = i;
		nn[i].bkErr = i;

		nn[i].weightCoeff = new weight_Coeff[numDendr + 1];			// 1 - bias
		nn[i].ptrDendrDataInput = new float* [numDendr + 1];
		nn[i].ptrDendrDataInput[numDendr] = &biasConst;					// 1 - bias
		nn[i].ptrOutWeightCoeff = new weight_Coeff* [numOut_Weight];	// 1 - bias
	}
}

convLayer::~convLayer()
{
	for(int i = 0; i < neuronNum; i++)
	{
		delete[] nn[i].weightCoeff;
		delete[] nn[i].ptrDendrDataInput;
		delete[] nn[i].ptrOutWeightCoeff;
	}
	delete[] nn;
}

void convLayer::setSpeedTrain(float val)
{
	speedTrain = val;
}

int convLayer::initWeight(void)
{
	float randNorm = 0;
	bool eval = true;
	float shift = 0.5;

	int numDendr = dendrNumX * dendrNumY;
	srand((unsigned int)clock());

	//выбор средней точки, максимальных значений и нормализация
	float weightMean = 1.0 / numDendr;
	for(int i = 0; i < neuronNum; i++)
	{
		// квантование среднего 0.5 или -0.5
		eval = !eval;
		//shift = eval ? 0.5 : 1;
		for(int j = 0; j < numDendr + 1; j++) // +1 - bias
		{
			// вычисление среднего значения веса
			// значения индуцированного окального поля должны быть в линейной обасти активационной функции
			randNorm = (((rand() - RAND_MAX / 2.0) / (RAND_MAX * 2.0)) + shift) * weightMean; // случайное число от -1 до 1
			nn[i].weightCoeff[j].weight = randNorm;
			nn[i].weightCoeff[j].weightNew = 0.0;	// на всякий случай ;)
		}
	}
	return 1;
}

int convLayer::initInputFullConnect(convLayer *previous_Layer)
{
	if(dendrNumX * dendrNumY != previous_Layer->neuronNum)
	{	// количество входов элемента и количество выходов предыдущего слоя должно быть одинаковым
		return -1;
	}

	int numInputDendr = dendrNumX * dendrNumY;
	// соединение с предыдущим слоем
	for(int i = 0; i < neuronNum; i++) // перебор элементов для данного слоя
	{
		for(int j = 0; j < numInputDendr; j++) // перебор весов для данного элемента
		{	// инициализация входов данного слоя
			nn[i].ptrDendrDataInput[j] = &(previous_Layer->nn[j].out);
			// инициализация указателей предыдущего слоя
			previous_Layer->nn[j].ptrOutWeightCoeff[i] = &(nn[i].weightCoeff[j]);
		}
	}

	previousLayer = previous_Layer;
	previousLayer->nextLayer = this;

	return 1;
}

int convLayer::directCalc()
{
	int i;
	for(i = 0; i < neuronNum; i++)
	{
		nn[i].sum = 0;
	}

	int numDendr = dendrNumX * dendrNumY + 1;

	for(i = 0; i < neuronNum; i++)
	{
		for(int j = 0; j < numDendr; j++)
		{
			nn[i].sum += (nn[i].weightCoeff[j].weight) * (*(nn[i].ptrDendrDataInput[j]));
		}
		nn[i].out = transferFcn(nn[i].sum);
		nn[i].diff = transferFcnBack(nn[i].out);
	}
	return 1;
}

// 1 вычисление суммарной ошибки для каждого элемента данного слоя, по весам следующего слоя
void convLayer::errCalc()
{
	int i;
	for(i = 0; i < neuronNum; i++)
	{
		nn[i].bkErr = 0;
		for(int j = 0; j < numOutWeight; j++)
		{
			nn[i].bkErr += nn[i].ptrOutWeightCoeff[j]->weightNew;
		}
		nn[i].bkErr = nn[i].bkErr * nn[i].diff; // 04062015
	}
}

// 2 вычисление ошибок весовых коэффициентов для данного слоя
void convLayer::errWeightCalc()
{
	int numWeight = dendrNumX * dendrNumY + 1;
	int i;
	for(i = 0; i < neuronNum; i++)
	{
		for(int j = 0; j < numWeight; j++)
		{
			nn[i].weightCoeff[j].weightNew = nn[i].bkErr * nn[i].weightCoeff[j].weight;
		}
	}
}

// 3 вычисление новых весовых коэффициентов
// direct - обновить коэффициенты
int convLayer::tuneWeightBP()
{
	int numWeight = dendrNumX * dendrNumY + 1; // 1 - bias
	int i;
	for(i = 0; i < neuronNum; i++)
	{
		for(int j = 0; j < numWeight; j++)
		{
			nn[i].weightCoeff[j].weight += *(nn[i].ptrDendrDataInput[j]) * nn[i].bkErr * speedTrain;
		}
	}
	return 1;
}


inline float convLayer::transferFcn(float inputData)
{
// поправку -0,5 необходимо учесть при вычислении производной для этой функции!
	return 1.0/(1.0 + exp(-inputData));// - 0.5;
}


inline float convLayer::transferFcnBack(float inputData)
{
// с учетом поправки -0,5
//	inputData += 0.5;
	return (inputData) * (1.0 - inputData);
}


// 4 запись обновленных весовых коэффициентов
int convLayer::writeNewWeight()
{
	int numWeight = dendrNumX * dendrNumY + 1; // 1 - bias
	int i;

	for(i = 0; i < neuronNum; i++)
	{
		for(int j = 0; j < numWeight; j++)
		{
			nn[i].weightCoeff[j].weight = nn[i].weightCoeff[j].weightNew;
		}
	}
	return 1;
}

float convLayer::RMSECalc(float *data)
{
	float RMSE = 0;
	for(int i = 0; i < neuronNum; i++)
	{
		RMSE += qPow(nn[i].out - data[i],2);
	}
	return qSqrt(RMSE / neuronNum);
}


int NetNNT::calc()
{
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		layers[i]->directCalc();
	}
	return 1;
}

// в выходной слой ошибки должны быть записаны заранее
int NetNNT::tune()
{
	for(int i = 0; i < layers[NUM_LAYERS - 1]->neuronNum; i++)
	{
		layers[NUM_LAYERS - 1]->nn[i].bkErr = layers[NUM_LAYERS - 1]->nn[i].diff * layers[NUM_LAYERS - 1]->nn[i].bkErr;
	}

	layers[NUM_LAYERS - 1]->errWeightCalc();

	for(int i = NUM_LAYERS-2; i >= 0; i--)
	{
		// вычисление ошибок сети
		layers[i]->errCalc();
		// вычисление ошибок для весов элементов
		layers[i]->errWeightCalc();
	}

	for(int i = NUM_LAYERS-1; i >= 0; i--)
	{
		layers[i]->tuneWeightBP();
	}
	return 1;
}

int NetNNT::loadFromCharBuff(char *data, int len)
{
	float *ptrWeight = (float*)data;
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		for(int j = 0; j < layers[i]->neuronNum; j++)
		{
			for(int k = 0; k < layers[i]->dendrNumX * layers[i]->dendrNumY + 1; k++)
			{
				layers[i]->nn[j].weightCoeff[k].weight = *ptrWeight;
				ptrWeight++;
			}
		}
	}
	return 1;
}


int NetNNT::loadFromFile(QFile *file)
{
	int lenFile = file->size();
	char *dataFile = new char[lenFile];
	file->read(dataFile, lenFile);

	loadFromCharBuff(dataFile, lenFile);

	delete[] dataFile;
	return 1;
}



int NetNNT::saveToCharBuff(float *data)
{
	int len = 0;

	float *ptrWeight = data;
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		for(int j = 0; j < layers[i]->neuronNum; j++)
		{
			for(int k = 0; k < layers[i]->dendrNumX * layers[i]->dendrNumY + 1; k++)
			{
				*ptrWeight = layers[i]->nn[j].weightCoeff[k].weight;
				ptrWeight++;
				len++;
			}
		}
	}
	return len;
}



int NetNNT::saveToFile(QFile *file)
{
	int sizeWeight = 0;
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		sizeWeight += layers[i]->neuronNum * (layers[i]->dendrNumX * layers[i]->dendrNumY + 1);
	}

	float *dataFile = new float[sizeWeight];

	sizeWeight = saveToCharBuff(dataFile);

	file->write((char *)dataFile, sizeWeight * sizeof(float));

	delete[] dataFile;
	return 1;
}



NetNNT::NetNNT()
{
	// (nNum, - количество нейронов в слое
	// dendrNum_X, dendrNum_Y, - количество связей для каждого элемента, по горизонтали и вертикали
	// numOut_Weight) количество выходных связей для каждого элемента
	layers[0] = new convLayer((DATA_IN_H/2) * (DATA_IN_W/2) * 3, DATA_IN_H, DATA_IN_W, 5 * 5);
	layers[1] = new convLayer(5 * 5, (DATA_IN_H/2)*3, DATA_IN_W/2, NUM_OUT);
	layers[2] = new convLayer(NUM_OUT, 5, 5, NUM_OUT);

	// инициализация весовых коэффициентов
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		layers[i]->initWeight();
	}

	// инициализация указателей на данные, для связи слоев
	for(int i = 0; i < NUM_LAYERS - 1; i++)
	{
		layers[i+1]->initInputFullConnect(layers[i]);
	}

}

NetNNT::~NetNNT()
{
	delete[] layers;
}


