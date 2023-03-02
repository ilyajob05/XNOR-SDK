#include "nndetect.h"
#include <QFile>

// сеть для 10 целевых значений

NNLayer::NNLayer(int nNum, int dendrNum_X, int dendrNum_Y, int numOut_Weight)
{
    neuronNum = nNum;
    dendrNumX = dendrNum_X;
    dendrNumY = dendrNum_Y;
    numOutWeight = numOut_Weight;

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

NNLayer::~NNLayer()
{
    for(int i = 0; i < neuronNum; i++)
    {
        delete[] nn[i].weightCoeff;
        delete[] nn[i].ptrDendrDataInput;
        delete[] nn[i].ptrOutWeightCoeff;
    }
    delete[] nn;
}

int NNLayer::initInputFullConnect(NNLayer *previous_Layer)
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

int NNLayer::directCalc()
{
    int i;
    for(i = 0; i < neuronNum; i++)
    {
        nn[i].sum = 0;
    }

    int numDendr = dendrNumX * dendrNumY + 1;

    for(int i = 0; i < neuronNum; i++)
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

float NNLayer::transferFcn(float inputData)
{
    return 1.0/(1.0 + exp(-inputData));// - 0.5;
}

float NNLayer::transferFcnBack(float inputData)
{
    return (inputData) * (1.0 - inputData);
}

NNDetect::NNDetect()
{
    layers[0] = new NNLayer(14*14, 28, 28, 5*5);
    layers[1] = new NNLayer(5*5, 14, 14, 10*1);
    layers[2] = new NNLayer(10*1, 5, 5, 10*1);

    layers[1]->initInputFullConnect(layers[0]);
    layers[2]->initInputFullConnect(layers[1]);
}

NNDetect::~NNDetect()
{
    delete[] layers;
}

int NNDetect::calc()
{
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		layers[i]->directCalc();
	}
}

int NNDetect::loadFromCharBuff(char *data, int len)
{
	float *ptrWeight = (float*)data;
	for(int i = 0; i < NUM_LAYERS; i++)
	{
		for(int j = 0; j < layers[i]->neuronNum; j++)
		{
			for(int k = 0; k < layers[i]->neuronNum; k++)
			{
				if(ptrWeight > (float*)(data + len)){	return 1; }
				layers[i]->nn[j].weightCoeff[k].weight = *ptrWeight;
				ptrWeight++;
			}
		}
	}
}

int NNDetect::loadFromFile(QString fileName)
{
	QFile file(fileName);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
		return -1;

	int lenFile = file.size();
	char *dataFile = new char[lenFile];
	file.read(dataFile, lenFile);

	loadFromCharBuff(dataFile, lenFile);

	delete[] dataFile;
	file.close();
}






