#ifndef NETLAYER_H
#define NETLAYER_H

#include <QString>

class traningSampleImg
{
public:
    // массив входных данных
	float *sampleData;
    // массив целей
    float *targetData;
    // количество целей
    int numTarget;
	// размер обучающей выборки

    // размерность входных данных, количество примеров
    int sizeX, sizeY, numSample;
	int loadDataFromFile();
	float* getImgNum(int num);
	float* getTargetNum(int num);
	traningSampleImg(int size_X, int size_Y, int num_Target, int num_Sample);
    ~traningSampleImg();
};

struct weight_Coeff
{
	// массив весовых коэффициентов
	float weight;
	// массив обновленных весовых коэффициентов, также используется для вычисления взвешенной ошибки
	float weightNew;

};

struct neurStruct
{
	// массив указателей на входные данные
	float **ptrDendrDataInput;
	// массив весовых коэффициентов
	weight_Coeff *weightCoeff;
	// данные сумматора
	float sum;
	// выход данных
	float out;
	// производная активационной функции
	float diff;
	// массив указателей на веса следующего слоя, для алг. обра. распространения
	weight_Coeff **ptrOutWeightCoeff;
	// расчитанная ошибка
	float bkErr;
};

class convLayer
{
public:
	float biasConst = 1;
	float speedTrain = 0.1;

	// количество связей для каждого нейрона
	int dendrNumX, dendrNumY;

	// количество нейронов в слое
	int neuronNum;

	// количество выходов нейрона
	int numOutWeight;

	// массив нейронов
	neurStruct *nn;

	int initInputFullConnect(convLayer *previous_Layer);

	int initWeight(void);

	int directCalc();

	int writeNewWeight();

	int tuneWeightBP(bool direct);

	convLayer *previousLayer = nullptr;
	convLayer *nextLayer = nullptr;

	void errCalc();

	float RMSECalc(float *data);

	void errWeightCalc();

	float transferFcn(float inputData);
	float transferFcnBack(float inputData);

	// nNum - количество нейронов в слое
	// dendrNum_X, dendrNum_Y - количество связей для каждого элемента, по горизонтали и вертикали
	// numOut_Weight количество выходных связей для каждого элемента
	convLayer(int nNum, int dendrNum_X, int dendrNum_Y, int numOut_Weight);
	~convLayer();

};

class NetNNT
{
public:
	static const int DATA_IN_H = 28;
	static const int DATA_IN_W = 28;
	static const int NUM_LAYERS = 3;

	int numLeyer;
	convLayer *layers = NULL;

	// загрузка конфигурации из файла
	int loadFromFile(QString fileName);
	// загрузка конфигурации из массива
	int loadFromCharBuff(char *data, int len);

	// сохранение конфигурации в массив
	int saveToFile(QString fileName);
	// сохранение конфигурации в массив
	int saveToCharBuff(char *data);

	int calc();

	NetNNT();
	~NetNNT();
};

#endif // NETLAYER_H
