#ifndef NNDETECT_H
#define NNDETECT_H

#include <QString>

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

class NNLayer
{
public:
    float biasConst = 1;
    //float speedTrain = 0.1;

    // количество связей для каждого нейрона
    int dendrNumX, dendrNumY;

    // количество нейронов в слое
    int neuronNum;

    // количество выходов нейрона
    int numOutWeight;

    // массив нейронов
    neurStruct *nn;

    int initInputFullConnect(NNLayer *previous_Layer);

    //int initWeight(void);

    int directCalc();

    //int writeNewWeight();

    //int tuneWeightBP(bool direct);

    NNLayer *previousLayer = nullptr;
    NNLayer *nextLayer = nullptr;

    //void errCalc();

    //float RMSECalc(float *data);

    //void errWeightCalc();

    float transferFcn(float inputData);
    float transferFcnBack(float inputData);

    NNLayer(int nNum, int dendrNum_X, int dendrNum_Y, int numOut_Weight);
    ~NNLayer();
};

class NNDetect
{
public:
    static const int DATA_IN_H = 28;
    static const int DATA_IN_W = 28;
    static const int NUM_LAYERS = 3;

    // указатель на слои
	NNLayer *layers[NUM_LAYERS];
    // указатель на массив входных данных
    float *ptrDataInput;
    // размер массива входных данных
    int inputDataLen = DATA_IN_H * DATA_IN_W;

    int calc();
    // загрузка конфигурации из файла
    int loadFromFile(QString fileName);
    // загрузка конфигурации из массива
	int loadFromCharBuff(char *data, int len);

    NNDetect();
    ~NNDetect();
};

#endif // NNDETECT_H
