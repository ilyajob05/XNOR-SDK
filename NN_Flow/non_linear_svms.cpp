#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#include <time.h>

#define TIMEWAITKEY 1
#define SIZE_HOG_H 4
#define SIZE_HOG_W 4

#define HOG_WIN_SIZE		24,24
#define HOG_BLOCK_SIZE		4,4
#define HOG_BLOCK_STRIDE	2,2
#define HOG_CELL_SIZE		2,2
#define HOG_NBINS			9

using namespace cv;
using namespace cv::ml;
using namespace std;

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void load_images(const string & filename, vector< Mat > & img_lst, bool equalisation, Size sizeImg = Size(0,0));
void normalised_images(Mat &src, Mat &dst);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void test_video_cam(const Size & size);
void test_image(const Size & size);

CascadeClassifier signCascade1;
CascadeClassifier signCascade2;
String sign_cascade_name1 = "C:/opencv/build/x64/vc12/bin/sign_detect_27_07_2016_v1/cascade.xml";
String sign_cascade_name2 = "C:/opencv/build/x64/vc12/bin/sign/cascade.xml";

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}



void load_images_original(const string & filename, vector< Mat > & img_lst, Size sizeImg)
{
	int count = 0;
	string line;
	ifstream file;
	clog << "load image fName: \"" << filename << "\"\n";
	file.open((filename).c_str());
	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	bool end_of_parsing = false;
	while (!end_of_parsing)
	{
		getline(file, line);
		if (line == "") // no more file to read
		{
			end_of_parsing = true;
			break;
		}
		Mat img = imread((line).c_str()); // load the image
		count++;
		clog << count << "\t" << line << "\n";
		if (img.empty()) // invalid image, just skip it.
			continue;
#ifdef _DEBUG
		imshow("image", img);
		waitKey(TIMEWAITKEY);
#endif
		if (sizeImg.height != 0)
		{
			resize(img, img, sizeImg);
		}

		img_lst.push_back(img.clone());
	}
}


void normalised_images(Mat &src, Mat &dst)
{
	cvtColor(src, dst, COLOR_BGR2GRAY);
	equalizeHist(dst, dst);
	GaussianBlur(dst, dst, Size(3, 3), 1.5);

#ifdef _DEBUG
	namedWindow("imageGauss", CV_WINDOW_FREERATIO);
	imshow("imageGauss", dst);
#endif
}

void load_images(const string & filename, vector< Mat > & img_lst, bool equalisation, Size sizeImg)
{
	int count = 0;
	string line;
	ifstream file;
	clog << "load image fName: \"" << filename << "\"\n";
	file.open((filename).c_str());
	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	bool end_of_parsing = false;
	while (!end_of_parsing)
	{
		getline(file, line);
		if (line == "") // no more file to read
		{
			end_of_parsing = true;
			break;
		}
		Mat img = imread((line).c_str()); // load the image
		if (equalisation)
		{
			normalised_images(img, img);
		}

		count++;
		clog << count << "\t" << line << "\n";
		if (img.empty()) // invalid image, just skip it.
			continue;
#ifdef _DEBUG
		imshow("imageOriginal", img);

		waitKey(TIMEWAITKEY);
#endif
		if (sizeImg.height != 0)
		{
			resize(img, img, sizeImg, INTER_CUBIC);
		}

		img_lst.push_back(img.clone());
		img.release();
	}
}



void load_images_and_save(const string & filename, vector< Mat > & img_lst, bool equalisation, Size sizeImg)
{
	int count = 0;
	string line;
	ifstream file;
	clog << "load image fName: \"" << filename << "\"\n";
	file.open((filename).c_str());
	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	bool end_of_parsing = false;
	Mat img;

	while (!end_of_parsing)
	{
		getline(file, line);
		if (line == "") // no more file to read
		{
			end_of_parsing = true;
			break;
		}
		img = imread((line).c_str()); // load the image
		if (equalisation)
		{
			normalised_images(img, img);
		}

		count++;
		clog << count << "\t" << line << "\n";
		if (img.empty()) // invalid image, just skip it.
			continue;

		cvtColor(img, img, COLOR_BGR2GRAY);
		resize(img, img, sizeImg, INTER_CUBIC);
		imshow("imageOriginal", img);
		String fileName = "C:/Users/Ilya/Documents/Projects/DATA/" + to_string(count) + "_positive_sign_h24_w24.bmp";
		imwrite(fileName, img);
		img.release();
	}
	clog << "\ncomplete";
}



void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	Mat tmp;

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();
	for (; img != end; ++img)
	{
		//box.x = rand() % (img->cols - size_x);
		//box.y = rand() % (img->rows - size_y);
		//Mat roi = (*img)(box);

		neg_lst.push_back(img->clone());
		//neg_lst.push_back(roi.clone());
#ifdef _DEBUG
		//imshow("img", roi.clone());
		imshow("img", img->clone());
		waitKey(TIMEWAITKEY);
#endif
	}
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 2;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{/*
	Size _winSize, Size _blockSize, Size _blockStride,
		Size _cellSize, int _nbins, int _derivAperture = 1, double _winSigma = -1,
		int _histogramNormType = HOGDescriptor::L2Hys, */
	HOGDescriptor hog = HOGDescriptor(Size(HOG_WIN_SIZE), Size(HOG_BLOCK_SIZE), Size(HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE), HOG_NBINS);
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		if (img->type() != CV_8UC1)
		{
			cvtColor(*img, gray, COLOR_BGR2GRAY);
		}
		//hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
		hog.compute(gray, descriptors, Size(2, 2), Size(0, 0), location);
		//hog.compute(gray, descriptors, Size(4, 4), Size(0, 0), location);
		gradient_lst.push_back(Mat(descriptors).clone());
#ifdef _DEBUG
		//imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		waitKey(TIMEWAITKEY);
#endif
	}
}

void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training SVM...";
	Ptr<SVM> svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	svm->save("road_sign_detector.yml");
}

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 3);
		}
	}
}

void draw_locations_weight(Mat &img, const vector<Rect> &locations, const vector<double> &weight)
{
	Mat inColorMap = Mat::ones(1, 1, CV_8U);
	Mat outColorMap;


	if (!locations.empty())
	{
		int i = 0;
		uchar color;
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{

			inColorMap.at<Vec3b>(Point(0, 0)) = Vec3b(127 * weight[i], 127 * weight[i], 127 * weight[i]);

			applyColorMap(inColorMap, outColorMap, COLORMAP_JET);

			rectangle(img, *loc, Scalar(outColorMap.at<Vec3b>(Point(0,0))), 2);
			i++;
		}
	}
}

void test_video_cam(const Size & size)
{
	char key = 27;
	Scalar reference(0, 255, 0);
	Scalar trained(0, 0, 255);
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	HOGDescriptor my_hog = HOGDescriptor(Size(HOG_WIN_SIZE), Size(HOG_BLOCK_SIZE), Size(HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE), HOG_NBINS);

	VideoCapture video;
	vector< Rect > locations;

	// Load the trained SVM.
	svm = StatModel::load<SVM>("road_sign_detector.yml");
	// Set the trained svm to my_hog
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);
	// Set the people detector.
	//hog.setSVMDetector(hog.getDefaultPeopleDetector());
	// Open the camera.
	//video.open("c:/Users/Ilya/Documents/Projects/DATA/road_sign/FILE0246(00h00m12s-00h02m46s).avi");
	video.open(0);
	if (!video.isOpened())
	{
		cerr << "Unable to open the device 0" << endl;
		exit(-1);
	}

	bool end_of_process = false;
	while (!end_of_process)
	{
		video >> img;
		if (img.empty())
			break;

		//draw = img.clone();

		cvtColor(img, draw, CV_BGR2GRAY);

		//resize(draw, draw, Size(1920/2, 1080/2), INTER_CUBIC);
		
		//locations.clear();
		//hog.detectMultiScale(img, locations);
		//draw_locations(draw, locations, reference);


		//(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
		//	double hitThreshold = 0, Size winStride = Size(),
		//	Size padding = Size(), double scale = 1.05,
		//	double finalThreshold = 2.0, bool useMeanshiftGrouping = false)


		locations.clear();
		my_hog.detectMultiScale(draw, locations, 0, Size(8,8));
		draw_locations(draw, locations, trained);

		imshow("Video", draw);
		key = (char)waitKey(TIMEWAITKEY);
		if (27 == key)
			end_of_process = true;
	}
}

void test_video_cam_haar(const Size & size)
{
	char key = 27;
	Scalar reference(0, 255, 0);
	Scalar trained(0, 0, 255);
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	HOGDescriptor my_hog = HOGDescriptor(Size(HOG_WIN_SIZE), Size(HOG_BLOCK_SIZE), Size(HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE), HOG_NBINS);

	VideoCapture video;
	vector< Rect > locations;

	// Load the trained SVM.
	svm = StatModel::load<SVM>("road_sign_detector.yml");
	// Set the trained svm to my_hog
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);
	// Set the people detector.
	//hog.setSVMDetector(hog.getDefaultPeopleDetector());
	// Open the camera.
	//video.open("c:/Users/Ilya/Documents/Projects/DATA/road_sign/FILE0246(00h00m12s-00h02m46s).avi");
	video.open("c:/Users/Ilya/Documents/Projects/DATA/road_sign/EMER0003(00h01m36s-00h01m39s).avi");
	if (!video.isOpened())
	{
		cerr << "Unable to open the device 0" << endl;
		cin >> key;
		exit(-1);
	}

	/*******************************************************************/
	// HAAR
	if (!signCascade1.load(sign_cascade_name1)) {
		cout << "Couldn't load sign detector \n";
	}
	Mat gray;
	vector<cv::Rect> signColl;
	/*******************************************************************/

	bool end_of_process = false;
	while (!end_of_process)
	{
		video >> img;
		if (img.empty())
			break;

		draw = img.clone();

		resize(draw, draw, Size(960+640, 540+360));
		GaussianBlur(draw, draw, Size(3, 3), 1.0);

		//locations.clear();
		//hog.detectMultiScale(img, locations);
		//draw_locations(draw, locations, reference);

		locations.clear();
		//my_hog.detectMultiScale(img, locations);
		//draw_locations(draw, locations, trained);


		/***************************************************************/
		cvtColor(draw, gray, CV_BGR2GRAY);
		cv::equalizeHist(gray, gray);
		// Detect sign
		signCascade1.detectMultiScale(gray, signColl, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(1, 1));

		cvtColor(gray, gray, CV_GRAY2BGR);
		draw_locations(gray, signColl, Scalar(0, 255, 0));
		cv::namedWindow("Image HAAR detect", CV_WINDOW_FREERATIO);
		imshow("Image HAAR detect", gray);
		/***************************************************************/

		key = (char)waitKey(TIMEWAITKEY);
		if (27 == key)
			end_of_process = true;
	}
}



void test_image(const Size & size)
{
	char key = 0;
	Scalar reference(0, 255, 0);
	Scalar trained(0, 0, 255);
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	HOGDescriptor my_hog = HOGDescriptor(Size(HOG_WIN_SIZE), Size(HOG_BLOCK_SIZE), Size(HOG_BLOCK_STRIDE), Size(HOG_CELL_SIZE), HOG_NBINS);

	VideoCapture video;
	vector< Rect > locations;

	// Load the trained SVM.
	svm = StatModel::load<SVM>("road_sign_detector.yml");
	// Set the trained svm to my_hog
	vector< float > hog_detector;

	vector<Point> centerObj;
	vector<double> weightObj;
	get_svm_detector(svm, hog_detector);
	my_hog.setSVMDetector(hog_detector);
	// Set the people detector.
	hog.setSVMDetector(hog.getDefaultPeopleDetector());
	
	
	vector<Mat> testImage;

	// Load image
	//char *fName = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/test_sign10.txt";
	//char *fName = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/test_sign28.txt";
	//char *fName = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/testIJCNN2013.txt";
	//char *fName = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/test_sign100_H400.txt";
	//char *fName = "c:/Users/Ilya/Documents/Projects/DATA/road_sign/SetPart3_www_isy_liu_secvlresearchtrafficSignsswedishSignsSumme/SetPart3_www_isy_liu_secvlresearchtrafficSignsswedishSignsSumme.txt";
	char *fName = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/test_sign_street.txt";
	load_images(fName, testImage, false, Size(0,0));
	

	/*******************************************************************/
	// HAAR
	if (!signCascade1.load(sign_cascade_name1)) {
		cout << "Couldn't load sign detector \n";
	}
	Mat gray1;
	vector<cv::Rect> signColl1;
	/*******************************************************************/

	/*******************************************************************/
	// HAAR
	if (!signCascade2.load(sign_cascade_name2)) {
		cout << "Couldn't load sign detector \n";
	}
	Mat gray2;
	vector<cv::Rect> signColl2;
	/*******************************************************************/



	int count = 0;
	int countHaar = 0;
	int countHOG = 0;


	int64 t0;
	int64 msecs;

	vector< Mat >::const_iterator imgIt = testImage.begin();
	vector< Mat >::const_iterator end = testImage.end();

	for (; imgIt != end; ++imgIt)
	{
		cout << "sign detecting...";
		
		//locations.clear();
		//hog.detectMultiScale(*imgIt, locations);
		//draw_locations(draw, locations, reference);

		locations.clear();
		//detectMultiScale(InputArray image, vector<Rect>& objects, double scaleFactor=1.1, 
		//int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())

		//! with result weights output
		//CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
		//	CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
		//	Size winStride = Size(), Size padding = Size(), double scale = 1.05,
		//	double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

		/***************************************************************/
		cvtColor(*imgIt, gray1, CV_BGR2GRAY);
		resize(gray1, gray1, Size(), 0.5, 0.5);
		//equalizeHist(gray1, gray1);
		img = imgIt->clone();
		draw = imgIt->clone();		// Detect sign
		//signCascade1.detectMultiScale(gray1, signColl1, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(1, 1));

		//vector< Rect >::const_iterator loc = signColl1.begin();
		//vector< Rect >::const_iterator end = signColl1.end();
		//for (; loc != end; ++loc)
		//{
		//	imwrite("haar_" + to_string(++countHaar) + "_.bmp", (*imgIt)(*loc));
		//}

		//draw_locations(img, signColl1, Scalar(0, 255, 0));
		//cv::namedWindow("Image HAAR detect", CV_WINDOW_FREERATIO);
		//imshow("Image HAAR detect", img);

		//equalizeHist(gray2, gray2);
		//signCascade2.detectMultiScale(gray1, signColl2, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(1, 1));
		//draw_locations(draw, signColl2, Scalar(0, 255, 0));
		//cv::namedWindow("Image HAAR detect EQ", CV_WINDOW_FREERATIO);
		//imshow("Image HAAR detect EQ", draw);
		/***************************************************************/


		t0 = getTickCount();

		
		//my_hog.detectMultiScale(gray1/**imgIt*/, locations, weightObj, 0.05/*0.1*/, Size(), Size(), 1.1, 2.0, false);
		my_hog.detectMultiScale(gray1/**imgIt*/, locations, weightObj);

		msecs = (int)((getTickCount() - t0) / getTickFrequency()*1000);
		clog << "Detect sign times passed in ms: " << msecs << "\n";

		//loc = locations.begin();
		//end = locations.end();
		//for (; loc != end; ++loc)
		//{
		//	imwrite("hog_" + to_string(++countHOG) + "_.bmp", (*imgIt)(*loc));
		//}

		cvtColor(gray1, draw, COLOR_GRAY2BGR);

		draw_locations_weight(draw, locations, weightObj);
		//draw_locations_weight(draw, locations, weightObj);

		

		cout << "complete\n";
		cv::namedWindow("Image HOG detect", CV_WINDOW_FREERATIO);
		imshow("Image HOG detect", draw);
		count++;
		


		//imwrite( to_string(count) + "_res.png", draw );
		key = (char)waitKey();
		if (27 == key)	// if ESCAPE KEY
			break;
	}

}


int main(int argc, char** argv)
{
	char *fNamePositive = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/pos_sign_square_clean.txt";
	//char *fNameNegative = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/neg_sign.txt";
	char *fNameNegative = "c:/Users/Ilya/Documents/Projects/visualStudio/PILOT_DETECT/SIGN_DETECT/SVM_create_HOG_proc/SVM_create/neg_sign_cut_and_auto.txt";

	//if (argc != 5)
	//{
	//	cout << "Wrong number of parameters." << endl
	//		<< "Usage: " << argv[0] << " pos_dir pos.lst neg_dir neg.lst" << endl
	//		<< "example: " << argv[0] << " /INRIA_dataset/ Train/pos.lst /INRIA_dataset/ Train/neg.lst" << endl;
	//	exit(-1);
	//}
	vector< Mat > pos_lst;
	vector< Mat > full_neg_lst;
	vector< Mat > neg_lst;
	vector< Mat > gradient_lst;
	vector< int > labels;
	
	cout << "Press \'t\' to start training";
	char key;
	key = (char)getchar();

	if (key == 't' || key == 'T')
	{
		load_images(fNamePositive, pos_lst, false, Size(24, 24));
		labels.assign(pos_lst.size(), +1);
		const unsigned int old = (unsigned int)labels.size();
		load_images(fNameNegative, full_neg_lst, false, Size(24, 24));
		sample_neg(full_neg_lst, neg_lst, Size(24, 24));
		labels.insert(labels.end(), neg_lst.size(), -1);
		CV_Assert(old < labels.size());

		compute_hog(pos_lst, gradient_lst, Size(24, 24));
		compute_hog(neg_lst, gradient_lst, Size(24, 24));

		train_svm(gradient_lst, labels);
	}

	//test_video_cam_haar(Size(24, 24)); // change with your parameters
	//test_video_cam(Size(24, 24)); // change with your parameters
	test_image(Size(24,24));

	//load_images_and_save(fNamePositive, pos_lst, false, Size(24, 24));
	cin >> key;
	return 0;
}
