// facetest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
// Face recognition.cpp : 定义控制台应用程序的入口点。
//



#include <iostream>  
#include <fstream>    
#include <opencv2/opencv.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include "opencv2/imgproc.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/ml.hpp"  
using namespace std;
using namespace cv;

//Parameters  
#define N_BINS 16           //Number of bins  
#define N_DIVS 3            //Number of cells = N_DIVS*N_DIVS  
#define N_PHOG N_DIVS*N_DIVS*N_BINS  
#define BIN_RANGE (2*CV_PI)/N_BINS  
//Haar Cascade Path  


//Input: Grayscale image  
//Output: HOG features  
Mat hog(const Mat &Img);


#define PosSamNO 80    //正样本个数    
#define NegSamNO 50    //负样本个数    
#define HardExampleNO 0     
#define TRAIN true    //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型    

int main()
{
	// initial SVM  
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定    
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数        
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人   


	if (TRAIN)
	{
		//依次读取正样本图片，生成HOG特征  
		for (int i = 1; i <= PosSamNO; i++)
		{
			char pic_name[64];
			sprintf(pic_name,"E:\\data\\face\\pos\\%02d.jpg", i);
			//  cout << pic_name << endl;  
			Mat src = imread(pic_name);//读取图片    
			resize(src, src, Size(64, 64));//将图片大小缩放为64*64  
			Mat img_gray;
			cvtColor(src, img_gray, CV_BGR2GRAY);//将彩色图片转换为灰度图  
			Mat feature = hog(img_gray);//提取HOG特征  
			if (1 == i)
			{
				DescriptorDim = feature.cols;//feature.size();//HOG描述子的维数    
											 //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat    
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人    
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
			}

			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(i - 1, j) = feature.at<float>(0, j);//第i个样本的特征向量中的第j个元素    
			sampleLabelMat.at<int>(i - 1, 0) = 1;//正样本类别为1，是人脸  


		}
		//依次读取负样本图片，生成HOG特征  
		for (int i = 1; i <= NegSamNO; i++)
		{
			char pic_name[64];
			sprintf_s(pic_name, "E:\\data\\face\\neg2\\%02d.jpg", i);
			//  cout << pic_name << endl;  
			Mat src = imread(pic_name);//读取图片    
			resize(src, src, Size(64, 64));
			Mat img_gray;
			cvtColor(src, img_gray, CV_BGR2GRAY);
			Mat feature = hog(img_gray);

			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(PosSamNO + i - 1, j) = feature.at<float>(0, j);//第i个样本的特征向量中的第j个元素    
			sampleLabelMat.at<int>(PosSamNO + i - 1, 0) = -1;//负样本类别为-1，非人脸  


		}

		////输出样本的HOG特征向量矩阵到文件    
		//ofstream fout("SampleFeatureMat.txt");  
		//for (int i = 0; i < PosSamNO + NegSamNO; i++)  
		//{  
		//  fout << i << endl;  
		//  for (int j = 0; j < DescriptorDim; j++)  
		//      fout << sampleFeatureMat.at<float>(i, j) << "  ";  
		//  fout << endl;  
		//}  

		//训练SVM分类器    
		//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();  
		svm->setType(cv::ml::SVM::Types::C_SVC);//设置SVM类型  
		svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);//设置核函数  
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

		// train operation  
		svm->train(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);

		svm->save("svm_save.xml");

	}
	else//若TRAIN为false，从XML文件读取训练好的分类器    
	{
		String filename = "svm_save.xml";
		svm = cv::ml::StatModel::load<cv::ml::SVM>(filename);
		//svm->load(filename);  
	}




	//下面开始预测  
	for (int i = 1; i < 40; i++)
	{
		char pic_name[64];
		sprintf_s(pic_name, "E:\\data\\face\\test\\%02d.jpg", i);
		cout << pic_name << ":";
		Mat src = imread(pic_name);//读取图片    
		resize(src, src, Size(64, 64));
		Mat img_gray;
		cvtColor(src, img_gray, CV_BGR2GRAY);
		Mat feature = hog(img_gray);

		float respose = svm->predict(feature);
		if (respose == 1)
			cout << "人脸" << endl;
		else if (respose == -1)
			cout << "非人脸" << endl;
	}


	getchar();
	return 0;
}





Mat hog(const Mat &Img)
{
	Mat Hog;
	Hog = Mat::zeros(1, N_PHOG, CV_32FC1);

	Mat Ix, Iy;

	//Find orientation gradients in x and y directions  
	Sobel(Img, Ix, CV_16S, 1, 0, 3);
	Sobel(Img, Iy, CV_16S, 0, 1, 3);

	int cellx = Img.cols / N_DIVS;
	int celly = Img.rows / N_DIVS;

	int img_area = Img.rows * Img.cols;

	for (int m = 0; m < N_DIVS; m++)
	{
		for (int n = 0; n < N_DIVS; n++)
		{
			for (int i = 0; i<cellx; i++)
			{
				for (int j = 0; j<celly; j++)
				{

					float px, py, grad, norm_grad, angle, nth_bin;

					//px = Ix.at(m*cellx+i, n*celly+j);  
					//py = Iy.at(m*cellx+i, n*celly+j);  
					px = static_cast<float>(Ix.at<int16_t>((m*cellx) + i, (n*celly) + j));
					py = static_cast<float>(Iy.at<int16_t>((m*cellx) + i, (n*celly) + j));
					grad = static_cast<float>(std::sqrt(1.0*px*px + py*py));
					norm_grad = grad / img_area;

					//Orientation  
					angle = std::atan2(py, px);

					//convert to 0 to 360 (0 to 2*pi)  
					if (angle < 0)
						angle += 2 * CV_PI;

					//find appropriate bin for angle  
					nth_bin = angle / BIN_RANGE;

					//add magnitude of the edges in the hog matrix  
					Hog.at<float>(0, (m*N_DIVS + n)*N_BINS + static_cast<int>(angle)) += norm_grad;

				}
			}
		}
	}

	//Normalization  
	for (int i = 0; i< N_DIVS*N_DIVS; i++)
	{
		float max = 0;
		int j;
		for (j = 0; j<N_BINS; j++)
		{
			if (Hog.at<float>(0, i*N_BINS + j) > max)
				max = Hog.at<float>(0, i*N_BINS + j);
		}
		for (j = 0; j<N_BINS; j++)
			Hog.at<float>(0, i*N_BINS + j) /= max;
	}
	return Hog;
}
