// facetest.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
// Face recognition.cpp : �������̨Ӧ�ó������ڵ㡣
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


#define PosSamNO 80    //����������    
#define NegSamNO 50    //����������    
#define HardExampleNO 0     
#define TRAIN true    //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��    

int main()
{
	// initial SVM  
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������    
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��        
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����   


	if (TRAIN)
	{
		//���ζ�ȡ������ͼƬ������HOG����  
		for (int i = 1; i <= PosSamNO; i++)
		{
			char pic_name[64];
			sprintf(pic_name,"E:\\data\\face\\pos\\%02d.jpg", i);
			//  cout << pic_name << endl;  
			Mat src = imread(pic_name);//��ȡͼƬ    
			resize(src, src, Size(64, 64));//��ͼƬ��С����Ϊ64*64  
			Mat img_gray;
			cvtColor(src, img_gray, CV_BGR2GRAY);//����ɫͼƬת��Ϊ�Ҷ�ͼ  
			Mat feature = hog(img_gray);//��ȡHOG����  
			if (1 == i)
			{
				DescriptorDim = feature.cols;//feature.size();//HOG�����ӵ�ά��    
											 //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat    
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����    
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
			}

			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(i - 1, j) = feature.at<float>(0, j);//��i�����������������еĵ�j��Ԫ��    
			sampleLabelMat.at<int>(i - 1, 0) = 1;//���������Ϊ1��������  


		}
		//���ζ�ȡ������ͼƬ������HOG����  
		for (int i = 1; i <= NegSamNO; i++)
		{
			char pic_name[64];
			sprintf_s(pic_name, "E:\\data\\face\\neg2\\%02d.jpg", i);
			//  cout << pic_name << endl;  
			Mat src = imread(pic_name);//��ȡͼƬ    
			resize(src, src, Size(64, 64));
			Mat img_gray;
			cvtColor(src, img_gray, CV_BGR2GRAY);
			Mat feature = hog(img_gray);

			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(PosSamNO + i - 1, j) = feature.at<float>(0, j);//��i�����������������еĵ�j��Ԫ��    
			sampleLabelMat.at<int>(PosSamNO + i - 1, 0) = -1;//���������Ϊ-1��������  


		}

		////���������HOG�������������ļ�    
		//ofstream fout("SampleFeatureMat.txt");  
		//for (int i = 0; i < PosSamNO + NegSamNO; i++)  
		//{  
		//  fout << i << endl;  
		//  for (int j = 0; j < DescriptorDim; j++)  
		//      fout << sampleFeatureMat.at<float>(i, j) << "  ";  
		//  fout << endl;  
		//}  

		//ѵ��SVM������    
		//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();  
		svm->setType(cv::ml::SVM::Types::C_SVC);//����SVM����  
		svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);//���ú˺���  
		svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

		// train operation  
		svm->train(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);

		svm->save("svm_save.xml");

	}
	else//��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����    
	{
		String filename = "svm_save.xml";
		svm = cv::ml::StatModel::load<cv::ml::SVM>(filename);
		//svm->load(filename);  
	}




	//���濪ʼԤ��  
	for (int i = 1; i < 40; i++)
	{
		char pic_name[64];
		sprintf_s(pic_name, "E:\\data\\face\\test\\%02d.jpg", i);
		cout << pic_name << ":";
		Mat src = imread(pic_name);//��ȡͼƬ    
		resize(src, src, Size(64, 64));
		Mat img_gray;
		cvtColor(src, img_gray, CV_BGR2GRAY);
		Mat feature = hog(img_gray);

		float respose = svm->predict(feature);
		if (respose == 1)
			cout << "����" << endl;
		else if (respose == -1)
			cout << "������" << endl;
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
