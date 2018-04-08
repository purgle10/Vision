#define IMAGEJUDGEDEBUG
#ifdef IMAGEJUDGEDEBUG
#include<opencv2\opencv.hpp>
#include<iostream>
#include<string>
#include<cmath>
#include<ctime>
#endif

double getMSE(const cv::Mat &test, const cv::Mat &reference);
double getPSNR(const cv::Mat &test, const cv::Mat &reference);//PSNR=10log(max(test)/mse)
double getSSIM(const cv::Mat &test, const cv::Mat &reference);
double getCV(const cv::Mat &test);
double getNR(const cv::Mat &test);
//void createMatrix(double **&matrix, cv::Size size);
//void releaseMatrix(double **&matrix, cv::Size size);
double getSMD2(const cv::Mat &test);
double getHVSNR(const cv::Mat &test, const cv::Mat &reference, int nlevels);