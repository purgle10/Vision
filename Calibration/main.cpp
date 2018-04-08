#include"ChessBoard.h"
#include<opencv2\opencv.hpp>
#include<time.h>
#include<string>
#include<iostream>

using namespace cv;

void harrisCorner(Mat &src, Mat &dst, float k)
{
	int aperture = 3; //Sobel kernel size
	Mat src_dx(src.size(), CV_32FC1);
	Mat src_dy(src.size(), CV_32FC1);

	Sobel(src, src_dx, CV_32FC1, 1, 0, aperture);
	Sobel(src, src_dy, CV_32FC1, 0, 1, aperture);

	Mat cov(src.size(), CV_32FC3);
	for (int i = 0; i < src.rows; ++i)
	{
		float* fcov = cov.ptr<float>(i);
		const float* fx = src_dx.ptr<float>(i);
		const float* fy = src_dy.ptr<float>(i);
		int j = 0;
		for (; j < src.cols; ++j)
		{
			float x = fx[j];
			float y = fy[j];

			fcov[j * 3] = x * x;
			fcov[j * 3 + 1] = x * y;
			fcov[j * 3 + 2] = y * y;
		}
	}

	boxFilter(cov, cov, cov.depth(), Size(2, 2));

	for (int i = 0; i < cov.rows; ++i)
	{
		float *fdst = dst.ptr<float>(i);
		const float *fcov = cov.ptr<float>(i);
		for (int j = 0; j < cov.cols; ++j)
		{
			float a = fcov[j * 3];
			float b = fcov[j * 3 + 1];
			float c = fcov[j * 3 + 2];

			fdst[j] = a*c - b*b - k*(a + c)*(a + c);
		}
	}
}

int main(int *argv, char **argc)
{
	//int ksize = 7;//Gaussian blur size, k = 0 means computed from sigma, k should be odd
	//double k = 0.04; //Harris corner response empirically determined constant: 0.04~0.06
	//double thresh = 180;
	//double sigma = 2.0;
	//Mat src;
	//std::string srcName = "cb01.bmp";
	//src = imread(srcName, CV_LOAD_IMAGE_GRAYSCALE);
	//Mat dst(src.size(), CV_32FC1), dst_norm, dst_norm_scaled;

	//try
	//{
	//	harrisCorner(src, dst, k);

	//	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//	convertScaleAbs(dst_norm, dst_norm_scaled);

	//	for (int i = 1; i < dst_norm.rows - 1; ++i)
	//	{
	//		float *fdst = dst_norm.ptr<float>(i);
	//		for (int j = 1; j < dst_norm.cols - 1; ++j)
	//		{
	//			if (fdst[j] > thresh &&
	//				fdst[j] >= (fdst - 1)[j - 1] &&
	//				fdst[j] >= (fdst - 1)[j] &&
	//				fdst[j] >= (fdst - 1)[j + 1] &&
	//				fdst[j] >= fdst[j - 1] &&
	//				fdst[j] >= fdst[j + 1] &&
	//				fdst[j] >= (fdst + 1)[j - 1] &&
	//				fdst[j] >= (fdst + 1)[j] &&
	//				fdst[j] >= (fdst + 1)[j + 1])
	//			{
	//				circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
	//				circle(src, Point(j, i), 5, Scalar(255), -1, 8, 0);
	//			}
	//		}
	//	}
	//}
	//catch (cv::Exception &e)
	//{
	//	const char* err_msg = e.what();
	//	std::cout << err_msg << std::endl;
	//}
	///// Showing the result  
	//imshow("Harris corner", dst_norm_scaled);
	//imshow(srcName, src);
	//waitKey();
	Mat src;
	src = imread("cb01.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	ChessCorners chess;

	clock_t start = clock();
	chess.findCorners(src, 0.01, 1);
	clock_t end = clock();
	printf("the findCorners time is %d ms\n", end - start);

	//start = clock();
	chess.chessboardsFromCorners();
	//end = clock();
	//printf("the chessboardsFromCorners time is %d ms\n", end - start);

	//start = clock();
	chess.drawCorners(src);
	//end = clock();
	//printf("the drawCorners time is %d ms\n", end - start);
	imshow("chessBoard", src);
	waitKey();
	return 0;
}