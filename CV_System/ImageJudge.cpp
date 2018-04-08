#include "stdafx.h"
#include "ImageJudge.h"

using namespace cv;
using namespace std;
double getMSE(const Mat& test, const Mat& reference)//MSE=sum((test-eval)^2)/(m*n)
{
	assert(test.rows == reference.rows && test.cols == reference.cols);
	assert(test.channels() == 1 && reference.channels() == 1);

	Mat _test, _reference, diff;
	test.convertTo(_test, CV_32FC1);
	reference.convertTo(_reference, CV_32FC1);
	absdiff(_test, _reference, diff);
	Mat multiple = diff.mul(diff);
	Scalar t = mean(multiple);
	return t.val[0];
}

double getPSNR(const Mat& test, const cv::Mat &reference)//PSNR=10log(max(test)/mse)
{
	double mse = getMSE(test, reference);
	const uchar* ix;
	int temp = 0;
	for (int i = 0; i < test.rows; ++i)
	{
		ix = test.ptr<uchar>(i);
		for (int j = 0; j < test.cols; ++j)
		{
			if (ix[j] > temp)
			{
				temp = ix[j];
			}
		}
	}
	return 10 * log10(temp*temp / (mse + 0.000001)); //add a small number avoiding the divider to be zero
}

double getSSIM(const Mat& test, const Mat& reference)//SSIM=(2*mean_test*mean_ref+c1)*(2*cov_test_ref+c2)/
//     ((mean_test^2+mean_ref^2+c1)*(sd_test^2+sd_ref^2+c2))
{
	const double C1 = 6.5025, C2 = 58.5525;

	assert(test.rows == reference.rows && test.cols == reference.cols);
	assert(test.channels() == 1 && reference.channels() == 1);


	Mat _test, _reference;
	test.convertTo(_test, CV_32FC1);
	reference.convertTo(_reference, CV_32FC1);
	Mat multiple = _test.mul(_reference);
	Scalar mean_test, mean_ref, mean_mul;

	mean_test = mean(_test);
	mean_ref = mean(_reference);
	mean_mul = mean(multiple);

	double cov = mean_mul.val[0] - mean_test.val[0] * mean_ref.val[0];
	double temp1 = (2 * mean_test.val[0] * mean_ref.val[0] + C1)*(2 * cov + C2);

	Mat test2 = _test.mul(_test);
	Mat ref2 = _reference.mul(_reference);
	Scalar mean_test2, mean_ref2;

	mean_test2 = mean(test2);
	mean_ref2 = mean(ref2);

	double temp2 = (mean_test.val[0] * mean_test.val[0] + mean_ref.val[0] * mean_ref.val[0] + C1)*(mean_test2.val[0] - mean_test.val[0] * mean_test.val[0] + mean_ref2.val[0] - mean_ref.val[0] * mean_ref.val[0] + C2);
	return temp1 / temp2;
}

double getCV(const Mat& test)//mean(Laplacian(iamge))
{
	Mat temp(test.rows, test.cols, CV_8UC1);
	if (test.channels() != 1)
	{
		cout << "ERROR: the tested or reference image should be gray style!" << endl;
		return -1;
	}
	Laplacian(test, temp, test.depth());
	Scalar t = mean(temp);
	return t.val[0];
}

double getNR(const Mat& test)//nr=mean|deNoise-noise|/mean(deNoise)
{
	Mat deNoise, diff;
	//GaussianBlur(test, deNoise)
	//three kinds of noise: Gaussian, Spotted, Salt and Pepper, how to decide which kind it is?
	medianBlur(test, deNoise, 3);
	absdiff(deNoise, test, diff);
	Scalar t1 = mean(diff);
	Scalar t2 = mean(deNoise);
	return t1.val[0] / t2.val[0];
}

void createMatrix(double **&matrix, Size size)
{
	matrix = new double*[size.height];
	for (int i = 0; i < size.height; ++i)
	{
		matrix[i] = new double[size.width]();
	}
}

void releaseMatrix(double **&matrix, Size size)
{
	for (int i = 0; i < size.height; ++i)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
}

double getSMD2(const Mat& test)//SMD2=sum(|test(x,y)-test(x+1,y)|*|test(x,y)-test(x,y+1)|)
{
	double temp = 0;
	const uchar* ix;
	const uchar* ix1;
	for (int i = 0; i < test.rows - 1; ++i)
	{
		ix = test.ptr<uchar>(i);
		ix1 = test.ptr<uchar>(i + 1);
		for (int j = 0; j < test.cols - 1; ++j)
		{
			temp += abs(ix[j] - ix1[j])*abs(ix[j] - ix[j + 1]);
		}
	}
	return temp / (test.rows * test.cols);
}

double **fwt97(double** matrix, int width, int height)
{
	//9 / 7 Coefficients:
	double a1 = -1.586134342;
	double a2 = -0.05298011854;
	double a3 = 0.8829110762;
	double a4 = 0.4435068522;

	//Scale coeff:
	double k1 = 0.81289306611596146; // 1 / 1.230174104914
	double k2 = 0.61508705245700002; // 1.230174104914 / 2

	for (int col = 0; col < width; ++col)
	{
		//Predict 1. y1
		for (int row = 1; row < height - 1; row += 2)//奇数列
		{
			matrix[row][col] += a1 * (matrix[row - 1][col] + matrix[row + 1][col]);
		}
		matrix[height - 1][col] += 2 * a1 * matrix[height - 2][col];

		//Update 1. y0
		for (int row = 2; row < height; row += 2)//偶数列
		{
			matrix[row][col] += a2 * (matrix[row - 1][col] + matrix[row + 1][col]);//这里注意不要越界
		}
		matrix[0][col] += 2 * a2 * matrix[1][col];

		//Predict 2.
		for (int row = 1; row < height - 1; row += 2)//奇数列
		{
			matrix[row][col] += a3 * (matrix[row - 1][col] + matrix[row + 1][col]);
		}
		matrix[height - 1][col] += 2 * a3 * matrix[height - 2][col];

		//Updata 2.
		for (int row = 2; row < height; row += 2)//偶数列
		{
			matrix[row][col] += a4 * (matrix[row - 1][col] + matrix[row + 1][col]);//
		}
		matrix[0][col] += 2 * a4 * matrix[1][col];
	}

	double **temp;
	createMatrix(temp, Size(width, height));
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			if (row % 2 == 0)
				temp[col][row / 2] = k1 * matrix[row][col];
			else
				temp[col][row / 2 + height / 2] = k2 * matrix[row][col];
		}
	}
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			matrix[row][col] = temp[row][col];
		}
	}

	releaseMatrix(temp, Size(width, height));

	return matrix;
}

Mat fwt97_2d(Mat image, int nlevels)
{
	int iWidth = image.rows, iHeight = image.cols;
	double **matrix;
	createMatrix(matrix, image.size());

	//convert mat to 2d matrix
	const uchar *ix;
	for (int row = 0; row < iHeight; ++row)
	{
		ix = image.ptr<uchar>(row);
		for (int col = 0; col < iWidth; ++col)
		{
			matrix[row][col] = double(uchar(ix[col]));
		}
	}

	int width = iWidth, height = iHeight;
	//do the wavelet decompose
	for (int i = 0; i < nlevels; ++i)
	{
		matrix = fwt97(matrix, width, height);
		matrix = fwt97(matrix, width, height);
		width /= 2;
		height /= 2;
	}

#ifdef SHOW_WAVELET
	Mat im1(image.size(), CV_8UC1);
	for (int row = 0; row < iHeight; ++row)
	{
		for (int col = 0; col < iWidth; ++col)
		{
			if (matrix[row][col] < 0)
				im1.at<uchar>(row, col) = 0;
			else if (matrix[row][col] > 255)
				im1.at<uchar>(row, col) = 255;
			else
				im1.at<uchar>(row, col) = uchar(matrix[row][col]);
		}
	}
	imshow("97wavelet", im1);
#endif
	//multiple the CSF coefficient with different frequence band
	double csf[4] = { 2.16, 2.87, 3.16, 2.56 };
	for (int i = 0; i < nlevels; ++i)
	{
		int tHeight = 0, tWidth = 0;
		for (int row = tHeight; row < height; ++row)
		{
			for (int col = tWidth; col < width; ++col)
			{
				matrix[row][col] = csf[i] * matrix[row][col];
			}
		}
		tWidth = width;
		tHeight = height;
		width *= 2;
		height *= 2;
	}

	Mat im(image.size(), CV_64FC1);
	for (int row = 0; row < iHeight; ++row)
	{
		double * dm = im.ptr<double>(row);
		for (int col = 0; col < iWidth; ++col)
		{
			dm[col] = matrix[row][col];
		}
	}
	releaseMatrix(matrix, image.size());
	return im;
}

double getHVSNR(const Mat &test, const Mat &reference, int nlevels)
{
	Mat _test(test.size(), CV_64FC1), _ref(reference.size(), CV_64FC1);
	_test = fwt97_2d(test, nlevels);
	_ref = fwt97_2d(reference, nlevels);

	//Minkovski nonlinear summation
	Mat diff(test.size(), CV_64FC1), powImg(test.size(), CV_64FC1);
	absdiff(_test, _ref, diff);
	pow(diff, 4, powImg);

	double temp = 0;
	temp = mean(diff).val[0];
	temp = pow(temp, 1. / 4);

	return 10 * log10(255 * 255 / (temp + 0.000001)); //add a small number avoiding the divider to be zero
}