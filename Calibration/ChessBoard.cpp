#include"ChessBoard.h"
#include<opencv2/opencv.hpp>
#include<vector>
#include<cmath>
#include<algorithm>
#include<time.h>

#define PI 3.1416
#define ONE_OVER_SQRT_2PI 0.39894228
using namespace cv;

template<typename T> 
void createMatrix(T **&matrix, Size size)
{
	matrix = new T*[size.height];
	for (int iInd = 0; iInd < size.height; iInd++)
		matrix[iInd] = new T[size.width]();//initialize to zero.When use this function to create a matGradMag matrix,
										   //we could avoid repeated computing by deciding whether the element is equal to 0.(Kind of memoziation)
}

template<typename T> 
void releaseMatrix(T **&matrix, Size size)
{
	for (int iInd = 0; iInd < size.height; iInd++)
		delete[] matrix[iInd];
	delete[] matrix;
}

void valueToMat(Mat image, int **m)
{
	for (int i = 0; i < image.rows; ++i)
	{
		for (int j = 0; j < image.cols; ++j)
		{
			image.at<char>(i, j) = m[i][j];
		}
	}
}

ChessCorners::ChessCorners()
{

}

ChessCorners::~ChessCorners()
{
	releaseMatrix(imgAngle, Size(width, height));
	releaseMatrix(imgMagnitude, Size(width, height));
}


float ChessCorners::normpdf(double x, double mu, double sigma)
{
	return (ONE_OVER_SQRT_2PI / sigma)*exp(-0.5*pow((x - mu) / sigma, 2));
}

void ChessCorners::createCorrelationPatch(Mat *model, double angle1, double angle2, double radius)
{
	//int width = int(radius) * 2 + 1;
	int Width = int(radius) * 2 + 1;
	int Height = int(radius) * 2 + 1;

	//std::vector<Mat> templ;
	//Mat model[4];
	model[0] = Mat::zeros(Height, Width, CV_32FC1);
	model[1] = Mat::zeros(Height, Width, CV_32FC1);
	model[2] = Mat::zeros(Height, Width, CV_32FC1);
	model[3] = Mat::zeros(Height, Width, CV_32FC1);

	// midpoint
	int mu = int(radius) + 1;
	int mv = int(radius) + 1;

	// compute normals form angles
	double n1[2] = { -sin(angle1), cos(angle1) };
	double n2[2] = { -sin(angle2), cos(angle2) };

	for (int u = 1; u < Width + 1; ++u)
	{
		for (int v = 1; v < Height + 1; ++v)
		{
			double vec[2] = { u - mu, v - mv };
			double dist = sqrt((u - mu)*(u - mu) + (v - mv)*(v - mv));
			double s1 = vec[0] * n1[0] + vec[1] * n1[1];
			double s2 = vec[0] * n2[0] + vec[1] * n2[1];

			if (s1 <= -0.1 && s2 <= -0.1)
				model[0].at<float>(v - 1, u - 1) = normpdf(dist, 0, radius / 2.0);
			else if (s1 >= 0.1 && s2 >= 0.1)
				model[1].at<float>(v - 1, u - 1) = normpdf(dist, 0, radius / 2.0);
			else if (s1 <= -0.1 && s2 >= 0.1)
				model[2].at<float>(v - 1, u - 1) = normpdf(dist, 0, radius / 2.0);
			else if (s1 >= 0.1 && s2 <= -0.1)
				model[3].at<float>(v - 1, u - 1) = normpdf(dist, 0, radius / 2.0);
		}
	}

	model[0] = model[0] / sum(model[0]).val[0];
	flip(model[0], model[0], -1);
	model[1] = model[1] / sum(model[1]).val[0];
	flip(model[0], model[0], -1);
	model[2] = model[2] / sum(model[2]).val[0];
	flip(model[0], model[0], -1);
	model[3] = model[3] / sum(model[3]).val[0];
	flip(model[0], model[0], -1);
}

void ChessCorners::nonMaximumSuppression(const Mat &image, int n, double tau, int margin)
{
	points.clear();
	Mat corners;
	image.convertTo(corners, CV_32FC1);
	for (int i = n + margin; i < width - n - margin; i += (n + 1))
	{
		for (int j = n + margin; j < height - n - margin; j += (n + 1))
		{
			int maxi = i, maxj = j;
			float maxval = corners.at<float>(j, i);
			float currval;
			for (int i2 = i; i2 < i + n + 1; ++i2)
			{
				for (int j2 = j; j2 < j + n + 1; ++j2)
				{
					currval = corners.at<float>(j2, i2);
					if (currval > maxval)
					{
						maxi = i2;
						maxj = j2;
						maxval = currval;
					}
				}
			}

			bool failed = 0;
			int w = min(maxi + n, width - margin-1);
			int h = min(maxj + n, height - margin-1);
			for (int i2 = maxi - n; i2 <= w; ++i2)
			{
				for (int j2 = maxj - n; j2 <= h; ++j2)
				{
					currval = corners.at<float>(j2, i2);
					if ((currval > maxval) && (i2<i || i2>i + n || j2<j || j2 >j + n))
					{
						failed = true;
						break;
					}
				}
				if (failed)
					break;
			}
			if (maxval >= tau && !failed)
			{
				Point2f p = Point2f(maxi, maxj);
				points.push_back(p);
			}
		}
	}
}
bool decent(const Point2f &m1, const Point2f &m2)
{
	return m1.y > m2.y;
}
std::vector<Point2f> ChessCorners::findModesMeanShift(float *hist, int sigma)
{
	std::vector <Point2f> modes;
	float *hist_smoothed = new float[32]();

	for (int i = 0; i < 32; ++i)
	{
		float temp = 0;
		for (int j = -2; j < 3; ++j)
		{
			temp += hist[((i + j) < 0?(32+i+j):(i+j)) % 32] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = temp;
	}

	for (int i = 0; i < 32; ++i)
	{
		bool same = false;
		int j = i;
		while (1)
		{
			float h0 = hist_smoothed[j];
			int j1 = j%32 + 1;
			int j2 = ((j - 1)<0?(30+j):(j-2)) % 32 + 1;
			float h1 = hist_smoothed[j1];
			float h2 = hist_smoothed[j2];
			if (h1 >= h0 && h1 >= h2)
				j = j1;
			else if (h2 > h0 && h2 > h1)
				j = j2;
			else
				break;
		}
		if (modes.empty())
		{
			modes.push_back(Point2f(float(j), hist_smoothed[j]));
		}
		for (int k = 0; k < modes.size(); ++k)
		{
			if (int(modes[k].x) == j)
			{
				same = true;
				break;
			}
		}
		if (!same)
		{
			modes.push_back(Point2f(float(j), hist_smoothed[j]));
		}
	}
	delete hist_smoothed;
	sort(modes.begin(), modes.end(), decent);
	return modes;
}

double ChessCorners::root(double **m)//return the minimum value as the eigenvalue
{
	double a, b, c, rambda;
	a = 1.0;
	b = -m[0][0] - m[1][1];
	c = m[0][0] * m[1][1] - m[0][1] * m[1][0];

	double t = b*b - 4 * a*c;
	if (t == 0)
		rambda = -b / (2 * a);
	else
		rambda = (-b - sqrt(t)) / (2 * a);

	return (rambda - m[0][0]) / m[0][1];
}

void ChessCorners::refineCorners(cv::Mat img_du, Mat img_dv, float **angle, float **mag, int r)
{
	//assert(points.size() != 0);

	//init orientations to invalid (corner is invalid if orientation=0)
	vec1.clear();
	vec2.clear();

	for (int i = 0; i < points.size(); ++i)
	{
		float *angle_hist = new float[32]();
		int cu = points[i].x;
		int cv = points[i].y;

		//estimate edge orientations
		int initv = max(cv - r, 0);
		int initu = max(cu - r, 0);
		int h = min(cv + r, height-1);
		int w = min(cu + r, width-1);
		float vec_angle;
		for (int col = initu; col <= w ; ++col)
		{
			for (int row = initv; row <= h; ++row)
			{
				vec_angle = angle[row][col] + PI / 2;
				if (vec_angle > PI)
					vec_angle -= PI;
				int bin = max(min(int(floor(vec_angle / (PI/32))), 31), 0);
				angle_hist[bin] += mag[row][col]; 
			}
		}
		std::vector<Point2f> modes;
		modes = findModesMeanShift(angle_hist, 1);

		delete angle_hist;

		if (modes.size() < 2)
		{
			vec1.push_back(Point2f(0, 0));
			vec2.push_back(Point2f(0, 0));
			continue;
		}
		for (int idx = 0; idx < modes.size(); ++idx)
			modes[idx].x = modes[idx].x * PI / 32.0;

		double delta_angle;
		if (modes[0].x > modes[1].x)
			delta_angle = min(modes[0].x - modes[1].x, float(PI + modes[1].x - modes[0].x));
		else
			delta_angle = min(modes[1].x - modes[0].x, float(PI + modes[0].x - modes[1].x));
		if (delta_angle <= 0.3)
		{
			vec1.push_back(Point2f(0, 0));
			vec2.push_back(Point2f(0, 0));
			continue;
		}
		Point2f p1 = Point2f(cos(modes[0].x), sin(modes[0].x));
		Point2f p2 = Point2f(cos(modes[1].x), sin(modes[1].x));
		if (modes[0].x > modes[1].x)
		{
			vec1.push_back(p2);
			vec2.push_back(p1);
		}
		else
		{
			vec1.push_back(p1);
			vec2.push_back(p2);
		}
		if ((vec1.back().x == 0 && vec1.back().y == 0) || (vec2.back().x == 0 && vec2.back().y == 0))
			continue;
		// corner orientation refinement

		double **A1;
		double **A2;
		createMatrix(A1, Size(2, 2));
		createMatrix(A2, Size(2, 2));
		
		for (int u = initu; u <= w; ++u)
		{
			for (int v = initv; v <= h; ++v)
			{
				float mg = mag[v][u];
				if (mg < 0.1)
					continue;
				float du = img_du.at<float>(v,u);
				float dv = img_dv.at<float>(v,u);
				//robust refinement of orientation 1
				if ((abs(du * vec1.back().x + dv * vec1.back().y) / mg) < 0.25) //inlier?
				{
					A1[0][0] += du * du;
					A1[0][1] += du * dv;
					A1[1][0] += dv * du;
					A1[1][1] += dv * dv;
				}

				//robust refinement of orientation 1
				if ((abs(du * vec2.back().x + dv * vec2.back().y) / mg) < 0.25) //inlier?
				{
					A2[0][0] += du * du;
					A2[0][1] += du * dv;
					A2[1][0] += dv * du;
					A2[1][1] += dv * dv;
				}
			}
		}
		float x1 = root(A1);
		float x2 = root(A2);
		float v11 = 1.0 / sqrt(1 + x1*x1);
		float v12 = x1 / sqrt(1 + x1*x1);
		float v21 = 1.0 / sqrt(1 + x2*x2);
		float v22 = x2 / sqrt(1 + x2*x2);
		if (v11*v12 > 0)//?
		{
			v11 = -v11;
			v12 = -v12;
		}
		if (v21*v22 > 0)//?
		{
			v21 = -v21;
			v22 = -v22;
		}
		else if (v21*v22 < 0)
		{
			if (abs(v21) > abs(v22))
			{
				v21 = -v21;
				v22 = -v22;
			}
		}

		vec1.back().x = v11;
		vec1.back().y = v12;
		vec2.back().x = v21;
		vec2.back().y = v22;

		//corner location refinement
		double **G;
		createMatrix(G, Size(2, 2));
		double b[2] = { 0, 0 };
		for (int u = initu; u <= w; ++u)
		{
			for (int v = initv; v <= h; ++v)
			{
				float mg = mag[v][u];

				if (mg < 0.1)
					continue;
				float du = img_du.at<float>(v, u);
				float dv = img_dv.at<float>(v, u);
				if (u != cu || v != cv)
				{
					int w1 = u - cu;
					int w2 = v - cv;
					double d1 = sqrt((w1 - w1*v11*v11 - w2 * v12*v11)*(w1 - w1*v11*v11 - w2 * v12*v11) +
						(w2 - w1*v11*v12 - w2 * v12*v12)*(w2 - w1*v11*v12 - w2 * v12*v12));
					double d2 = sqrt((w1 - w1*v21*v21 - w2 * v22*v21)*(w1 - w1*v21*v21 - w2 * v22*v21) +
						(w2 - w1*v21*v22 - w2 * v22*v22)*(w2 - w1*v21*v22 - w2 * v22*v22));
					if ((d1 < 3 && abs(du * v11 + dv * v12) / mg < 0.25) ||
						(d2 < 3 && abs(du * v21 + dv * v22) / mg < 0.25))
					{
						G[0][0] += du*du;
						G[0][1] += dv*du;
						G[1][0] = G[0][1];
						G[1][1] += dv*dv;
						b[0] += du*du*(u+1) + du*dv*(v+1);
						b[1] += du*dv*(u+1) + dv*dv*(v+1);
					}
				}
			}
		}

		// set new corner location if G has full rank
		float g = (G[0][0] * G[1][1] - G[1][0] * G[0][1]);
		if (abs(g) > 0.0001)
		{
			float n2 = (b[1] * G[0][0] - b[0] * G[1][0]) / g;
			float n1 = (b[0] - G[0][1] * n2) / (G[0][0] + 0.000000001);
			float old1 = points[i].x+1;
			float old2 = points[i].y+1;
			points[i].x = n1-1;
			points[i].y = n2-1;
			if (sqrt((old1 - n1)*(old1 - n1) + (old2 - n2)*(old2 - n2)) >= 4)
			{
				vec1[i] = Point2f(0, 0);
				vec2[i] = Point2f(0, 0);
			}
		}
		else //set corner to invalid
		{
			vec1[i] = Point2f(0, 0);
			vec2[i] = Point2f(0, 0);
		}
		releaseMatrix(A1, Size(2, 2));
		releaseMatrix(A2, Size(2, 2));
		releaseMatrix(G, Size(2, 2));
	}
}

float ChessCorners::stdDeviation(Mat image, float m)
{
	Mat s(image.size(), CV_32FC1), p;
	s = image - m;
	p = s.mul(s);
	float result = sqrt(sum(p).val[0] / (image.rows*image.cols - 1));
	return result;
}

float ChessCorners::cornerCorrelationScore(cv::Mat roi, cv::Mat mag_roi, cv::Point2f v1, cv::Point2f v2)
{
	int row = roi.rows;
	int col = roi.cols;
	int cx = (row + 1) / 2;
	int cy = cx;
	Mat img_filter = -Mat::ones(Size(row,col), CV_32FC1);

	// compute gradient filter kernel(bandwith = 3 px)
	for (int x = 0; x < mag_roi.cols; ++x)
	{
		for (int y = 0; y < mag_roi.rows; ++y)
		{
			int p11 = x + 1 - cx;
			int p12 = y + 1- cy;
			float v11_2 = v1.x*v1.x;
			float v12 = v1.x*v1.y;
			float v12_2 = v1.y*v1.y;
			float v21_2 = v2.x*v2.x;
			float v21 = v2.x*v2.y;
			float v22_2 = v2.y*v2.y;
			float result11 = p11  - p11*v11_2 - p12*v12;
			float result12 = p12 - p11*v12 - p12*v12_2;
			float result21 = p11  - p11*v21_2 - p12*v21;
			float result22 = p12 - p11*v21 - p12*v22_2;
			if (sqrt(result11*result11 + result12*result12) <= 1.5 || sqrt(result21*result21 + result22*result22) <= 1.5)
			{
				img_filter.at<float>(y, x) = 1;
			}
		}
	}

	// convert into vectors
	float mag_mean = mean(mag_roi).val[0];
	float filter_mean = mean(img_filter).val[0];
	// normalize
	mag_roi = (mag_roi - mag_mean) / stdDeviation(mag_roi, mag_mean);
	img_filter = (img_filter - filter_mean) / stdDeviation(img_filter, filter_mean);

	// compute gradient score
	Mat mutil;
	mutil = mag_roi.mul(img_filter);
	float score_gradient = max(sum(mutil).val[0] / (row*col - 1.0), 0.0);

	// create intensity filter kernel
	Mat model[4];
	createCorrelationPatch(model, atan2f(v1.y, v1.x), atan2f(v2.y, v2.x), cx - 1);

	// checkerboard responses
	float a1 = sum(model[0].mul(roi)).val[0];
	float a2 = sum(model[1].mul(roi)).val[0];
	float b1 = sum(model[2].mul(roi)).val[0];
	float b2 = sum(model[3].mul(roi)).val[0];

	// mean
	float mu = (a1 + a2 + b1 + b2) / 4;

	// case 1: a=white, b=black
	float score_a = min(a1 - mu, a2 - mu);
	float score_b = min(mu - b1, mu - b2);
	float score1 = min(score_a, score_b);

	// case 2: b=white, a=black
	score_a = min(mu - a1, mu - a2);
	score_b = min(b1 - mu, b2 - mu);
	float score2 = min(score_a, score_b);

	//intensity score: max. of the 2 cases
	double t = max(score1, score2);
	float score_intensity = max(t, 0.0);

	//final score: product of gradient and intensity score
	return score_gradient * score_intensity;
}

void ChessCorners::scoreCorners(Mat image, float **angle, float ** mag, int (&radius)[3])
{
	//Mat Mag(Size(width, height), CV_32FC1, mag);
	//fAngle = Mat_(angle);
	for (int i = 0; i < points.size(); ++i)
	{
		int u = round(points[i].x);
		int v = round(points[i].y);
		float *score = new float[sizeof(radius) / sizeof(radius[0])]();
		float temp = 0;
		for (int j = 0; j < sizeof(radius) / sizeof(radius[0]); ++j)
		{
			if (u >= radius[j] && u < width - radius[j] && v >= radius[j] && v < height - radius[j])
			{
				Mat roi(Size(2 * radius[j] + 1, 2 * radius[j] + 1), CV_32FC1), mag_roi(Size(2 * radius[j] + 1, 2 * radius[j] + 1), CV_32FC1);

				for (int m = 0; m < roi.rows; m++)
				{
					for (int n = 0; n < roi.cols; n++)
					{
						roi.at<float>(m, n) = image.at<float>(v - radius[j] + m, u - radius[j] + n);
						mag_roi.at<float>(m, n) = mag[v - radius[j] + m][u - radius[j] + n];
					}
				}
				score[j] = cornerCorrelationScore(roi, mag_roi, vec1[i], vec2[i]);
			}
			if (score[j] > temp)
				temp = score[j];
		}

		// take highest score
		scores.push_back(temp);
		delete []score;
	}
}

void ChessCorners::findCorners(Mat image, double tau, bool refine_corners = true)
{
	CV_Assert(image.type() == CV_8UC1);
	//clock_t start, end;
	//start = clock();
	height = image.rows;
	width = image.cols;
	Size sSize = image.size();
	Mat img;
	Mat img_du, img_dv;
	image.convertTo(img, CV_32FC1);
	img = img / 255;

	createMatrix(imgMagnitude, sSize);
	createMatrix(imgAngle, sSize);

	int radius[3] = { 4, 8, 12 };

	Mat sobel = (Mat_<char>(3,3)<<-1, 0, 1,
									-1, 0, 1,
									-1, 0, 1);
	Mat isobel = (Mat_<char>(3, 3) << -1, -1, -1,
										 0,  0,  0,
										 1,  1,  1);

	filter2D(img, img_du, img.depth(), sobel, Point(-1,-1), 0.0, BORDER_CONSTANT);
	filter2D(img, img_dv, img.depth(), isobel, Point(-1,-1), 0.0, BORDER_CONSTANT);

	img_du = -img_du;
	img_dv = -img_dv;
	float fu, fv, angle;
	for (int row = 0; row < height; ++row)
	{
		const float* fdu = img_du.ptr<float>(row);
		const float* fdv = img_dv.ptr<float>(row);
		for (int col = 0; col < width; ++col)
		{
			fu = fdu[col];
			fv = fdv[col];  // read x, y derivatives

			imgMagnitude[row][col] = sqrt(fu*fu + fv*fv);
			angle = atan2(fv, fu);
			if (angle < 0)
				imgAngle[row][col] = angle + PI;
			else if (angle > PI)
				imgAngle[row][col] = angle - PI;
			else
				imgAngle[row][col] = angle;

		}
	}


	double _min, _max;
	minMaxLoc(img, &_min, &_max);
	img = (img - float(_min)) / float(_max - _min);

	double model_props[6][3] = { 
		{ 0, 1.5708, 4.0 }, {0.7854, -0.7854, 4.0},
		{ 0, 1.5708, 8.0 }, {0.7854, -0.7854, 8.0},
		{ 0, 1.5780, 12.0 }, {0.7854, -0.7854, 12.0}
	};

	imgCorners = Mat::zeros(sSize, CV_32FC1);
	Mat img_corners_mu(sSize, CV_32FC1);
	Mat model[4];
	Mat img_corners[4];
	Mat img_corners_a, img_corners_b, img_corners1, img_corners2, temp1, temp2;
	imgCorners.convertTo(temp1, CV_32FC1);
	for (int i = 0; i < 6; ++i)
	{
		createCorrelationPatch(model, model_props[i][0], model_props[i][1], model_props[i][2]);
		filter2D(img, img_corners[0], img.depth(), model[0], Point(-1, -1), 0.0, BORDER_CONSTANT);
		filter2D(img, img_corners[1], img.depth(), model[1], Point(-1, -1), 0.0, BORDER_CONSTANT);
		filter2D(img, img_corners[2], img.depth(), model[2], Point(-1, -1), 0.0, BORDER_CONSTANT);
		filter2D(img, img_corners[3], img.depth(), model[3], Point(-1, -1), 0.0, BORDER_CONSTANT);
		img_corners_mu = (img_corners[0] + img_corners[1] + img_corners[2] + img_corners[3]) / 4.0;

		//case 1: a = white, b = black
		img_corners[0] = img_corners[0] - img_corners_mu;
		img_corners[1] = img_corners[1] - img_corners_mu;
		img_corners[2] = img_corners_mu - img_corners[2];
		img_corners[3] = img_corners_mu - img_corners[3];
		img_corners_a = min(img_corners[0], img_corners[1]);
		img_corners_b = min(img_corners[2], img_corners[3]);
		img_corners1 = min(img_corners_a, img_corners_b);

		//case 2: b = white, a = black
		img_corners[0] = -img_corners[0];
		img_corners[1] = -img_corners[1];
		img_corners[2] = -img_corners[2];
		img_corners[3] = -img_corners[3];
		img_corners_a = min(img_corners[0], img_corners[1]);
		img_corners_b = min(img_corners[2], img_corners[3]);
		img_corners2 = min(img_corners_a, img_corners_b);
		//update corner map
		temp2 = max(temp1, img_corners1);
		temp1 = max(temp2, img_corners2);

	}
	imgCorners = temp1;

	nonMaximumSuppression(imgCorners, 3, 0.025, 5);

	//printf("size = %d\n", points.size());
	//printf("Refining...\n");

	if (refine_corners)
		refineCorners(img_du, img_dv, imgAngle, imgMagnitude, 10);
	//end = clock();
	//printf("refine time is %d ms\n", end - start);

	//start = clock();
	std::vector<bool> isZero;
	for (int i = 0; i < vec1.size(); ++i)
	{
		isZero.push_back(false);
		if (vec1[i].x == 0 && vec1[i].y == 0)
			isZero[i] = true;
	}
	std::vector<Point2f> tpoints;
	std::vector<Point2f> tvec1;
	std::vector<Point2f> tvec2;
	std::vector<float> tscore;
	for (int i = 0; i < isZero.size(); ++i)
	{
		if (!isZero[i])
		{
			tpoints.push_back(points[i]);
			tvec1.push_back(vec1[i]);
			tvec2.push_back(vec2[i]);
		}
	}
	points.clear();
	vec1.clear();
	vec2.clear();
	points = tpoints;
	vec1 = tvec1;
	vec2 = tvec2;
	tpoints.clear();
	tvec1.clear();
	tvec2.clear();

	scoreCorners(img, imgAngle, imgMagnitude, radius);

	//remove low scoring corners
	std::vector<bool> isGreater;
	for (int i = 0; i < scores.size(); ++i)
	{
		isGreater.push_back(true);
		if (scores[i]>tau)
			isGreater[i] = false;
	}
	for (int i = 0; i < isGreater.size(); ++i)
	{
		if (!isGreater[i])
		{
			tpoints.push_back(points[i]);
			tvec1.push_back(vec1[i]);
			tvec2.push_back(vec2[i]);
			tscore.push_back(scores[i]);
		}
	}
	points.clear();
	vec1.clear();
	vec2.clear();
	scores.clear();
	points = tpoints;
	vec1 = tvec1;
	vec2 = tvec2;
	scores = tscore;

	//make v1 positive
	for (int i = 0; i < vec1.size(); ++i)
	{
		if (vec1[i].x + vec1[i].y < 0)
		{
			vec1[i].x = -vec1[i].x;
			vec1[i].y = -vec1[i].y;
		}

	}
	//make systems right-handed (reduces matching ambiguities from 8 to 4)
	tvec1.clear();
	for (int i = 0; i < vec1.size(); ++i)
	{
		if (vec1[i].y*vec2[i].x - vec1[i].x*vec2[i].y>0)
		{
			vec2[i].x = -vec2[i].x;
			vec2[i].y = -vec2[i].y;
		}
	}
	//end = clock();
	//printf("score time is %d ms\n", end - start);
}

/******** belong to chessboardsFromCorners function *********/

float ChessCorners::directionalNeighbor(int idx, Point2f v, int **m, int &index)
{
	//list of neighboring elements, which are currently not in use
	int *used = new int[9]();
	int count = 0;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (m[i][j] != -1)
			{
				used[count] = m[i][j];
				count++;
			}
		}
	}

	std::vector<Point2f> tpoints;
	std::vector<float> dist;
	std::vector<float> dist_edge;
	tpoints = points;

	for (int i = 0; i < count; ++i)
	{
		tpoints[used[i]] = Point2f(0,0);
	}
	for (int i = 0; i < tpoints.size(); ++i)
	{
		if (tpoints[i].x == 0 && tpoints[i].y == 0)
			continue;
		Point2f n = tpoints[i] - points[idx];
		float temp = n.x * v.x + n.y * v.y;
		dist.push_back(temp);

		n = n - temp * v;
		temp = sqrt(n.x*n.x + n.y*n.y);
		dist_edge.push_back(temp);
	}

	//find best neighbor
	std::vector<float> dist_point;
	for (int i = 0; i < dist.size(); ++i)
	{
		if (dist[i] < 0)
			dist[i] = 9999;
		dist_point.push_back(dist[i] + dist_edge[i] * 5);
	}

	int min_idx;
	float min_dist = 100000;

	for (int i = 0; i < dist_point.size(); ++i)
	{
		if (min_dist > dist_point[i])
		{
			min_dist = dist_point[i];
			min_idx = i;
		}
	}

	count = 0;
	for (int j = 0; j < tpoints.size(); ++j)
	{
		if (tpoints[j].x == 0 && tpoints[j].y == 0)
			continue;
		if (count == min_idx)
		{
			index = j;
			break;
		}
		count++;
	}

	delete [] used;
	return min_dist;
}

bool ChessCorners::stdMean(float *dist, int s)
{
	float dSum = 0.0;
	for (int i = 0; i < s; ++i)
	{
		dSum += dist[i];
	}
	float dMean = dSum / s;
	dSum = 0;

	for (int i = 0; i < s; ++i)
	{
		dSum += (dist[i] - dMean)*(dist[i] - dMean);
	}
	float dStd = sqrt(dSum / (s - 1));
	return dStd / dMean > 0.3;
}

bool ChessCorners::initChessboard(int idx, int **&chessboard)
{
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			chessboard[i][j] = -1;
		}
	}
	// return if not enough corners
	if (points.size() < 9)
	{
		return true;
	}

	//extract feature index and orientation (central element)
	Point2f v1 = vec1[idx];
	Point2f v2 = vec2[idx];
	chessboard[1][1] = idx;

	//find left/right/top/bottom neighbors
	float dist1[2], dist2[6];
	dist1[0] = directionalNeighbor(idx, v1, chessboard, chessboard[1][2]);
	dist1[1] = directionalNeighbor(idx, -v1, chessboard, chessboard[1][0]);
	dist2[0] = directionalNeighbor(idx, v2, chessboard, chessboard[2][1]);
	dist2[1] = directionalNeighbor(idx, -v2, chessboard, chessboard[0][1]);

	//find top-left/top-right/bottom-left/bottom-right neighbors
	dist2[2] = directionalNeighbor(chessboard[1][0], -v2, chessboard, chessboard[0][0]);
	dist2[3] = directionalNeighbor(chessboard[1][0], v2, chessboard, chessboard[2][0]);
	dist2[4] = directionalNeighbor(chessboard[1][2], -v2, chessboard, chessboard[0][2]);
	dist2[5] = directionalNeighbor(chessboard[1][2], v2, chessboard, chessboard[2][2]);

	//initialization must be homogenously distributed
	if (stdMean(dist1, 2) || stdMean(dist2, 6))
	{
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				chessboard[i][j] = -1;
			}
		}
		return true;
	}
	return false;
}

float ChessCorners::chessboardEnergy(Mat mcb)
{
	float structure = 0;
	int row = mcb.rows, col = mcb.cols;
	int **m;
	createMatrix(m, mcb.size());
	for (int j = 0; j < row; ++j)
	{
		for (int k = 0; k < col; ++k)
		{
			m[j][k] = mcb.at<char>(j, k); 
		}
	}

	Point2f n1, n2, n3;
	Point2f x1, x2;
	for (int j = 0; j < row; ++j)
	{
		for (int k = 0; k < col-2; ++k)
		{
			n1 = points[m[j][k]];
			n2 = points[m[j][k+1]];
			n3 = points[m[j][k+2]];
			x1 = n1 + n3 - 2 * n2;
			x2 = n1 - n3;
			float temp = sqrt(x1.x*x1.x + x1.y*x1.y) / sqrt(x2.x*x2.x + x2.y*x2.y);
			structure = max(structure, temp);
		}
	}

	for (int j = 0; j < col; ++j)
	{
		for (int k = 0; k < row-2; ++k)
		{
			n1 = points[m[k][j]];
			n2 = points[m[k+1][j]];
			n3 = points[m[k+2][j]];
			x1 = n1 + n3 - 2 * n2;
			x2 = n1 - n3;
			float temp = sqrt(x1.x*x1.x + x1.y*x1.y) / sqrt(x2.x*x2.x + x2.y*x2.y);
			structure = max(structure, temp);
		}
	}
	releaseMatrix(m, mcb.size());
	return row*col*(structure-1);
}

std::vector<int> ChessCorners::assignClosestCorners(std::vector<Point2f> cand, std::vector<Point2f> pred, int length)
{
	int *idx = new int[length]();
	std::vector<int> index;
	index.clear();
	if (cand.size() < 3)
	{
		return index;
	}

	//build distance matrix
	float **D;
	createMatrix(D, Size(length, cand.size()));//Size(_width, _height);

	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < cand.size(); ++j)
		{
			Point2f temp = cand[j]- pred[i];
			D[j][i] = sqrt(temp.x*temp.x + temp.y*temp.y);
		}
	}

	//search greedily for closet corners
	for (int c = 0; c < length; ++c)
	{
		float minD = 10000;
		int row, col;
		bool found =  false;
		for (int j = 0; j < length; ++j)
		{
			for (int k = 0; k < cand.size(); k++)
			{
				if (minD > D[k][j])
					minD = D[k][j];
			}
		}
		for (int j = 0; j < length; ++j)
		{
			for (int k = 0; k < cand.size(); k++)
			{
				if (D[k][j] == minD)
				{
					row = k;
					col = j;
					found = true;
					break;
				}
			}
			if (found)
				break;
		}
		idx[col] = row;
		for (int k = 0; k < length; k++)
		{
			D[row][k] = 9999;
		}
		for (int k = 0; k < cand.size(); k++)
		{
			D[k][col] = 9999;
		}

	}
	for (int i = 0; i < length; ++i)
	{
		index.push_back(idx[i]);
	}
	delete []idx;
	releaseMatrix(D, Size(length, cand.size()));
	return index;
}

bool ChessCorners::anyZero(std::vector<int> s)
{
	bool flag = false;
	if (s.size() == 0)
		return true;

	auto result = std::find(s.begin(), s.end(), -1);
	if (result != s.cend())
		flag = true;
	return flag;
}

Mat ChessCorners::growChessboard(Mat mcb, int type)
{
	int row = mcb.rows, col = mcb.cols;
	int **m;
	createMatrix(m, mcb.size());

	int *used = new int[row*col]();
	int *unused = new int[points.size()-row*col]();
	int count = 0;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			m[i][j] = mcb.at<char>(i, j);
			if (m[i][j] != -1)
			{
				used[count] = m[i][j];
				count++;
			}
		}
	}
	std::vector<Point2f> tpoints;
	std::vector<Point2f> cand;

	tpoints = points;
	for (int i = 0; i < count; ++i)
	{
		tpoints[used[i]] = Point2f(0, 0);
	}
	count = 0;
	for (int i = 0; i < tpoints.size(); ++i)
	{
		if (tpoints[i].x != 0 && tpoints[i].y != 0)
		{
			Point2f n = tpoints[i];
			cand.push_back(n);
			unused[count] = i;
			count++;
		}
	}
	std::vector<int> idx;
	std::vector<Point2f> pred;

	int **chessboard;
	if (type == 0)
	{
		col++;
		for (int j = 0; j < row; ++j)
		{
			pred.push_back(2 * points[m[j][col - 2]] - points[m[j][col - 3]]);//2*p3-p2
		}
		createMatrix(chessboard, Size(col, row));
		idx = assignClosestCorners(cand, pred, row);
		if (!anyZero(idx))//rewrite
		{
			for (int i = 0; i < row; ++i)
			{
				for (int j = 0; j < col - 1; ++j)
				{
					chessboard[i][j] = m[i][j]; 
				}
				chessboard[i][col - 1] = unused[idx[i]]; 
			}

			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			Mat temp(Size(col, row), CV_8SC1);
			valueToMat(temp, chessboard);
			releaseMatrix(chessboard, Size(col, row));
			return temp;
		}
		else
		{
			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			return mcb;
		}
	}

	else if (type == 1)
	{
		row++;
		for (int j = 0; j < col; ++j)
		{
			pred.push_back(2 * points[m[row - 2][j]] - points[m[row - 3][j]]);
		}
		createMatrix(chessboard, Size(col, row));
		idx = assignClosestCorners(cand, pred, col);
		if (!anyZero(idx))
		{
			for (int i = 0; i < col; ++i)
			{
				for (int j = 0; j < row - 1; ++j)
				{
					chessboard[j][i] = m[j][i];
				}
				chessboard[row - 1][i] = unused[idx[i]];
			}

			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			Mat temp(Size(col, row), CV_8SC1);
			valueToMat(temp, chessboard);
			releaseMatrix(chessboard, Size(col, row));
			return temp;
		}
		else
		{
			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			return mcb;
		}
	}

	else if (type == 2)
	{
		col++;
		for (int j = 0; j < row; ++j)
		{
			pred.push_back(2 * points[m[j][0]] - points[m[j][1]]);
		}
		createMatrix(chessboard, Size(col, row));
		idx = assignClosestCorners(cand, pred, row);
		if (!anyZero(idx))
		{
			for (int i = 0; i < row; ++i)
			{
				chessboard[i][0] = unused[idx[i]];
				for (int j = 0; j < col - 1; ++j)
				{
					chessboard[i][j + 1] = m[i][j];
				}
			}
			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			Mat temp(Size(col, row), CV_8SC1);
			valueToMat(temp, chessboard);
			releaseMatrix(chessboard, Size(col, row));
			return temp;
		}
		else
		{
			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			return mcb;
		}
	}

	else if (type == 3)
	{
		row++;
		for (int j = 0; j < col; ++j)
		{
			pred.push_back(2 * points[m[0][j]] - points[m[1][j]]);
		}
		createMatrix(chessboard, Size(col, row));
		idx = assignClosestCorners(cand, pred, col);
		if (!anyZero(idx))
		{
			for (int i = 0; i < col; ++i)
			{
				chessboard[0][i] = unused[idx[i]];
				for (int j = 0; j < row - 1; ++j)
				{
					chessboard[j + 1][i] = m[j][i];
				}
			}

			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			Mat temp(Size(col, row), CV_8SC1);
			valueToMat(temp, chessboard);
			releaseMatrix(chessboard, Size(col, row));
			return temp;
		}
		else
		{
			delete[]used;
			delete[]unused;
			releaseMatrix(m, mcb.size());
			return mcb;
		}
	}
	return Mat();
}

void ChessCorners::chessboardsFromCorners()
{
	//printf("Structure recovery:\n");
	int **chessboard;

	for (int i = 0; i < points.size(); ++i)
	{
		createMatrix(chessboard, Size(3, 3));
		bool empty = initChessboard(i, chessboard);//chessboard size 3¡Á3

		Mat mboard(Size(3, 3), CV_8SC1);
		valueToMat(mboard, chessboard);

		if (empty || chessboardEnergy(mboard) > 0)
			continue;
		Mat temp = mboard;

		while (true)
		{
			Mat mproposal[4];// = new Mat[4]();
			float energy = chessboardEnergy(temp);
			float pEnergy[4];
			for (int j = 0; j < 4; ++j)
			{
				mproposal[j] = growChessboard(temp, j);
				pEnergy[j] = chessboardEnergy(mproposal[j]);

			}

			int minIdx;
			float minEnergy = 10000;
			for (int j = 0; j < 4; ++j)
			{
				if (minEnergy > pEnergy[j])
				{
					minEnergy = pEnergy[j];
					minIdx = j;
				}
			}

			if (minEnergy < energy)
			{
				temp = mproposal[minIdx];
			}
			else
			{
				break;
			}
		}

		//if chessboard has low energy (corresponding to high quality)
		if (chessboardEnergy(temp) < -10)
		{
			//check if new chessboard proposal overlaps with existing chessboards
			//float **overlap;
			std::vector<float> overlap;
			std::vector<int> index;
			//index.clear();
			if (chessboards.length != 0)
			{
				//createMatrix(overlap, Size(chessboards.length, 2));
				overlap.assign(chessboards.length, 0.0);
				for (int j = 0; j < chessboards.length; ++j)
				{
					int r = chessboards.board[j].rows;
					int c = chessboards.board[j].cols;
					bool bFlag = false;
					for (int m = 0; m < r; ++m)
					{
						for (int n = 0; n < c; ++n)
						{
							int b = chessboards.board[j].at<char>(m, n);
							for (int s = 0; s < temp.rows; ++s)
							{
								for (int t = 0; t < temp.cols; ++t)
								{
									if (b == temp.at<char>(s, t))
									{
										//overlap[j][1] = 1;
										//overlap[j][2] = chessboardEnergy(chessboards.board[j]);
										overlap[j] = chessboardEnergy(chessboards.board[j]);
										index.push_back(j);
										bFlag = true;
										break;
									}
								}
								if (bFlag)
									break;
							}
							if (bFlag)
								break;
						}
						if (bFlag)
							break;
					}
				}
			}
			

			// add chessboard (and replace overlapping if neccessary)
			if(chessboards.length == 0)
			{
				chessboards.board.push_back(temp);
				chessboards.length++;
			}
			else
			{
				for (int k = 0; k < index.size(); ++k)
				{
					if (overlap[k]> chessboardEnergy(temp))
					{
						//chessboards.board[index[k]] = Mat();
						chessboards.board[index[k]] = temp;//
						break;
					}
				}
			}
		}
	}


}

void ChessCorners::drawCorners(Mat image)
{
	if (chessboards.length == 1)
	{
		Mat temp = chessboards.board[0];
		for (int i = 0; i <temp.rows; ++i)
		{
			for (int j = 0; j < temp.cols; ++j)
			{
				int idx = temp.at<char>(i, j);
				Point p;
				p.x = int(points[idx].x);
				p.y = int(points[idx].y);
				circle(image, p, 3, Scalar(0), 1, 1);
			}
		}
	}

}