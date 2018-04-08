#define _CHESSBOARD
#ifdef _CHESSBOARD
#include<opencv2/opencv.hpp>
#endif

struct chessBoard
{
	int length;
	std::vector<cv::Mat> board;
	chessBoard() :length(0) {};
	//~chessBoard(){ releaseMatrix(board, cv::Size(row, col)); };
};

class ChessCorners
{
public:
	ChessCorners();
	~ChessCorners();

	float **imgMagnitude; //gradient magnitude
	float **imgAngle; //edge angle
	int height; //
	int width; //
	std::vector<cv::Point2f> points;
	std::vector<cv::Point2f> vec1;
	std::vector<cv::Point2f> vec2;
	std::vector<float> scores;
	cv::Mat imgCorners;
	void findCorners(cv::Mat I, double tau, bool refine_corner);
	void chessboardsFromCorners();

	//belong to findCorners function
	void createCorrelationPatch(cv::Mat *model, double angle1, double angle2, double radius);
	float normpdf(double x, double mu, double sigma);
	void nonMaximumSuppression(const cv::Mat &corners, int n, double tau, int margin);
	void refineCorners(cv::Mat img_du, cv::Mat img_dv, float **angle, float **magnitude, int r);
	std::vector<cv::Point2f> findModesMeanShift(float *hist, int sigma);
	double root(double **m);
	void scoreCorners(cv::Mat image, float **angle, float **mag, int (&radius)[3]);
	float cornerCorrelationScore(cv::Mat roi, cv::Mat angle_roi, cv::Point2f p1, cv::Point2f p2);
	float stdDeviation(cv::Mat image, float m);

	//belong to chessboardsFromCorners function
	chessBoard chessboards;
	bool initChessboard(int idx, int** &m);
	float directionalNeighbor(int idx, cv::Point2f v, int **m, int &index);
	bool stdMean(float *dist, int s);
	float chessboardEnergy(cv::Mat m);
	bool anyZero(std::vector<int> s);
	cv::Mat growChessboard(cv::Mat m, int type);
	std::vector<int> assignClosestCorners(std::vector<cv::Point2f> cand, std::vector<cv::Point2f> pred, int length);
	//void plotChessboard();

	void drawCorners(cv::Mat image);
};

template<typename T>  void createMatrix(const T **&matrix, cv::Size size);
template<typename T>  void releaseMatrix(const T **&matrix, int size);
void valuation(cv::Mat image, int** m);