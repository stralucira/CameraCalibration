#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;
using namespace std;

void startVideoCapture();
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);


int main()
{
	float aspectRatio;

	float squareSize = 1.0f;

	vector<Mat> rvecs, tvecs;

	//Create a 3x3 identity matrix
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	// Distortion coefficients of 8 elements
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	
	// Considering the aspect ratio is fixed (CALIB_FIX_ASPECT_RATIO) set fx/fy
	
	
	//startVideoCapture();

	Size testPatternSize(6, 9);
	vector<Point2f> pointBuffer;

	cv::Mat img = cv::imread("data/chess_test.jpg");

	aspectRatio = (float) img.size().width / (float) img.size().height;

	cameraMatrix.at<double>(0, 0) = aspectRatio;

	bool patternFound = findChessboardCorners(img, testPatternSize, pointBuffer,
		CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
		+ CALIB_CB_FILTER_QUADS);

	drawChessboardCorners(img, testPatternSize, pointBuffer, patternFound);
	cv::imshow("Chess Test", img);
	cv::waitKey(0);

	vector<vector<Point3f>> objectPoints(1);
	vector<vector<Point2f>> imagePoints;

	imagePoints.push_back(pointBuffer);

	calcBoardCornerPositions(testPatternSize, squareSize, objectPoints[0]);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms = calibrateCamera(objectPoints, imagePoints, img.size(), cameraMatrix,
		distCoeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	return 0;
}

// Convert the 2D points obtained from image to 3D coordinates
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
	for (int i = 0; i < boardSize.height; ++i)
		for (int j = 0; j < boardSize.width; ++j)
			//Set z=0 for every point, considering the chessboard is planar.
			corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
}
void startVideoCapture() {
	VideoCapture stream1(0);

	if (!stream1.isOpened()) { //check if video device is initialised
		cout << "cannot open camera";
	}

	while (true) {
		Mat cameraFrame;
		stream1.read(cameraFrame);
		imshow("cam", cameraFrame);
		waitKey(30);
	}
};