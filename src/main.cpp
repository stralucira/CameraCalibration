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

void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);

int boardCount = 4; //Number of boards to be found before calibration.

int main()
{
	float aspectRatio = 16.0f / 9.0f;
	float squareSize = 1.0f;

	vector<Mat> rvecs, tvecs;
	int successes = 0;

	//Create a 3x3 identity matrix
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	// Distortion coefficients of 8 elements
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	
	// Considering the aspect ratio is fixed (CALIB_FIX_ASPECT_RATIO) set fx/fy
	Size testPatternSize(6, 9);
	int cornersPerBoard = 6 * 9;

	//CvCapture* capture = cvCreateCameraCapture(0);
	VideoCapture stream1(0);
	Mat cameraFrame;
	Mat cameraFrame_bw;

	stream1.read(cameraFrame);
	aspectRatio = (float)cameraFrame.size().width / (float)cameraFrame.size().height;
	cameraMatrix.at<double>(0, 0) = aspectRatio;
	cameraMatrix.at<double>(1, 1) = 1.0f;

	cvNamedWindow("Camera");
	cvNamedWindow("Pattern");

	if (!stream1.isOpened()) { //check if video device is initialised
		cout << "cannot open camera";
	}

	vector<Point2f> pointBuffer;

	vector<vector<Point3f>> objectPoints(1);
	vector<vector<Point2f>> imagePoints;

	calcBoardCornerPositions(testPatternSize, squareSize, objectPoints[0]);

	//Find boardCount number of patterns and record the coordinates.
	while (successes < boardCount) {
		stream1.read(cameraFrame);
		imshow("Camera", cameraFrame);

		//Try to find the chessboard pattern when "p" is pressed
		if (waitKey(1) == 'p') {
			bool patternFound = findChessboardCorners(cameraFrame, testPatternSize, pointBuffer, 
				CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS);

			if (patternFound) {
				cvtColor(cameraFrame, cameraFrame_bw, CV_BGR2GRAY);
				cornerSubPix(cameraFrame_bw, pointBuffer, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

				drawChessboardCorners(cameraFrame, testPatternSize, pointBuffer, patternFound);
				cv::imshow("Pattern", cameraFrame);
				
				imagePoints.push_back(pointBuffer);

				successes++;
			}
		}
	}
	
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms = calibrateCamera(objectPoints, imagePoints, cameraFrame.size(), cameraMatrix,
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
