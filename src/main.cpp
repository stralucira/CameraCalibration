#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;
using namespace std;

void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults);
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs);
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients);

// Real world chessboard square size in meters
const float calibrationSquareDimension = 0.01905f;
// Considering the aspect ratio is fixed (CALIB_FIX_ASPECT_RATIO) set fx/fy
const Size chessboardDimensions = Size(6, 9);

int boardCount = 4; // Number of boards to be found before calibration.

int main()
{
	/*const Scalar red = Scalar(255, 0, 0);
	const Scalar green = Scalar(0, 255, 0);
	const Scalar blue = Scalar(0, 0, 255);

	float aspectRatio = 16.0f / 9.0f;
	float squareSize = 1.0f;
	
	int successes = 0;*/

	// Create a 3x3 identity matrix
	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

	// Distortion coefficients of 8 elements
	Mat distCoeffs;

	// Manually save a good calibrated image
	vector<Mat> savedImages;

	// Points that are found
	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	// CvCapture* capture = cvCreateCameraCapture(0);
	VideoCapture stream1(0);
	Mat cameraFrame;
	///Mat cameraFrame_bw;
	Mat drawToFrame;

	if (!stream1.isOpened()) // Check if video device is initialised
	{
		cout << "cannot open camera";
	}

	int framesPerSecond = 20;

	namedWindow("Camera", CV_WINDOW_AUTOSIZE);
	///namedWindow("Pattern", CV_WINDOW_AUTOSIZE);

	// Try to find the chessboard pattern from the camera
	while (true)
	{
		if (!stream1.read(cameraFrame)) break;
		//stream1.read(cameraFrame);
		//aspectRatio = (float)cameraFrame.size().width / (float)cameraFrame.size().height;
		//cameraMatrix.at<double>(0, 0) = aspectRatio;
		//cameraMatrix.at<double>(1, 1) = 1.0f;

		vector<Vec2f> pointBuffer;
		bool patternFound = false;
		patternFound = findChessboardCorners(cameraFrame, chessboardDimensions, pointBuffer,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);

		//bool patternFound = findChessboardCorners(cameraFrame, testPatternSize, pointBuffer,
		//	CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FILTER_QUADS);
		
		cameraFrame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions, pointBuffer, patternFound);
		
		// Drawing routine
		if (patternFound)
		{	
			putText(drawToFrame, "Pattern found", cvPoint(30, 30),
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			
			imshow("Camera", drawToFrame);

			//cvtColor(cameraFrame, cameraFrame_bw, CV_BGR2GRAY);
			//cornerSubPix(cameraFrame_bw, pointBuffer, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			//drawChessboardCorners(cameraFrame, testPatternSize, pointBuffer, patternFound);
			//putText(cameraFrame, "Patterns found: " + to_string(successes + 1), cvPoint(30, 30),
			//	FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			//arrowedLine(cameraFrame, Point(0.0, 0.0), Point(1, 0), red, 5, 8, 0, 1.0001);
			//imshow("Pattern", cameraFrame);
			//imagePoints.push_back(pointBuffer);
			//successes++;

			//cvtColor(cameraFrame, cameraFrame_bw, CV_BGR2GRAY);
			//cornerSubPix(cameraFrame_bw, pointBuffer1, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			//objectPoints.resize(pointBuffer1.size(), objectPoints[0]);
			//Mat rvec(3, 1, DataType<double>::type);
			//Mat tvec(3, 1, DataType<double>::type);
			// Find the rotation and translation vectors for this particular pose
			// --- EXCEPTION
			//solvePnP(objectPoints, pointBuffer1, cameraMatrix, distCoeffs, rvec, tvec, SOLVEPNP_ITERATIVE);
			// Draw the coordinate axes on the board
			//drawAxis(0, 1, 0, blue, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
			//drawAxis(1, 0, 0, blue, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
			//drawAxis(0, 0, 1, blue, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
			//drawnFrame = cameraFrame;
			//imshow("Pattern", drawnFrame);
		}
		else
		{
			imshow("Camera", cameraFrame);
		}
		
		// Input handling
		char character = waitKey(1000 / framesPerSecond);
		switch (character)
		{
		case ' ':
			// Save image if pattern is found
			if (patternFound)
			{
				Mat temp;
				cameraFrame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13:
			// Start camera calibration if there are over 15 valid images
			if (savedImages.size() > 15)
			{
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distCoeffs);
				saveCameraCalibration("CalibratedCamera", cameraMatrix, distCoeffs);
			}
			break;
		case 27:
			// Exit
			return 0;
			break;
		}
	}

	//double rms = calibrateCamera(objectPoints, imagePoints, cameraFrame.size(), cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS);
	//cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	return 0;
}

// Convert the 2D points obtained from image to 3D coordinates
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; ++i)
	{
		for (int j = 0; j < boardSize.width; ++j)
		{
			// Set z = 0 for every point, considering the chessboard is planar.
			corners.push_back(Point3f(j * squareSize, i * squareSize, 0.0f));
		}
	}
}

// Extract chessboard corners that have been detected from image
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	// Iterate vector of images
	for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
	{
		// Buffer to hold detected points
		vector<Point2f> pointBuffer;
		bool patternFound = findChessboardCorners(*iter, chessboardDimensions, pointBuffer,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE); // OpenCV flags
	
		if (patternFound)
		{
			// Push back all detected points from point buffer
			allFoundCorners.push_back(pointBuffer);
		}

		if (showResults)
		{
			// Draw the chessboard corners
			drawChessboardCorners(*iter, chessboardDimensions, pointBuffer, patternFound);
			imshow("Pattern", *iter);
			waitKey(0);
		}
	}
}

// Camera calibration from images with detected patterns
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs)
{
	vector<vector<Point2f>> imagePoints;
	vector<vector<Point3f>> objectPoints(1);
	getChessboardCorners(calibrationImages, imagePoints, false);

	calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	// Radial vectors and tangential vectors
	vector<Mat> rvecs, tvecs;
	// Distance coefficients of 8 elements
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	// The Magic of OpenCV!
	calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, rvecs, tvecs);
}

// Save camera calibration matrix into a file
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distCoeffs)
{
	ofstream outStream(name);
	if (outStream)
	{
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		rows = distCoeffs.rows;
		columns = distCoeffs.cols;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = distCoeffs.at<double>(r, c);
				outStream << value << endl;
			}
		}

		outStream.close();
		return true;
	}

	return false;
}

void drawAxis(float x, float y, float z, Scalar color, Mat rvec, Mat tvec, Mat &cameraMatrix, Mat &distCoeffs, Mat &image) {
	std::vector<cv::Point3f> points;
	std::vector<cv::Point2f> projectedPoints;

	//fills input array with 2 points
	points.push_back(cv::Point3f(0, 0, 0));
	points.push_back(cv::Point3f(x, y, z));

	//projects points using projectPoints method
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	//draws corresponding line
	arrowedLine(image, projectedPoints[0], projectedPoints[1], color);
}
