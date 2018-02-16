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
void drawAxis(float x, float y, float z, Scalar color, Mat rvec, Mat tvec, Mat &cameraMatrix, Mat &distCoeffs, Mat &image);

// Real world chessboard square size in meters
const float calibrationSquareDimension = 0.01905f;
// Considering the aspect ratio is fixed (CALIB_FIX_ASPECT_RATIO) set fx/fy
const Size chessboardDimensions = Size(6, 9);

int boardCount = 4; // Number of boards to be found before calibration.

int main()
{
	bool cameraCalibrated = false;

	const Scalar red = Scalar(255, 0, 0);
	const Scalar green = Scalar(0, 255, 0);
	const Scalar blue = Scalar(0, 0, 255);

	// Create a 3x3 identity matrix
	Mat cameraMatrix = Mat(3, 3, CV_64F);

	// Distortion coefficients of 8 elements
	Mat distCoeffs;

	// Manually save a good calibrated image
	vector<Mat> savedImages;

	// Points that are found
	vector<vector<Point2f>> markerCorners, rejectedCandidates;

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
	namedWindow("Drawing", CV_WINDOW_AUTOSIZE);

	// Try to find the chessboard pattern from the camera
	while (true)
	{
		if (!stream1.read(cameraFrame)) break;

		vector<Point2f> pointBuffer;
		bool patternFound = false;
		patternFound = findChessboardCorners(cameraFrame, chessboardDimensions, pointBuffer,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		
		cameraFrame.copyTo(drawToFrame);

		if (!cameraCalibrated)
		{
			drawChessboardCorners(drawToFrame, chessboardDimensions, pointBuffer, patternFound);
		}
		
		// Drawing routine
		if (patternFound)
		{	
			putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + " /4.", cvPoint(30, 30),
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

			if (cameraCalibrated)
			{
				//vector<vector<Point2f>> imagePoints;
				vector<Point3f> objectPoints;

				calcBoardCornerPositions(chessboardDimensions, calibrationSquareDimension, objectPoints);
				//objectPoints.resize(imagePoints.size(), objectPoints[0]);

				Mat rvec, tvec;
				solvePnP(objectPoints, pointBuffer, cameraMatrix, distCoeffs, rvec, tvec);

				// Draw the coordinate axes on the board
				drawAxis(0.1, 0, 0, red, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				drawAxis(0, 0.1, 0, green, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				drawAxis(0, 0, 0.1, blue, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				drawToFrame = cameraFrame;

				imshow("Camera", drawToFrame);
			}
			imshow("Drawing", drawToFrame);
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
			if (savedImages.size() > 3)
			{
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distCoeffs);
				saveCameraCalibration("CalibratedCamera", cameraMatrix, distCoeffs);
				cameraCalibrated = true;
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

	// Radial vectors and tangential vectors?
	// Rotation vectors and translation vectors?
	vector<Mat> rvecs, tvecs;
	// Distortion coefficients of 8 elements
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	// The Magic of OpenCV!
	calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	// SOLVEPNP WORKS LIKE THIS!
	//Mat rvec, tvec;
	//solvePnP(objectPoints[0], imagePoints[0], cameraMatrix, distCoeffs, rvec, tvec);
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

void drawAxis(float x, float y, float z, Scalar color, Mat rvec, Mat tvec, Mat &cameraMatrix, Mat &distCoeffs, Mat &image)
{
	vector<Point3f> points;
	vector<Point2f> projectedPoints;

	//fills input array with 2 points
	points.push_back(Point3f(0, 0, 0));
	points.push_back(Point3f(x, y, z));

	//projects points using projectPoints method
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	//draws corresponding line
	arrowedLine(image, projectedPoints[0], projectedPoints[1], color);
}
