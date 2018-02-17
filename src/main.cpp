#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#define RED		Scalar(255, 0, 0)
#define GREEN	Scalar(0, 255, 0)
#define BLUE	Scalar(0, 0, 255)
#define WHITE	Scalar(255, 255, 255)
#define ORIGIN	Point3f(0.f)

using namespace cv;
using namespace std;

void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults);
void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs);
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients);
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distCoeffs);
void drawAxis(float x, float y, float z, Scalar color, Mat rvec, Mat tvec, Mat &cameraMatrix, Mat &distCoeffs, Mat &image);
void drawCube(float length, int thickness, Scalar color, Mat rvec, Mat tvec, Mat& cameraMatrix, Mat& distCoeffs, Mat& image);

// Real world chessboard square length in meters
const float calibrationSquareDimension = 0.025f; // paper chessboard length size
//const float calibrationSquareDimension = 0.0059f; // iPod chessboard length size

// Considering the aspect ratio is fixed (CALIB_FIX_ASPECT_RATIO) set fx/fy
const Size chessboardDimensions = Size(6, 9);

bool cameraUndistorted = true;	// Toggle camera distortion fix
bool cameraCalibrated = false;
int framesPerSecond = 20;
int boardCount = 4; // Number of boards to be found before calibration.

int main()
{
	// Create a 3x3 identity matrix
	Mat cameraMatrix = Mat(3, 3, CV_64F);

	// Distortion coefficients of 8 elements
	Mat distCoeffs;

	// Manually save a good calibrated image
	vector<Mat> savedImages;

	// Start video capture and initialize image containers
	VideoCapture stream1(0);
	Mat cameraFrame;
	Mat cameraFrame_bw;
	Mat cameraFrameUndistorted;
	Mat drawToFrame;

	if (!stream1.isOpened()) // Check if video device is initialised
	{
		cout << "cannot open camera";
	}

	// Create windows
	namedWindow("Camera", CV_WINDOW_AUTOSIZE);
	moveWindow("Camera", 50, 50);

	// Try to find the chessboard pattern from the camera
	while (true)
	{
		if (!stream1.read(cameraFrame)) break;

		vector<Point2f> pointBuffer;
		bool patternFound = false;
		patternFound = findChessboardCorners(cameraFrame, chessboardDimensions, pointBuffer,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		
		cameraFrame.copyTo(drawToFrame);

		// Draw chessboard corners
		if (!cameraCalibrated)
		{
			drawChessboardCorners(drawToFrame, chessboardDimensions, pointBuffer, patternFound);
		}
		
		// Drawing routine
		if (patternFound) // If chessboard is detected
		{	
			// Canny edge detector, improves accuracy
			cvtColor(cameraFrame, cameraFrame_bw, CV_BGR2GRAY);
			cornerSubPix(cameraFrame_bw, pointBuffer, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

			if (cameraCalibrated)
			{
				vector<Point3f> objectPoints;
				calcBoardCornerPositions(chessboardDimensions, calibrationSquareDimension, objectPoints);

				// Finds an object pose from 3D-2D point correspondences, calculating the rotation and translation vector
				Mat rvec, tvec;
				solvePnP(objectPoints, pointBuffer, cameraMatrix, distCoeffs, rvec, tvec);

				// Draw the coordinate axes on the board
				drawAxis(0.1f, 0.0f, 0.0f, RED, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				drawAxis(0.0f, 0.1f, 0.0f, GREEN, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				drawAxis(0.0f, 0.0f, 0.1f, BLUE, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);

				// Draw a cube from the origin
				drawCube(0.05f, 2, WHITE, rvec, tvec, cameraMatrix, distCoeffs, cameraFrame);
				
				// Remove camera distortion
				if (cameraUndistorted)
				{
					undistort(cameraFrame, cameraFrameUndistorted, cameraMatrix, distCoeffs);
					drawToFrame = cameraFrameUndistorted;
				}
				else
				{
					drawToFrame = cameraFrame;
				}
			}
			
			if (!cameraCalibrated)
			{
				putText(drawToFrame, "Pattern found. Press Space to save. " + to_string(savedImages.size()) + " /" + to_string(boardCount) + ".", cvPoint(30, 30),
					FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(200, 200, 250), 1, CV_AA);
			}

			imshow("Camera", drawToFrame);
		}
		else // Let through unprocessed camera stream
		{
			if (cameraCalibrated && cameraUndistorted)
			{
				undistort(cameraFrame, cameraFrameUndistorted, cameraMatrix, distCoeffs);
				imshow("Camera", cameraFrameUndistorted);
			}
			else
			{
				imshow("Camera", cameraFrame);
			}
		}
		
		// Input handling
		char character = waitKey(1000 / framesPerSecond);
		switch (character)
		{
		case 32: ///Space
			// Save image if pattern is found
			if (patternFound)
			{
				Mat temp;
				cameraFrame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
		case 13: ///Enter
			// Start camera calibration if there are over boardCount valid images
			if (savedImages.size() >= boardCount)
			{
				cameraCalibration(savedImages, chessboardDimensions, calibrationSquareDimension, cameraMatrix, distCoeffs);
				saveCameraCalibration("CalibratedCamera", cameraMatrix, distCoeffs);
				cameraCalibrated = true;
			}
			break;
		case 'l':
			// Load camera calibration data from file
			loadCameraCalibration("CalibratedCamera", cameraMatrix, distCoeffs);
			cameraCalibrated = true;
			break;
		case 'r':
			// Reset camera calibration
			cameraMatrix = Mat(3, 3, CV_64F);
			distCoeffs = Mat::zeros(8, 1, CV_64F);
			cameraCalibrated = false;
			break;
		case 'd':
			// Toggle distortion
			cameraUndistorted = !cameraUndistorted;
			break;
		case 27: ///Escape
			// Exit
			return 0;
			break;
		}
	}

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

	// Rotation vectors and translation vectors
	vector<Mat> rvecs, tvecs;
	// Distortion coefficients of 8 elements
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	// The Magic of OpenCV!
	calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, rvecs, tvecs);
	//double rms = calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS);
	//cout << "Re-projection error reported by calibrateCamera: " << rms << endl;
}

// Save camera calibration matrix into a file
bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distCoeffs)
{
	ofstream outStream(name);
	if (outStream)
	{
		// Camera matrix
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;

		outStream << rows << endl;
		outStream << columns << endl;

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = cameraMatrix.at<double>(r, c);
				outStream << value << endl;
			}
		}

		// Distortion coefficients
		rows = distCoeffs.rows;
		columns = distCoeffs.cols;

		outStream << rows << endl;
		outStream << columns << endl;

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

// Load camera calibration matrix from a file
bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distCoeffs)
{
	ifstream inStream(name);
	if (inStream)
	{
		uint16_t rows;
		uint16_t columns;

		// Camera matrix
		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = 0.0f;
				inStream >> value;
				cameraMatrix.at<double>(r, c) = value;
				cout << cameraMatrix.at<double>(r, c) << "\n";
			}
		}

		// Distortion coefficients
		inStream >> rows;
		inStream >> columns;

		distCoeffs = Mat::zeros(rows, columns, CV_64F);

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < columns; c++)
			{
				double value = 0.0f;
				inStream >> value;
				distCoeffs.at<double>(r, c) = value;
				cout << distCoeffs.at<double>(r, c) << "\n";
			}
		}

		inStream.close();
		return true;
	}
	
	return false;
}

void drawAxis(float x, float y, float z, Scalar color, Mat rvec, Mat tvec, Mat& cameraMatrix, Mat& distCoeffs, Mat& image)
{
	vector<Point3f> points;
	vector<Point2f> projectedPoints;

	// Fills input array with 2 points
	points.push_back(ORIGIN);
	points.push_back(Point3f(x, y, -z));

	// Projects points using projectPoints method
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	// Draws corresponding line
	arrowedLine(image, projectedPoints[0], projectedPoints[1], color);
}

void drawCube(float length, int thickness, Scalar color, Mat rvec, Mat tvec, Mat& cameraMatrix, Mat& distCoeffs, Mat& image)
{
	vector<Point3f> points;
	vector<Point2f> projectedPoints;

	// Declare cube points from the origin
	points.push_back(Point3f(0.f,    0.f,    0.f)); // Point 0
	points.push_back(Point3f(length, 0.f,    0.f)); // Point 1
	points.push_back(Point3f(length, length, 0.f)); // Point 2
	points.push_back(Point3f(0.f,    length, 0.f)); // Point 3

	points.push_back(Point3f(0.f,    0.f,    -length)); // Point 4
	points.push_back(Point3f(length, 0.f,    -length)); // Point 5
	points.push_back(Point3f(length, length, -length)); // Point 6
	points.push_back(Point3f(0.f,    length, -length)); // Point 7

	// Projects points using projectPoints method
	projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

	// Create lines from cube points
	line(image, projectedPoints[0], projectedPoints[1], color, thickness);
	line(image, projectedPoints[1], projectedPoints[2], color, thickness);
	line(image, projectedPoints[2], projectedPoints[3], color, thickness);
	line(image, projectedPoints[3], projectedPoints[0], color, thickness);

	line(image, projectedPoints[4], projectedPoints[5], color, thickness);
	line(image, projectedPoints[5], projectedPoints[6], color, thickness);
	line(image, projectedPoints[6], projectedPoints[7], color, thickness);
	line(image, projectedPoints[7], projectedPoints[4], color, thickness);

	line(image, projectedPoints[0], projectedPoints[4], color, thickness);
	line(image, projectedPoints[1], projectedPoints[5], color, thickness);
	line(image, projectedPoints[2], projectedPoints[6], color, thickness);
	line(image, projectedPoints[3], projectedPoints[7], color, thickness);
}