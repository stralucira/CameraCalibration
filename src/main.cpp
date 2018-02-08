#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture stream1(0);

	if (!stream1.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}

	while (true) {
		Mat cameraFrame;
		stream1.read(cameraFrame);
		imshow("cam", cameraFrame);
		waitKey(30);
	}

	/*cv::Mat img = cv::imread("data/logo.png");
	cv::namedWindow("OpenCV");
	cv::imshow("OpenCV", img);
	cv::waitKey(0);*/
	return 0;
}