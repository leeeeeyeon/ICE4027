#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat src_img = imread("images/landing.jpg", 1);

	imshow("Test window", src_img);
	waitKey(0);
	destroyWindow("Test window");
}