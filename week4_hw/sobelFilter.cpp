#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main3() {
	Mat src_img = imread("images/gear.jpg", 0);

	Mat img = mySobelFilter(src_img);
	imshow("Test window", img);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}