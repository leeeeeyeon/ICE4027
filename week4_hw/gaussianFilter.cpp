#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat src_img = imread("images/gear.jpg", 0);

	Mat img = myGaussianFilter(src_img);
	imshow("Test window", img);
	waitKey(0);
	destroyWindow("Test window");

	Mat hist_img = getHistogram(src_img);
	imshow("원본 사진", hist_img);
	waitKey(0);
	destroyWindow("Test window");

	hist_img = getHistogram(img);
	imshow("9*9 Gaussian filter를 적용했을 때", hist_img);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}