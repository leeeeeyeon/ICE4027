#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main2() {
	Mat src_img = imread("images/gear.jpg", 0);

	Mat img = saltAndPepper(src_img, 500);
	//Mat img = myGaussianFilter(src_img);
	imshow("Salt and pepper noise", img);

	Mat gaussianImg = myGaussianFilter(img);
	imshow("Gaussian Filter", img);

	waitKey(0);
	destroyAllWindows();

	Mat hist_img = getHistogram(img);
	imshow("원본 사진", hist_img);

	hist_img = getHistogram(gaussianImg);
	imshow("9*9 Gaussian filter를 적용했을 때", hist_img);

	waitKey(0);
	destroyAllWindows();

	return 0;
}