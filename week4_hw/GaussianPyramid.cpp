#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main4() {
	Mat src_img = imread("images/gear.jpg", 1);

	vector<Mat> pyramid = myGaussianPyramid(src_img);

	imshow("1", pyramid[0]);
	imshow("2", pyramid[1]);
	imshow("3", pyramid[2]);
	imshow("4", pyramid[3]);

	waitKey(0);
	destroyAllWindows();

	return 0;
}