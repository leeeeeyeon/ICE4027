#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat getHistogram(Mat& src);

Mat myCopy(Mat srcImg);

int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);

int myColorKernerConv3x3(uchar* arr, int kernel[][3], int col, int row, int k, int width, int height);

int myKernerConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height);

Mat myGaussianFilter(Mat srcImg);

Mat myColorGaussianFilter(Mat srcImg);

Mat saltAndPepper(Mat img, int num);

// À±°û¼± ÃßÃâ
Mat mySobelFilter(Mat srcImg);

Mat mySampling(Mat srcImg);

vector<Mat> myGaussianPyramid(Mat srcImg);

vector<Mat> myLaplacianPyramid(Mat srcImg);