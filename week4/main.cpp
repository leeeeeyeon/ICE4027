#include <iostream>
#include <vector>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat myCopy(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;


	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = srcData[y * width + x];
		}
	}

	return dstImg;
}

int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }
	else return sum;
}

// 약간 블러 처리
Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[3][3] = { 1,2,1,
						 2,4,2,
						 1,2,1 };

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = myKernerConv3x3(srcData, kernel, x, y, width, height);
		}
	}

	return dstImg;
}

// 윤곽선 추출
Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1,0,1,
						   -2,0,2,
						   -1,0,1 };
	int kernelY[3][3] = { -1,-2,-1,
						  0,0,0,
						  1,2,1 };

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernerConv3x3(srcData, kernelX, x, y, width, height))
				+ abs(myKernerConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		}
	}

	return dstImg;

}

Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = srcData[(y * 2) * (width * 2) + (x * 2)];
		}
	}

	return dstImg;
}

vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	Vec.push_back(srcImg);

	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg);
		srcImg = myGaussianFilter(srcImg);

		Vec.push_back(srcImg);
	}

	return Vec;
}

vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg;

			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter(srcImg);

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			Vec.push_back(highImg - lowImg + 128); // 0~255 범위를 벗어나는 것을 방지하기 위해 128을 더함
		}
		else {
			Vec.push_back(srcImg);
		}
	}

	return Vec;
}

int main() {
	Mat src_img = imread("images/gear.jpg", 0);
	Mat dst_img;

	// Mat img = myGaussianFilter(src_img);
	// Mat img = mySobelFilter(src_img);
	// Mat img = mySampling(src_img);
	// vector<Mat> vectorResult = myGaussianPyramid(src_img);
	// imshow("Test window", img);
	// waitKey(0);
	// destroyWindow("Test window");

	vector<Mat> VecLap = myLaplacianPyramid(src_img);

	imshow("1", VecLap[0]);
	imshow("2", VecLap[1]);
	imshow("3", VecLap[2]);
	imshow("4", VecLap[3]);
	waitKey(0);
	destroyAllWindows();

	/*
	reverse(VecLap.begin(), VecLap.end());

	for (int i = 0; i < VecLap.size(); i++) {
		if (i == 0) {
			dst_img = VecLap[i];
		}
		else {
			resize(dst_img, dst_img, VecLap[i].size());
			dst_img = dst_img + VecLap[i] - 128;
		}

		string fname = "ex5_lap_pyr" + to_string(i) + ".png";
		imwrite(fname, dst_img);
		imshow("ex5", dst_img);
		waitKey(0);
		destroyWindow("ex5");
	}
	*/

}