#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat getHistogram(Mat& src) {
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	// ������׷� ���
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	// ������׷� plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	// ����ȭ
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}

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

// �÷� ���� ���� Gaussian filter���� Ȱ���ϴ� ����ũ �迭 �Լ�
int myColorKernerConv3x3(uchar* arr, int kernel[][3], int col, int row, int k, int width, int height)
{
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1; i <= 1; i++)
		{
			if ((row + j) >= 0 && (row + j) < height && (col + i) >= 0 && (col + i) < width)
			{
				// RGB �÷� �����̹Ƿ� row, col�� ���� �׿� 3�� ���� ���� ����Ѵ�.
				int color = arr[(row + j) * 3 * width + (col + i) * 3 + k];
				sum += color * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	return sum / sumKernel;
}

// ����ũ�� ���� convolution
int myKernerConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	// Ư�� ȭ���� ��� �̿�ȭ�ҿ� ���� ����ϵ��� �ݺ��� ����
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			// ���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; } // ���� 1�� ����ȭ�ǵ��� ��
	else return sum;
}

Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	// 9x9 ������ Gaussian ����ũ �迭
	int kernel[9][9] = { 0,1,1,2,2,2,1,1,0,
						 1,2,4,5,5,5,4,2,1,
						 1,2,4,5,5,5,4,2,1,
						 2,5,3,-12,-24,-12,3,5,2,
						 2,5,0,-24,-40,-24,0,5,2,
						 2,5,3,-12,-24,-12,3,5,2,
						 1,2,4,5,5,5,4,2,1,
						 1,2,4,5,5,5,4,2,1,
						 0,1,1,2,2,2,1,1,0 };

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = myKernerConv9x9(srcData, kernel, x, y, width, height);
		}
	}

	return dstImg;
}

Mat myColorGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[3][3] = { 1,2,1,
						 2,4,2,
						 1,2,1 };

	Mat dstImg(srcImg.size(), CV_8UC3); // �÷� �����̱� ������ CV_8UC3�� ���
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = y * width * 3 + x * 3; // RGB �÷� �����̹Ƿ� data ���� 3�� ���ؼ� ����ؾ� ��
			int index2 = (y * 2) * (width * 2) * 3 + (x * 2) * 3;

			for (int k = 0; k < 3; k++)
			{
				dstData[index + k] = myColorKernerConv3x3(srcData, kernel, x, y, k, width, height);
			}
		}
	}

	return dstImg;
}

Mat saltAndPepper(Mat img, int num) {
	// num : ���� ���� ����
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; // img.cols�� �̹����� �� ������ ����
		int y = rand() % img.rows; // img.rows�� �̹����� ���� ������ ����
		/*
		* �������� ������ ���� ���� �� �����Ƿ�
		* �̹����� ũ�⸦ ����� �ʵ��� �����ϴ� ������ ����
		*/

		if (img.channels() == 1) {
			// img.channels() : �̹����� ä�� ���� ��ȯ
			if (n % 2 == 0) {
				img.at<char>(y, x) = 255; // ���� ä�� ����
			}
			else {
				img.at<char>(y, x) = 0;
			}
		}

		else {
			if (n % 2 == 0) {
				img.at<Vec3b>(y, x)[0] = 255; // Blue ä�� ����
				img.at<Vec3b>(y, x)[1] = 255; // Green ä�� ����
				img.at<Vec3b>(y, x)[2] = 255; // Red ä�� ����
			}
			else {
				img.at<Vec3b>(y, x)[0] = 0; // Blue ä�� ����
				img.at<Vec3b>(y, x)[1] = 0; // Green ä�� ����
				img.at<Vec3b>(y, x)[2] = 0; // Red ä�� ����
			}
		}
	}

	return img;
}

// ������ ����
Mat mySobelFilter(Mat srcImg) {
	int kernel45[3][3] = { -2,-1,0,
						   -1,0,1,
						   0,1,2 }; // 45�� �밢 ���� ����

	int kernel135[3][3] = { 0,1,2,
						  -1,0,1,
						  -2,-1,0 }; // 135�� �밢 ���� ����

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernerConv3x3(srcData, kernel45, x, y, width, height))
				+ abs(myKernerConv3x3(srcData, kernel135, x, y, width, height))) / 2;
			// �� ���� ����� ���� �� ���·� ��������� ����
		}
	}

	return dstImg;

}

Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3); // ����, ���ΰ� �Է� ������ ������ ������ ���� ����
	// �÷� �����̹Ƿ� CV_8UC1 ��� CV_8UC3�� ���
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// �÷� ���� �°� data�� ������ �� 3�� ���� ���� ���
			int index = y * width * 3 + x * 3;
			int index2 = (y * 2) * (width * 2) * 3 + (x * 2) * 3;

			dstData[index + 0] = srcData[index2 + 0];
			dstData[index + 1] = srcData[index2 + 1];
			dstData[index + 2] = srcData[index2 + 2];
		}
	}

	return dstImg;
}

vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	Vec.push_back(srcImg);

	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg); // down sampling
		srcImg = myColorGaussianFilter(srcImg); // gaussian filtering

		Vec.push_back(srcImg);
	}

	return Vec;
}

vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg; // �����ϱ� ���� ������ ���

			srcImg = mySampling(srcImg); // down sampling
			srcImg = myColorGaussianFilter(srcImg); // gaussian filtering

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size()); // �۾��� ������ ����� ������ ũ��� Ȯ��
			Vec.push_back(highImg - lowImg + 128);
			// �� ������ ���� �迭�� ����
			// 0~255 ������ ����� ���� �����ϱ� ���� 128�� ����
		}
		else {
			Vec.push_back(srcImg);
		}
	}

	return Vec;
}