#include <iostream>
#include <cmath>
#include <algorithm>

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

int main2() {
	Mat img = imread("images/img2.jpg", 0);

	imshow("Test window", img);
	waitKey(0);
	destroyWindow("Test window");

	Mat hist_img = getHistogram(img);
	imshow("���� ������ ������׷�", hist_img);
	waitKey(0);
	destroyWindow("Test window");

	for (int row = 0; row < img.rows; row++) { // 0 ~ 400
		for (int col = 0; col < img.cols; col++) {

			// ���� ������ ���� ��ο�
			img.at<uchar>(row, col) = max(img.at<uchar>(row, col) - 255 * (1 - row / 400.0), 0.0);
		}
	}

	imshow("���� ������ ���� ��ο�", img);
	waitKey(0);
	destroyWindow("Test window");

	hist_img = getHistogram(img);
	imshow("���� ������ ���� ��ο�", hist_img);
	waitKey(0);
	destroyWindow("Test window");

	Mat img2 = imread("images/img2.jpg", 0);

	for (int row = 0; row < img2.rows; row++) { // 0 ~ 400
		for (int col = 0; col < img2.cols; col++) {
			// �Ʒ��� ������ ���� ��ο�
			img2.at<uchar>(row, col) = max(img2.at<uchar>(row, col) - 255 * (row / 400.0), 0.0);
		}
	}

	imshow("�Ʒ��� ������ ���� ��ο�", img2);
	waitKey(0);
	destroyWindow("Test window");

	Mat hist_img2 = getHistogram(img2);
	imshow("�Ʒ��� ������ ���� ��ο�", hist_img2);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}