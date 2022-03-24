#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat SpreadSalts(Mat img, int num) {
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
			img.at<char>(y, x) = 255; // ���� ä�� ����
		}
		else {
			img.at<Vec3b>(y, x)[0] = 255; // Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 255; // Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 255; // Rud ä�� ����
		}
	}

	return img;
}

int main() {
	Mat src_img = imread("images/landing.jpg", 0);

	Mat salted_img = SpreadSalts(src_img, 1000);

	imshow("Test window", salted_img);
	waitKey(0);
	destroyWindow("Test window");
}