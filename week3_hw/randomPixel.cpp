#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// ���� ���� ���� �����ϴ� �Լ�
Mat redSpreadSalts(Mat img, int num) {
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
			img.at<Vec3b>(y, x)[0] = 0; // Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 0; // Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 255; // Red ä�� ����
		}
	}

	return img;
}

// �ʷ� ���� ���� �����ϴ� �Լ�
Mat greenSpreadSalts(Mat img, int num) {
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
			img.at<Vec3b>(y, x)[0] = 0; // Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 255; // Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0; // Red ä�� ����
		}
	}
	
	return img;
}

// �Ķ� ���� ���� �����ϴ� �Լ�
Mat blueSpreadSalts(Mat img, int num) {
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
			img.at<Vec3b>(y, x)[1] = 0; // Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0; // Red ä�� ����
		}
	}

	return img;
}

// ���� ���� ���� ī��Ʈ�ϴ� �Լ�
int countRedPixel(Mat img) {
	int redCount = 0;
	// �̹��� ��ü�� ���� ���� ���� �ش��ϴ� Vec3b(0, 0, 255)�� �ȼ��� ������
	// redCount��� ������ 1 ������Ų��.
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			if (img.at<Vec3b>(row, col) == Vec3b(0, 0, 255))
			{
				redCount++;
			}

		}
	}

	return redCount;
}

// �ʷ� ���� ���� ī��Ʈ�ϴ� �Լ�
int countGreenPixel(Mat img) {
	int greenCount = 0;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			if (img.at<Vec3b>(row, col) == Vec3b(0, 255, 0))
			{
				greenCount++;
			}

		}
	}

	return greenCount;
}

// �Ķ� ���� ���� ī��Ʈ�ϴ� �Լ�
int countBluePixel(Mat img) {
	int blueCount = 0;
	for (int row = 0; row < img.rows; row++) {
		for (int col = 0; col < img.cols; col++) {
			if (img.at<Vec3b>(row, col) == Vec3b(255, 0, 0))
			{
				blueCount++;
			}

		}
	}

	return blueCount;
}

int main1() {
	Mat src_img = imread("images/img1.jpg", 1);

	Mat salted_img = redSpreadSalts(src_img, 100);
	salted_img = greenSpreadSalts(src_img, 200);
	salted_img = blueSpreadSalts(src_img, 300);

	int redCount = countRedPixel(salted_img);
	int greenCount = countGreenPixel(salted_img);
	int blueCount = countBluePixel(salted_img);

	cout << "���� ���� ��: " << redCount << "��\n";
	cout << "�ʷ� ���� ��: " << greenCount << "��\n";
	cout << "�Ķ� ���� ��: " << blueCount << "��\n";

	imshow("Test window", salted_img);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}