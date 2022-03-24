#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// 빨강 색의 점을 생성하는 함수
Mat redSpreadSalts(Mat img, int num) {
	// num : 점을 찍을 개수
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; // img.cols는 이미지의 폭 정보를 저장
		int y = rand() % img.rows; // img.rows는 이미지의 높이 정보를 저장
		/*
		* 나머지는 나누는 수를 넘을 수 없으므로
		* 이미지의 크기를 벗어나지 않도록 제한하는 역할을 해줌
		*/

		if (img.channels() == 1) {
			// img.channels() : 이미지의 채널 수르 반환
			img.at<char>(y, x) = 255; // 단일 채널 접근
		}

		else {
			img.at<Vec3b>(y, x)[0] = 0; // Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 0; // Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 255; // Red 채널 접근
		}
	}

	return img;
}

// 초록 색의 점을 생성하는 함수
Mat greenSpreadSalts(Mat img, int num) {
	// num : 점을 찍을 개수
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; // img.cols는 이미지의 폭 정보를 저장
		int y = rand() % img.rows; // img.rows는 이미지의 높이 정보를 저장
		/*
		* 나머지는 나누는 수를 넘을 수 없으므로
		* 이미지의 크기를 벗어나지 않도록 제한하는 역할을 해줌
		*/

		if (img.channels() == 1) {
			// img.channels() : 이미지의 채널 수르 반환
			img.at<char>(y, x) = 255; // 단일 채널 접근
		}

		else {
			img.at<Vec3b>(y, x)[0] = 0; // Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 255; // Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 0; // Red 채널 접근
		}
	}
	
	return img;
}

// 파랑 색의 점을 생성하는 함수
Mat blueSpreadSalts(Mat img, int num) {
	// num : 점을 찍을 개수
	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; // img.cols는 이미지의 폭 정보를 저장
		int y = rand() % img.rows; // img.rows는 이미지의 높이 정보를 저장
		/*
		* 나머지는 나누는 수를 넘을 수 없으므로
		* 이미지의 크기를 벗어나지 않도록 제한하는 역할을 해줌
		*/

		if (img.channels() == 1) {
			// img.channels() : 이미지의 채널 수르 반환
			img.at<char>(y, x) = 255; // 단일 채널 접근
		}

		else {
			img.at<Vec3b>(y, x)[0] = 255; // Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 0; // Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 0; // Red 채널 접근
		}
	}

	return img;
}

// 빨강 색의 점을 카운트하는 함수
int countRedPixel(Mat img) {
	int redCount = 0;
	// 이미지 전체를 돌며 빨간 점에 해당하는 Vec3b(0, 0, 255)인 픽셀이 있으면
	// redCount라는 변수를 1 증가시킨다.
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

// 초록 색의 점을 카운트하는 함수
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

// 파랑 색의 점을 카운트하는 함수
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

	cout << "빨강 색의 점: " << redCount << "개\n";
	cout << "초록 색의 점: " << greenCount << "개\n";
	cout << "파랑 색의 점: " << blueCount << "개\n";

	imshow("Test window", salted_img);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}