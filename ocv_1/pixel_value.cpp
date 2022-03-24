#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat SpreadSalts(Mat img, int num) {
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
			img.at<Vec3b>(y, x)[1] = 255; // Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 255; // Rud 채널 접근
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