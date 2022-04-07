#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "helper.h"

using namespace cv;
using namespace std;

int main() {
	Mat srcImg = imread("images/img3.jpg", 0);

	srcImg.convertTo(srcImg, CV_32F); // frequency domain으로 영상 접근

	// 이미지의 픽셀이 홀수 개이면 짝수 개가 되도록 변환
	Rect img = Rect(0, 0, srcImg.cols & -2, srcImg.rows & -2);
	srcImg = srcImg(img);

	Mat imgPSD;
	calcPSD(srcImg, imgPSD);
	centralize(imgPSD);
	normalize(imgPSD, imgPSD, 0, 255, NORM_MINMAX);

	//H calculation
	Mat H = Mat(img.size(), CV_32F, Scalar(1)); // img와 크기가 같고, 1로 값들이 채워진 영상을 frequency domain으로 접근

	// 변수 r의 값을 조정하여 flickering 현상을 제거
	const int r = 15;

	// 필터를 제작
	synthesizeFilterH(H, Point(408, 420), r);
	synthesizeFilterH(H, Point(491, 360), r);
	synthesizeFilterH(H, Point(574, 299), r);

	Mat imgHPlusPSD = imgPSD + H * 50;
	normalize(imgHPlusPSD, imgHPlusPSD, 0, 255, NORM_MINMAX);
	imgHPlusPSD.convertTo(imgHPlusPSD, CV_8U);

	// 영상과 필터의 convolution을 수행
	Mat imgOut;
	centralize(H);
	filter2DFreq(srcImg, imgOut, H);

	// 영상 출력
	srcImg.convertTo(srcImg, CV_8U);
	normalize(srcImg, srcImg, 0, 255, NORM_MINMAX);

	imshow("PSD", imgPSD);
	waitKey(0);
	destroyWindow("PSD");

	imshow("imgHPlusPSD", imgHPlusPSD);
	waitKey(0);
	destroyWindow("imgHPlusPSD");

	centralize(H);
	normalize(H, H, 0, 255, NORM_MINMAX);

	imshow("Filter", H);
	waitKey(0);
	destroyWindow("Filter");

	imgOut.convertTo(imgOut, CV_8U); // spatial domain으로 변환
	normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);

	imshow("result", imgOut);
	waitKey(0);
	destroyWindow("result");

	return 0;
}