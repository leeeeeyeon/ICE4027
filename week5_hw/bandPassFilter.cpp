#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "helper.h"
using namespace std;
using namespace cv;

int main() {
	Mat srcImg = imread("images/img1.jpg", 0);

	Mat padded;
	// dft 연산을 수행하기 위한 최적의 크기로 변환
	int m = getOptimalDFTSize(srcImg.rows);
	int n = getOptimalDFTSize(srcImg.cols); // on the border add zero values

	// 크기를 optimize하며 생기는 빈 공간에 0 값을 추가
	copyMakeBorder(srcImg, padded, 0, m - srcImg.rows, 0, n - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {
		Mat_<float>(padded),
		Mat::zeros(padded.size(), CV_32F)
	};

	Mat complexI;

	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]); // magnitude 영상 취득

	Mat magI = planes[0];

	// 영상을 보기 쉽도록 log scale로 변환
	magI += Scalar::all(1);
	log(magI, magI);

	centralize(magI); // 좌표계 중앙 이동

	normalize(magI, magI, 0, 1, CV_MINMAX); // 정규화

	// 입력 영상을 창에 출력
	imshow("Input Image", srcImg);
	waitKey(0);
	destroyWindow("Input Image");

	// 입력 영상의 magnitude 영상을 창에 출력
	imshow("magnitude", magI);
	waitKey(0);
	destroyWindow("magnitude");


	Mat bandpass = complexI.clone(); // complexI 영상을 complexBandPass 영상으로 복사

	Mat bandPassFilterMask = doBPF(bandpass, 30, 90); // band pass filter 제작

	Mat bandPassKernel = getFilterKernel(bandPassFilterMask);

	Mat bandPassDFTproduct = multiplyDFT(bandpass, bandPassKernel); // 영상과 kernel의 conlvolution을 수행

	Mat bandPassIDFT = doIDFT(bandPassDFTproduct);

	imshow("Band Pass Filtered", bandPassIDFT);
	waitKey(0);
	destroyWindow("Band Pass Filtered");

	return 0;
}