#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "helper.h"

// 좌표계 중앙 이동 함수
void centralize(Mat magI) {
	// row나 column의 픽셀 수가 홀수이면 사진을 자른다.
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy)); // 좌측 상단
	Mat q1(magI, Rect(cx, 0, cx, cy)); // 우측 상단
	Mat q2(magI, Rect(0, cy, cx, cy)); // 좌측 하단
	Mat q3(magI, Rect(cx, cy, cx, cy)); // 우측 하단

	Mat tmp;

	// 좌표계 중앙 이동
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

// 영상과 필터의 convolution을 수행하는 함수
Mat multiplyDFT(Mat complexI, Mat kernel) {
	centralize(complexI); // 좌표계 중앙 이동

	mulSpectrums(complexI, kernel, complexI, DFT_ROWS); // complexI(영상)와 kernel(필터)의 convolution을 계산

	return complexI;
}

Mat doIDFT(Mat complexI)
{
	centralize(complexI); // 좌표계 중앙이동
	Mat idftcvt;

	idft(complexI, idftcvt); // IDFT를 이용한 원본 영상 취득

	// complexI와 크기가 동일한 영상을 만듦
	Mat plane[] = {
		Mat::zeros(complexI.size(), CV_32F),
		Mat::zeros(complexI.size(), CV_32F)
	};

	split(idftcvt, plane);

	magnitude(plane[0], plane[1], idftcvt);
	normalize(idftcvt, idftcvt, 0, 1, NORM_MINMAX);


	return idftcvt;
}

Mat getFilterKernel(Mat img)
{
	Mat plane[] = {
		Mat_<float>(img),
		Mat::zeros(img.size(), CV_32F)
	};
	Mat kernel;
	merge(plane, 2, kernel); // 채널을 하나로 합침

	return kernel;
}

// Low pass filter
Mat doLPF(Mat complexI, int radius) {

	// complexI와 크기가 동일하고 0으로 초기화된 영상을 생성
	Mat maskImg = Mat::zeros(complexI.rows, complexI.cols, CV_32F);

	// Low pass Filter을 제작
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), radius, Scalar(1), -1, 8);

	return maskImg;
}

// Band pass filter
Mat doBPF(Mat complexI, int radius, int radius2)
{
	// 차 연산을 이용하여 Band pass filter 영상을 제작
	Mat bandPassFilterMask = doLPF(complexI, radius2) - doLPF(complexI, radius);

	imshow("Band Pass Filter Mask", bandPassFilterMask);
	return bandPassFilterMask;

}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = {
		Mat_<float>(inputImg.clone()), // 입력 영상의 복사본
		Mat::zeros(inputImg.size(), CV_32F) // 값이 0으로 채워진 영상
	};

	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = {
		Mat_<float>(H.clone()), // H를 복사한 영상
		Mat::zeros(H.size(), CV_32F) // H와 크기가 동일하고, 값이 0으로 채워진 영상
	};

	Mat complexH;
	merge(planesH, 2, complexH);

	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0); // complexI 영상과 complexH 영상을 convolution

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}

void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius)
{
	Point c2 = center, c3 = center, c4 = center;

	c2.y = inputOutput_H.rows - center.y;
	c3.x = inputOutput_H.cols - center.x;
	c4 = Point(c3.x, c2.y);

	circle(inputOutput_H, center, radius, 0, -1, 8);
	circle(inputOutput_H, c2, radius, 0, -1, 8);
	circle(inputOutput_H, c3, radius, 0, -1, 8);
	circle(inputOutput_H, c4, radius, 0, -1, 8);
}

void calcPSD(const Mat& inputImg, Mat& outputImg)
{
	Mat planes[2] = {
		Mat_<float>(inputImg.clone()),
		Mat::zeros(inputImg.size(), CV_32F)
	};
	Mat complexI;

	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes); // 실수부, 허수부 분리

	planes[0].at<float>(0) = 0;
	planes[1].at<float>(0) = 0;

	Mat imgPSD;
	magnitude(planes[0], planes[1], imgPSD); // magnitude 영상 취득
	pow(imgPSD, 2, imgPSD);
	outputImg = imgPSD;
}

// mySobelFilter 함수에서 사용
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
			// 두 에지 결과의 절댓값 합 형태로 최종결과를 도출
		}
	}

	return dstImg;
}

// 영상과 필터의 convolution을 수행, 차분을 이용
Mat multiplyInFrequencyDomain(const Mat& image, const Mat& mask) {
	Mat result(image.rows, image.cols, CV_32FC2, Scalar::all(0));

	//multiply each element as a complex number
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			result.at<Vec2f>(y, x)[0] = image.at<Vec2f>(y, x)[0] * mask.at<Vec2f>(y, x)[0] - image.at<Vec2f>(y, x)[1] * mask.at<Vec2f>(y, x)[1];
			result.at<Vec2f>(y, x)[1] = image.at<Vec2f>(y, x)[0] * mask.at<Vec2f>(y, x)[1] + image.at<Vec2f>(y, x)[1] * mask.at<Vec2f>(y, x)[0];
		}
	}

	return result;
}

Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);

	Mat padded;
	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));

	return padded;
}

Mat doDft(Mat src_img) {
	Mat float_img;
	src_img.convertTo(float_img, CV_32F);

	Mat complex_img;
	dft(float_img, complex_img, DFT_COMPLEX_OUTPUT);

	return complex_img;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}