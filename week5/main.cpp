#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

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

Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);

	return phaImg;
}

Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat setComplex(Mat mag_img, Mat pha_img) {
	exp(mag_img, mag_img);
	mag_img -= Scalar::all(1);

	Mat planes[2];
	polarToCart(mag_img, pha_img, planes[0], planes[1]);

	Mat complex_img;
	merge(planes, 2, complex_img);

	return complex_img;
}

Mat doIdft(Mat complex_img) {
	Mat idftcvt;
	idft(complex_img, idftcvt);

	Mat planes[2];
	split(idftcvt, planes);

	Mat dst_img;
	magnitude(planes[0], planes[1], dst_img);
	normalize(dst_img, dst_img, 255, 0, NORM_MINMAX);
	dst_img.convertTo(dst_img, CV_8UC1);

	return dst_img;
}

Mat doLPF(Mat src_img) {
	// DFT
	Mat pad_img = padding(src_img);
	Mat complex_img = doDft(pad_img);
	Mat center_complex_img = centralize(complex_img);
	Mat mag_img = getMagnitude(center_complex_img);
	Mat pha_img = getPhase(center_complex_img);

	// LPF
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(mag_img, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(mag_img, mag_img, 0, 1, NORM_MINMAX);

	Mat mask_img = Mat::zeros(mag_img.size(), CV_32F);
	circle(mask_img, Point(mask_img.cols / 2, mask_img.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat mag_img2;
	multiply(mag_img, mask_img, mag_img2);

	// IDFT
	normalize(mag_img2, mag_img2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complex_img2 = setComplex(mag_img2, pha_img);
	Mat dst_img = doIdft(complex_img2);

	return myNormalize(dst_img);
}

Mat doHPF(Mat srcImg) {

	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//HPF
	double minVal, maxVal;
	Point minLoc, maxLoc;

	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//imshow("HW1", magImg2);
	//waitKey(0);
	//destroyWindow("HW1");

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

int main() {
	Mat src_img = imread("images/gear.jpg", 0);

	// 실습 1. Magnitude 영상 취득
	// Mat dst_img = doDft(src_img);
	//dst_img = getMagnitude(dst_img);
	//dst_img = myNormalize(dst_img);
	
	// 실습 2. Phase 영상 취득
	// Mat dst_img = doDft(src_img);
	//dst_img = getPhase(dst_img);
	//dst_img = myNormalize(dst_img);

	// 실습 3. 좌표계 중앙 이동
	// 실습 4. 2D IDFT

	// 실습 5. Low pass filtering
	// Mat dst_img = doLPF(src_img);

	// 실습 6. High pass filtering
	Mat dst_img = doHPF(src_img);


	imshow("Test window", dst_img);
	waitKey(0);
	destroyWindow("Test window");

}