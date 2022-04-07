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
	// dft ������ �����ϱ� ���� ������ ũ��� ��ȯ
	int m = getOptimalDFTSize(srcImg.rows);
	int n = getOptimalDFTSize(srcImg.cols); // on the border add zero values

	// ũ�⸦ optimize�ϸ� ����� �� ������ 0 ���� �߰�
	copyMakeBorder(srcImg, padded, 0, m - srcImg.rows, 0, n - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {
		Mat_<float>(padded),
		Mat::zeros(padded.size(), CV_32F)
	};

	Mat complexI;

	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]); // magnitude ���� ���

	Mat magI = planes[0];

	// ������ ���� ������ log scale�� ��ȯ
	magI += Scalar::all(1);
	log(magI, magI);

	centralize(magI); // ��ǥ�� �߾� �̵�

	normalize(magI, magI, 0, 1, CV_MINMAX); // ����ȭ

	// �Է� ������ â�� ���
	imshow("Input Image", srcImg);
	waitKey(0);
	destroyWindow("Input Image");

	// �Է� ������ magnitude ������ â�� ���
	imshow("magnitude", magI);
	waitKey(0);
	destroyWindow("magnitude");


	Mat bandpass = complexI.clone(); // complexI ������ complexBandPass �������� ����

	Mat bandPassFilterMask = doBPF(bandpass, 30, 90); // band pass filter ����

	Mat bandPassKernel = getFilterKernel(bandPassFilterMask);

	Mat bandPassDFTproduct = multiplyDFT(bandpass, bandPassKernel); // ����� kernel�� conlvolution�� ����

	Mat bandPassIDFT = doIDFT(bandPassDFTproduct);

	imshow("Band Pass Filtered", bandPassIDFT);
	waitKey(0);
	destroyWindow("Band Pass Filtered");

	return 0;
}