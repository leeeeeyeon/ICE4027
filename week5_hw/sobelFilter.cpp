#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "helper.h"

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

using namespace cv;
using namespace std;

int main() {
	Mat src_img = imread("images/img2.jpg", 0);

	// Spatial domain������ Sobel Filter
	Mat img = mySobelFilter(src_img);
	imshow("Spatial domain", img);
	waitKey(0);
	destroyWindow("Spatial domain");

    // Frequency domain������ Sobel Filter
    Mat input = doDft(src_img);
    input = getMagnitude(input);
    centralize(input);
    input = myNormalize(input);

    imshow("Magnitude", input);
    waitKey(0);
    destroyWindow("Magnitude");

    // dft ������ �����ϱ� ���� ������ ũ��� ��ȯ
    Mat padded_image;
    Size padded_size(
        getOptimalDFTSize(src_img.cols),
        getOptimalDFTSize(src_img.rows));

    // ũ�⸦ optimize�ϸ� ����� �� ������ 0 ���� �߰�
    copyMakeBorder(src_img, padded_image, 0, padded_size.height - src_img.rows, 0,padded_size.width - src_img.cols,
        BORDER_CONSTANT, //make the border constant (as opposed to a copy of the data
        Scalar::all(0)); //make the border black


    // DFT ������ ���� ���� �۾� (1) Plane ����
    Mat planes[] = {
        Mat_<float>(padded_image),
        Mat::zeros(padded_size,CV_32F)
    };

    // DFT ������ ���� ���� �۾� (2) Plane ����
    Mat complex_image;
    merge(planes, 2, complex_image);

    // complex_image�� ���� dft ���� ����
    dft(complex_image, complex_image);

    // ���� ���� ����
    Mat first_mask(padded_size, complex_image.type(), Scalar::all(0));

    Rect first_roi(first_mask.cols / 2 - 1, first_mask.rows / 2 - 1, 3, 3);

    Mat first_mask_center(first_mask, first_roi);
    first_mask_center.at<Vec2f>(0, 0)[0] = -1;
    first_mask_center.at<Vec2f>(1, 0)[0] = -2;
    first_mask_center.at<Vec2f>(2, 0)[0] = -1;
    first_mask_center.at<Vec2f>(0, 2)[0] = 1;
    first_mask_center.at<Vec2f>(1, 2)[0] = 2;
    first_mask_center.at<Vec2f>(2, 2)[0] = 1;

    // ���� ���� ���Ϳ� fourier transform ����
    dft(first_mask, first_mask);
    
    Mat mask1 = getMagnitude(first_mask);
    mask1 = myNormalize(mask1);
    imshow("Test window", mask1);
    waitKey(0);
    destroyWindow("Test window");

    // ����� ������ convolution ����
    Mat first_filtered_image = multiplyInFrequencyDomain(complex_image, first_mask);

    // ���� ���� ����
    Mat second_mask(padded_size, complex_image.type(), Scalar::all(0));

    Rect second_roi(second_mask.cols / 2 - 1, second_mask.rows / 2 - 1, 3, 3);
    Mat second_mask_center(second_mask, second_roi);
    second_mask_center.at<Vec2f>(0, 0)[0] = 1;
    second_mask_center.at<Vec2f>(0, 1)[0] = 2;
    second_mask_center.at<Vec2f>(0, 2)[0] = 1;
    second_mask_center.at<Vec2f>(2, 0)[0] = -1;
    second_mask_center.at<Vec2f>(2, 1)[0] = -2;
    second_mask_center.at<Vec2f>(2, 2)[0] = -1;

    // ���� ���� ���Ϳ� fourier transform ����
    dft(second_mask, second_mask);
    
    Mat mask2 = getMagnitude(second_mask);
    mask2 = myNormalize(mask2);
    imshow("Test window", mask2);
    waitKey(0);
    destroyWindow("Test window");

    // ����� ������ convolution ����
    Mat second_filtered_image = multiplyInFrequencyDomain(complex_image, second_mask);
 
    // ���͸��� ���� ���
    Mat firstImg = getMagnitude(first_filtered_image);
    firstImg = myNormalize(firstImg);
    imshow("Test window", firstImg);
    waitKey(0);
    destroyWindow("Test window");

    Mat secondImg = getMagnitude(second_filtered_image);
    secondImg = myNormalize(secondImg);
    imshow("Test window", secondImg);
    waitKey(0);
    destroyWindow("Test window");

    // ���͸��� �̹����� spatial domain���� ��ȯ
    dft(second_filtered_image, second_filtered_image, DFT_INVERSE | DFT_REAL_OUTPUT);
    dft(first_filtered_image, first_filtered_image, DFT_INVERSE | DFT_REAL_OUTPUT);

    // ��ǥ�� �߾� �̵�
   centralize(second_filtered_image);
   centralize(first_filtered_image);

    // Sobel filter ����
    Mat result(first_filtered_image.rows, first_filtered_image.cols, CV_32FC1, Scalar::all(0));
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            float first = first_filtered_image.at<float>(y, x);
            float second = second_filtered_image.at<float>(y, x);
            result.at<float>(y, x) = sqrt(first * first + second * second);
        }
    }

    // ��� ������ frequency domain���� ���
    Mat resultImg = doDft(result);
    resultImg = getMagnitude(resultImg);
    centralize(resultImg);
    resultImg = myNormalize(resultImg);
    imshow("Test window", resultImg);
    waitKey(0);
    destroyWindow("Test window");

    // ��� ������ spatial domain���� ���
    normalize(result, result, 0, 1, CV_MINMAX);
    imshow("result", result);
    waitKey(0);
    destroyWindow("result");

	return 0;
}
