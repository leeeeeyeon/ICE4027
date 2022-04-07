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

	// Spatial domain에서의 Sobel Filter
	Mat img = mySobelFilter(src_img);
	imshow("Spatial domain", img);
	waitKey(0);
	destroyWindow("Spatial domain");

    // Frequency domain에서의 Sobel Filter
    Mat input = doDft(src_img);
    input = getMagnitude(input);
    centralize(input);
    input = myNormalize(input);

    imshow("Magnitude", input);
    waitKey(0);
    destroyWindow("Magnitude");

    // dft 연산을 수행하기 위한 최적의 크기로 변환
    Mat padded_image;
    Size padded_size(
        getOptimalDFTSize(src_img.cols),
        getOptimalDFTSize(src_img.rows));

    // 크기를 optimize하며 생기는 빈 공간에 0 값을 추가
    copyMakeBorder(src_img, padded_image, 0, padded_size.height - src_img.rows, 0,padded_size.width - src_img.cols,
        BORDER_CONSTANT, //make the border constant (as opposed to a copy of the data
        Scalar::all(0)); //make the border black


    // DFT 연산을 위한 사전 작업 (1) Plane 제작
    Mat planes[] = {
        Mat_<float>(padded_image),
        Mat::zeros(padded_size,CV_32F)
    };

    // DFT 연산을 위한 사전 작업 (2) Plane 제작
    Mat complex_image;
    merge(planes, 2, complex_image);

    // complex_image에 대해 dft 연산 진행
    dft(complex_image, complex_image);

    // 수평 방향 필터
    Mat first_mask(padded_size, complex_image.type(), Scalar::all(0));

    Rect first_roi(first_mask.cols / 2 - 1, first_mask.rows / 2 - 1, 3, 3);

    Mat first_mask_center(first_mask, first_roi);
    first_mask_center.at<Vec2f>(0, 0)[0] = -1;
    first_mask_center.at<Vec2f>(1, 0)[0] = -2;
    first_mask_center.at<Vec2f>(2, 0)[0] = -1;
    first_mask_center.at<Vec2f>(0, 2)[0] = 1;
    first_mask_center.at<Vec2f>(1, 2)[0] = 2;
    first_mask_center.at<Vec2f>(2, 2)[0] = 1;

    // 수평 방향 필터에 fourier transform 적용
    dft(first_mask, first_mask);
    
    Mat mask1 = getMagnitude(first_mask);
    mask1 = myNormalize(mask1);
    imshow("Test window", mask1);
    waitKey(0);
    destroyWindow("Test window");

    // 영상과 필터의 convolution 수행
    Mat first_filtered_image = multiplyInFrequencyDomain(complex_image, first_mask);

    // 수직 방향 필터
    Mat second_mask(padded_size, complex_image.type(), Scalar::all(0));

    Rect second_roi(second_mask.cols / 2 - 1, second_mask.rows / 2 - 1, 3, 3);
    Mat second_mask_center(second_mask, second_roi);
    second_mask_center.at<Vec2f>(0, 0)[0] = 1;
    second_mask_center.at<Vec2f>(0, 1)[0] = 2;
    second_mask_center.at<Vec2f>(0, 2)[0] = 1;
    second_mask_center.at<Vec2f>(2, 0)[0] = -1;
    second_mask_center.at<Vec2f>(2, 1)[0] = -2;
    second_mask_center.at<Vec2f>(2, 2)[0] = -1;

    // 수직 방향 필터에 fourier transform 적용
    dft(second_mask, second_mask);
    
    Mat mask2 = getMagnitude(second_mask);
    mask2 = myNormalize(mask2);
    imshow("Test window", mask2);
    waitKey(0);
    destroyWindow("Test window");

    // 영상과 필터의 convolution 수행
    Mat second_filtered_image = multiplyInFrequencyDomain(complex_image, second_mask);
 
    // 필터링된 영상 출력
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

    // 필터링된 이미지를 spatial domain으로 변환
    dft(second_filtered_image, second_filtered_image, DFT_INVERSE | DFT_REAL_OUTPUT);
    dft(first_filtered_image, first_filtered_image, DFT_INVERSE | DFT_REAL_OUTPUT);

    // 좌표계 중앙 이동
   centralize(second_filtered_image);
   centralize(first_filtered_image);

    // Sobel filter 적용
    Mat result(first_filtered_image.rows, first_filtered_image.cols, CV_32FC1, Scalar::all(0));
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            float first = first_filtered_image.at<float>(y, x);
            float second = second_filtered_image.at<float>(y, x);
            result.at<float>(y, x) = sqrt(first * first + second * second);
        }
    }

    // 결과 영상을 frequency domain에서 출력
    Mat resultImg = doDft(result);
    resultImg = getMagnitude(resultImg);
    centralize(resultImg);
    resultImg = myNormalize(resultImg);
    imshow("Test window", resultImg);
    waitKey(0);
    destroyWindow("Test window");

    // 결과 영상을 spatial domain에서 출력
    normalize(result, result, 0, 1, CV_MINMAX);
    imshow("result", result);
    waitKey(0);
    destroyWindow("result");

	return 0;
}
