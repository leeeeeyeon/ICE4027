#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

// °øÅë
void centralize(Mat magI); // ÁÂÇ¥°è Áß¾Ó ÀÌµ¿
Mat padding(Mat img);
Mat doDft(Mat src_img);
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);

// band pass filter
Mat doLPF(Mat complexI, int radius); // Low-pass filter
Mat doBPF(Mat complexI, int radius, int radius2); // Band-pass filter
Mat multiplyDFT(Mat complexI, Mat kernel); // convolution
Mat doIDFT(Mat complexI);
Mat getFilterKernel(Mat FilterMask);

// sobel filter
int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);
Mat mySobelFilter(Mat srcImg);

void removeOutliers(const cv::Mat magnitude, cv::Mat* real, cv::Mat* imaginary);
void rearrangeQuadrants(cv::Mat* magnitude);
Mat multiplyInFrequencyDomain(const cv::Mat& image, const cv::Mat& mask);

// flickering
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void synthesizeFilterH(Mat& inputOutput_H, Point center, int radius);
void calcPSD(const Mat& inputImg, Mat& outputImg);