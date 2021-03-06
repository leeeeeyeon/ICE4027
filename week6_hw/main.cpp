#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace chrono;

// 가우시안 수식을 코드로 구현
double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

// 가우시안 수식을 코드로 구현
double gaussian2D(float c, float r, double sigma) {
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

// 두 픽셀 사이의거리를 구하는 함수
float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

void myKernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn) {

	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	// 픽셀 인덱싱
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			sum = 0.f;

			// 커널 인덱싱
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum;
			else tmp = abs(tmp);

			if (tmp > 255.f) tmp = 255.f;

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

void myGaussian(const Mat& src_img, Mat& dst_img, Size size) {
	//커널 생성
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 1.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++) {
		for (int r = 0; r < kn.rows; r++) {
			kn_data[r * kn.cols + c] = (float)gaussian2D((float)(c - kn.cols / 2),
				(float)(r - kn.rows / 2), sigma);
		}
	}

	myKernelConv(src_img, dst_img, kn);
}

void bilateral(const Mat& src_img, Mat & dst_img, int c, int r, int diameter, double sig_r, double sig_s) {
	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;

	// 커널 인덱싱
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			// range sigma에 대한 가우시안 연산
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);

			// space sigma에 대한 가우시안 연산
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);

			// 둘을 곱하여 가중치(weight)을 제작
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; // 정규화
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {

	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows; // src_img의 열과 행
	int radius = diameter / 2;

	// 가장자리를 제외한 픽셀 인덱싱, bilateral() 함수 적용
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1);

}

void doBilateralEx() {
	cout << "---doBilateralEx()---" << endl;
	Mat src_img = imread("images/rock.png", 0);
	Mat dst_img;
	if (!src_img.data)printf("no image data\n");

	Mat result_img;

	// diameter가 클수록 더 많은 범위의 픽셀에 접근할 수 있음. diameter을 30으로 택하여 실험을 진행
	// Case 1. range sigma를 25.0으로 고정, space sigma를 점점 키움
	myBilateral(src_img, dst_img, 30, 25.0, 0.1);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 1", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 25.0, 1.8);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 2", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 25.0, 100.0);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 3", result_img);
	waitKey(0);

	// Case 2. range sigma를 100.0으로 고정, space sigma를 점점 키움
	myBilateral(src_img, dst_img, 30, 100.0, 0.1);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 1", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 100.0, 1.8);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 2", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 100.0, 100.0);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 3", result_img);
	waitKey(0);

	// Case 3. range sigma를 무한대(적당히 큰 수)로 고정, space sigma를 점점 키움
	myBilateral(src_img, dst_img, 30, 1000000000000.0, 0.1);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 1", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 1000000000000.0, 1.8);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 2", result_img);
	waitKey(0);

	myBilateral(src_img, dst_img, 30, 1000000000000.0, 100.0);
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx() 3", result_img);
	waitKey(0);
}

// Median Filter
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows; // src_img의 열과 행, 256 256
	int kwd = kn_size.width; int khg = kn_size.height; // kernel의 size
	int rad_w = kwd / 2; int rad_h = khg / 2; // kn_size의 절반, 중간값 indexing에 사용

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	int table_size = kwd * khg;
	float* table = new float[table_size](); // 커널 테이블 동적할당
	float tmp;
	
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			// 픽셀 값 sort
			// 중간값 indexing

			// 기준 픽셀를 중앙으로 하는 주변 5*5 픽셀들을 테이블에 저장
			for (int y = 0; y < 5; y++) {
				for (int x = 0; x < 5; x++) {
					table[x + y * 5] = src_data[c - 1 + x + (r - 1 + y) * 256];
				}
			}
			
			sort(table, table + 25); // 5*5 테이블의 원소들을 정렬
			tmp = table[13]; // 중앙값을 tmp 변수에 담음

			dst_data[c + r * 256] = tmp; // 현재 픽셀의 값을 tmp(중앙값)으로 변경
		}
	}

	delete[] table; // 동적할당 해제


}

void doMedianEx() {
	cout << "--- doMedianEx() --- \n" << endl;

	// 입력
	Mat src_img = imread("images/salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img;
	// Median 필터링 수행
	myMedian(src_img, dst_img, Size(5, 5));

	Mat result_img;
	hconcat(src_img, dst_img, result_img); // 원본 이미지와 필터링을 수행한 이미지를 함께 띄움

	imshow("doMedianEx()", result_img);
	waitKey(0);
	destroyWindow("doMedianEx()");

}

void doCannyEx() {
	cout << "--- doCannyEx() --- \n" << endl;
	Mat src_img = imread("images/rock.png", 0);

	if (!src_img.data) cout << "No image data\n";

	// Case 1
	Mat dst_img;
	system_clock::time_point start = system_clock::now();
	Canny(src_img, dst_img, 180, 240);
	system_clock::time_point end = system_clock::now();
	cout << "180, 240: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);

	// Case 2
	start = system_clock::now();
	Canny(src_img, dst_img, 180, 480);
	end = system_clock::now();
	cout << "180, 480: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);

	// Case 3
	start = system_clock::now();
	Canny(src_img, dst_img, 90, 120);
	end = system_clock::now();
	cout << "90, 120: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);

	// Case 4
	start = system_clock::now();
	Canny(src_img, dst_img, 90, 240);
	end = system_clock::now();
	cout << "90, 240: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);

	// Case 5
	start = system_clock::now();
	Canny(src_img, dst_img, 360, 480);
	end = system_clock::now();
	cout << "360, 480: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);

	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	waitKey(0);
}

int main() {
	// doMedianEx();
	// doCannyEx();
	doBilateralEx();
}
