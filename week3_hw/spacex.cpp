#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat img3 = imread("images/img3.jpg", 1);
	Mat img4 = imread("images/img4.jpg", 1);

	resize(img4, img4, Size(img3.cols, img3.rows));

	/*
	cv::Mat father = cv::imread("father.jpg"); // 로고를 넣어줄 영상 읽기
	cv::Mat logo = cv::imread("TJU_logo1.jpg"); // 로고 영상 읽기
	cv::Mat logo_gray = cv::imread("TJU_logo1.jpg", CV_LOAD_IMAGE_GRAYSCALE); // 로고 영상 그레이 레벨로 읽기, 마스크로 사용하기 위해서
	std::cout << "image size:"<< father.rows <<" x " << father.cols << ", logo size:" << logo.rows << " x " <<logo.cols << std::endl; // 로고 넣어줄 영상 및 로고 영상 사이즈 출력
	cv::Mat imageROI(father, cv::Rect(father.cols - logo.cols, father.rows - logo.rows, logo.cols, logo.rows)); // 영상의 오른쪽 하단에서 영상 관심영역 (ROI) 정의 
	cv::Mat mask(120 - logo_gray); // '120 - 로고 그레이 영상'을 마스크로 사용
	logo.copyTo(imageROI, mask); // 마스크 값이 0이 아닌 위치에만 로고를 ROI에 삽입(로고의 흰 배경을 없애기 위한 방법)
	cv::imshow("Image", father);  // 로고가 들어간 영상 전시
	*/

	Mat dst = img3 - img4; // img3, img4를 합침

	Mat logo = imread("images/img5.jpg", 1);
	Mat gray_logo = imread("images/img5.jpg", 0);

	Mat imageROI(dst, Rect((dst.cols - logo.cols) / 2, (dst.rows - logo.rows) / 2 + 80, logo.cols, logo.rows));
	Mat mask(180 - gray_logo);
	logo.copyTo(imageROI, mask);

	imshow("Test window", dst);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}