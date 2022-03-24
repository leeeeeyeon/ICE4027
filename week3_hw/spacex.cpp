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
	cv::Mat father = cv::imread("father.jpg"); // �ΰ� �־��� ���� �б�
	cv::Mat logo = cv::imread("TJU_logo1.jpg"); // �ΰ� ���� �б�
	cv::Mat logo_gray = cv::imread("TJU_logo1.jpg", CV_LOAD_IMAGE_GRAYSCALE); // �ΰ� ���� �׷��� ������ �б�, ����ũ�� ����ϱ� ���ؼ�
	std::cout << "image size:"<< father.rows <<" x " << father.cols << ", logo size:" << logo.rows << " x " <<logo.cols << std::endl; // �ΰ� �־��� ���� �� �ΰ� ���� ������ ���
	cv::Mat imageROI(father, cv::Rect(father.cols - logo.cols, father.rows - logo.rows, logo.cols, logo.rows)); // ������ ������ �ϴܿ��� ���� ���ɿ��� (ROI) ���� 
	cv::Mat mask(120 - logo_gray); // '120 - �ΰ� �׷��� ����'�� ����ũ�� ���
	logo.copyTo(imageROI, mask); // ����ũ ���� 0�� �ƴ� ��ġ���� �ΰ� ROI�� ����(�ΰ��� �� ����� ���ֱ� ���� ���)
	cv::imshow("Image", father);  // �ΰ� �� ���� ����
	*/

	Mat dst = img3 - img4; // img3, img4�� ��ħ

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