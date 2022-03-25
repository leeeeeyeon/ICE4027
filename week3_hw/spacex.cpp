#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat img3 = imread("images/img3.jpg", 1);
	Mat img4 = imread("images/img4.jpg", 1);

	resize(img4, img4, Size(img3.cols, img3.rows));

	Mat dst = img3 - img4; // img3, img4를 합침

	Mat logo = imread("images/img5.jpg", 1);
	Mat gray_logo = imread("images/img5.jpg", 0);

	// 서로 다른 크기를 가지는 영상을 합칠 때 사용
	Mat imageROI(dst, Rect((dst.cols - logo.cols) / 2, (dst.rows - logo.rows) / 2 + 80, logo.cols, logo.rows));
	Mat mask(180 - gray_logo); // 흰색 부분 제거
	logo.copyTo(imageROI, mask);

	imshow("Test window", dst);
	waitKey(0);
	destroyWindow("Test window");

	return 0;
}