#include <iostream>
#include <vector>
#include <string>

#include "helper.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main() {
	Mat src_img = imread("images/gear.jpg", 1);
	Mat dst_img;

	vector<Mat> pyramid = myLaplacianPyramid(src_img);

	imshow("1", pyramid[0]);
	imshow("2", pyramid[1]);
	imshow("3", pyramid[2]);
	// imshow("4", pyramid[3]);

	waitKey(0);
	destroyAllWindows();

	reverse(pyramid.begin(), pyramid.end()); // 작은 영상부터 처리하기 위해 vector의 순서를 반대로 함

	for (int i = 0; i < pyramid.size(); i++) { // Vector의 크기만큼 반복
		if (i == 0) { // 가장 작은 영상은 차 영상이 아니므로 바로 출력
			dst_img = pyramid[i];

			imshow("Test window1", dst_img);
			waitKey(0);
			destroyWindow("Test window1");
		}
		else {
			resize(dst_img, dst_img, pyramid[i].size()); // 작은 영상을 확대
			dst_img = dst_img + pyramid[i] - 128; // 차 영상을 다시 더해 큰 영상으로 복원
			// 앞서 더했던 128을 다시 빼준다.

			imshow("Test window2", dst_img); 
			waitKey(0);
			destroyWindow("Test window2");
		}
	}
	waitKey(0);
	destroyAllWindows();

	return 0;
}