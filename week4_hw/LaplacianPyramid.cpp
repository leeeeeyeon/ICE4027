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

	reverse(pyramid.begin(), pyramid.end()); // ���� ������� ó���ϱ� ���� vector�� ������ �ݴ�� ��

	for (int i = 0; i < pyramid.size(); i++) { // Vector�� ũ�⸸ŭ �ݺ�
		if (i == 0) { // ���� ���� ������ �� ������ �ƴϹǷ� �ٷ� ���
			dst_img = pyramid[i];

			imshow("Test window1", dst_img);
			waitKey(0);
			destroyWindow("Test window1");
		}
		else {
			resize(dst_img, dst_img, pyramid[i].size()); // ���� ������ Ȯ��
			dst_img = dst_img + pyramid[i] - 128; // �� ������ �ٽ� ���� ū �������� ����
			// �ռ� ���ߴ� 128�� �ٽ� ���ش�.

			imshow("Test window2", dst_img); 
			waitKey(0);
			destroyWindow("Test window2");
		}
	}
	waitKey(0);
	destroyAllWindows();

	return 0;
}