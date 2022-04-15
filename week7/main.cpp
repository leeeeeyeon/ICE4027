#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat MyKmeans(Mat src_img, int n_cluster) {
	vector<Scalar>clustersCenters;
	vector<vector<Point>>ptInClusters;
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;

	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	while (diffChange > threshold) {

		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }

		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);

		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);


	}
	Mat dst_img = applyFinalClusterTolmage(src_img, n_cluster, ptInClusters, clustersCenters);

	imshow("results", dst_img);
	//waitKey(0);


	return dst_img;
}

Mat CvKmeans(Mat src_img, int k) {

	//2차원 영상 -> 1차원 벡터
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x + src_img.rows) = (float)src_img.at<uchar>(y, x);
			}
		}
	}

	//opencv k-means 수행
	Mat labels;
	Mat centers;
	int attemps = 5;

	kmeans(samples, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		attemps, KMEANS_PP_CENTERS, centers);

	//1차원 벡터 => 2차원 영상
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] = (uchar)centers.at<float>(cluster_idx, z);
				}
			}
			else {
				dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	return dst_img;
}

void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters,
	vector<vector<Point>>& ptInClusters) {

	RNG random(cv::getTickCount());

	for (int k = 0; k < n_cluster; k++) {
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point>ptInClustersK;
		ptInClusters.push_back(ptInClustersK);
	}


}


double computeColorDistance(Scalar pixel, Scalar clusterPixel) {

	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));

	return distance;

}
