#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// 영상의 색상 표현을 BGR을 HSV로 바꾸는 함수
Mat myBgr2Hsv(Mat src_img) {
    double b, g, r, h = 0.0, s, v;
    
    Mat dst_img(src_img.size(), src_img.type());
    for (int y=0; y<src_img.rows; y++) {
        for (int x=0; x<src_img.cols; x++) {
            // BGR 취득, RGB가 아닌 BGR임을 주의
            b = (double)src_img.at<Vec3b>(y, x)[0];
            g = (double)src_img.at<Vec3b>(y, x)[1];
            r = (double)src_img.at<Vec3b>(y, x)[2];
            
            b /= 255.0;
            g /= 255.0;
            r /= 255.0;
            
            // 수식을 이용한 변환
            double cMax = max({b, g, r}); // max() 함수로 3개를 비교할 땐 {}
            double cMin = min({b, g, r});
            
            double delta = cMax - cMin;
            
            v = cMax;
            
            if (v == 0) s = 0;
            else s = (delta / cMax);
            
            if (delta == 0) h = 0;
            else if (cMax == r) h = 60 * (g - b) / (v - cMin);
            else if (cMax == g) h = 120 + 60 * (b - r) / (v - cMin);
            else if (cMax == b) h = 240 + 60 * (r - g) / (v - cMin);
            
            if (h < 0) h+= 360;
            
            v *= 255;
            s *= 255;
            h /= 2;
            
            // 오버플로우 방지
            h = h > 255.0 ? 255.0 : h < 0 ? 0 : h;
            s = s > 255.0 ? 255.0 : s < 0 ? 0 : s;
            v = v > 255.0 ? 255.0 : v < 0 ? 0 : v;
            
            // 변환된 색상 대입
            dst_img.at<Vec3b>(y, x)[0] = (uchar) h;
            dst_img.at<Vec3b>(y, x)[1] = (uchar) s;
            dst_img.at<Vec3b>(y, x)[2] = (uchar) v;
            
        }
    }
    
    return dst_img;
}

// 입력 영상의 픽셀 값이 색상 범위에 포함되면 흰색, 그렇지 않으면 검은색으로 채워진 마스크 영상을 반환하는 함수
// 색상 범위 값은 BGR을 기준으로 할당
Mat myInRange (Mat src_img, Scalar lowerb, Scalar upperb) {
    // lowerb <= src <= upperb : 255
    // etc : 0

    Mat dst_img(src_img.size(), src_img.type());
    
    for (int y=0; y<src_img.rows; y++) {
        for (int x=0; x<src_img.cols; x++) {
            double b, g, r;
            
            // BGR 취득
            b = (double)src_img.at<Vec3b>(y, x)[0];
            g = (double)src_img.at<Vec3b>(y, x)[1];
            r = (double)src_img.at<Vec3b>(y, x)[2];
            
            Scalar bgrScalar(b, g, r);
            bool range0 = (lowerb[0] <= bgrScalar[0]) && (bgrScalar[0] <= upperb[0]); // Blue에 대해 색상 범위에 포함되었는지
            bool range1 = (lowerb[1] <= bgrScalar[1]) && (bgrScalar[1] <= upperb[1]); // Green에 대해 색상 범위에 포함되었는지
            bool range2 = (lowerb[2] <= bgrScalar[2]) && (bgrScalar[2] <= upperb[2]); // Red에 대해 색상 범위에 포함되었는지
            
            // B, G, R에 대해 모두 색상 범위에 포함된 경우 흰 색으로 영상을 채움
            if (range0 && range1 && range2) {
                dst_img.at<Vec3b>(y, x)[0] = 255;
                dst_img.at<Vec3b>(y, x)[1] = 255;
                dst_img.at<Vec3b>(y, x)[2] = 255;
            }
            else { // 그렇지 않다면 검은 색으로 영상을 채움
                dst_img.at<Vec3b>(y, x)[0] = 0;
                dst_img.at<Vec3b>(y, x)[1] = 0;
                dst_img.at<Vec3b>(y, x)[2] = 0;
            }
        }
    }
    
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
    Mat labels; // 군집 판별 결과가 담길 1차원 벡터
    Mat centers; // 각 군집의 중앙값(대표값)
    int attemps = 5;

    kmeans(samples, k, labels,
           TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 10000, 0.0001),
           attemps, KMEANS_PP_CENTERS, centers);

    //1차원 벡터 => 2차원 영상
    Mat dst_img(src_img.size(), src_img.type());

    for (int y = 0; y < src_img.rows; y++) {
        for (int x = 0; x < src_img.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
                if (src_img.channels() == 3) {
                    for (int z = 0; z < src_img.channels(); z++) {
                        dst_img.at<Vec3b>(y, x)[z] = (uchar)centers.at<float>(cluster_idx, z);
                        //군집판별 결과에 따라 각 군집의 중앙값으로 결과 생성
                    }
                }
                else {
                    dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
                }
        }
    }

    imshow("results - CVkMeans", dst_img);
    waitKey(0);

    return dst_img;
}

// 군집을 생성하는 함수
void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters) {
    RNG random(cv::getTickCount()); // OpenCV에서 무작위 값을 설정하는 함수

    for (int k = 0; k < n_cluster; k++) { // 군집별 계산
        // 무작위 좌표 획득
        Point centerKPoint;
        centerKPoint.x = random.uniform(0, imgInput.cols);
        centerKPoint.y = random.uniform(0, imgInput.rows);
        Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

        // 무작위 좌표의 화소값으로 군집별 중앙값 설정
        Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
        clustersCenters.push_back(centerK);

        vector<Point>ptInClustersK;
        ptInClusters.push_back(ptInClustersK);
    }
}

// 현재 픽셀과 군집의 픽셀 사이의 거리를 계산하는 함수
double computeColorDistance (Scalar pixel, Scalar clusterPixel) {
    double diffBlue = pixel.val[0] - clusterPixel[0];
    double diffGreen = pixel.val[1] - clusterPixel[1];
    double diffRed = pixel.val[2] - clusterPixel[2];
    
    // Euclidian distance를 활용
    double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));
    
    return distance;
}

// 평균값을 계산하여 군집에 대해 다시 clustering
double adjustClusterCenters(Mat src_img, int n_cluster, vector<Scalar>& clustersCenters,
    vector<vector<Point>>ptInClusters, double& oldCenter, double newCenter) {
    double diffChange;
    
    for (int k = 0; k < n_cluster; k++) { // 군집별 계산
        vector<Point>ptInCluster = ptInClusters[k];
        double newBlue = 0;
        double newGreen = 0;
        double newRed = 0;

        // 평균값 계산
        for (int i = 0; i < ptInCluster.size(); i++) {
            Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
            newBlue += pixel.val[0];
            newGreen += pixel.val[1];
            newRed += pixel.val[2];
        }

        newBlue /= ptInCluster.size();
        newGreen /= ptInCluster.size();
        newRed /= ptInCluster.size();
        
        // 계산한 평균값으로 군집 중앙값 대체
        Scalar newPixel(newBlue, newGreen, newRed);
        newCenter += computeColorDistance(newPixel, clustersCenters[k]);
        
        // 모든 군집에 대한 평균값도 같이 계산
        clustersCenters[k] = newPixel;
    }

    newCenter /= n_cluster;
    diffChange = abs(oldCenter - newCenter); //모든 군집에 대한 평균값 변화량 계산
    oldCenter = newCenter;

    return diffChange;

}

// 군집을 판별하는 함수
void findAssociatedCluster (Mat imgInput, int n_cluster, vector<Scalar> clustersCenters, vector<vector<Point>>& ptInClusters) {
    for (int r=0; r<imgInput.rows; r++) {
        for (int c=0; c<imgInput.cols; c++) {
            double minDistance = INFINITY;
            int closestClusterIndex = 0;
            Scalar pixel = imgInput.at<Vec3b>(r, c);
            
            for (int k=0; k<n_cluster; k++) { // 군집별 계산
                // 각 군집 중앙값과 차이를 계산
                Scalar clusterPixel = clustersCenters[k];
                double distance = computeColorDistance(pixel, clusterPixel);
                
                // 차이가 가장 적은 군집으로 좌표의 군집을 판별
                if (distance < minDistance) {
                    minDistance = distance;
                    closestClusterIndex = k;
                }
            }
            
            ptInClusters[closestClusterIndex].push_back(Point(c, r));
        }
    }
}

// 군집을 적용하여 영상의 픽셀 값을 바꾸는 함수
Mat applyFinalClusterToImage (Mat src_img, int n_cluster, vector<vector<Point>> ptInClusters, vector<Scalar> clustersCenters) {
    Mat dst_img(src_img.size(), src_img.type());
    
    for (int k=0; k<n_cluster; k++) {
        vector<Point> ptInCluster = ptInClusters[k];
        
        for (int j=0; j<ptInCluster.size(); j++) {
            dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
            dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
            dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
        }
    }
    
    return dst_img;
}

// 군집을 적용하여 영상의 픽셀 값을 바꾸는 함수
// 군집별 색을 무작위의 색으로 지정
Mat applyFinalClusterToRandomImage (Mat src_img, int n_cluster, vector<vector<Point>> ptInClusters, vector<Scalar> clustersCenters) {
    Mat dst_img(src_img.size(), src_img.type());

    // 랜덤 색으로 변경
    for (int k=0; k<n_cluster; k++) {
        vector<Point> ptInCluster = ptInClusters[k];
        
        clustersCenters[k].val[0] = rand() % 255;
        clustersCenters[k].val[1] = rand() % 255;
        clustersCenters[k].val[2] = rand() % 255;
        
        for (int j=0; j<ptInCluster.size(); j++) {
            dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
            dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
            dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
        }
    }

    return dst_img;
}

Mat MyKmeans(Mat src_img, int n_cluster, int flag = 0) {

    vector<Scalar>clustersCenters; // 군집 중앙값 벡터
    vector<vector<Point>>ptInClusters; // 군집별 좌표 벡터
    double threshold = 0.001;
    double oldCenter = INFINITY;
    double newCenter = 0;
    double diffChange = oldCenter - newCenter; // 군집 조정의 변화량


    // 초기 설정 - 군집 중앙값을 무작위로 할당 및 군집별 좌표값을 저장할 벡터 할당
    createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

    // 중앙값 조정 및 화소별 군집 판별
    // 반복적인 방법으로 군집 중앙값 조정
    // 설정한 임계값보다 군집 조정의 변화가 작을 때까지 반복
    while (diffChange > threshold) {
        // 초기화
        newCenter = 0;
        for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }
        
        // 현재의 군집 중앙값을 기준으로 군집 탐색
        findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);
        
        // 군집 중앙값 조절
        diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);
    }
    
    // 군집 중앙값으로만 이루어진 영상 생성
    Mat dst_img;
    if (flag == 0) dst_img = applyFinalClusterToImage(src_img, n_cluster, ptInClusters, clustersCenters);
    else if (flag == 1) dst_img = applyFinalClusterToRandomImage(src_img, n_cluster, ptInClusters, clustersCenters);
    
    return dst_img;

}

Mat isYellow(Mat src_img) {
    Scalar lowerYellow = Scalar(20, 20, 100);
    Scalar upperYellow = Scalar(32, 255, 255);
    
    Mat hsv_img = myBgr2Hsv(src_img);
    
    Mat result_img;
    
    Mat mask = myInRange(hsv_img, lowerYellow, upperYellow);
    cout<<"Yellow"<<endl;
    
    return mask;
}

Mat isRed(Mat src_img) {
    Scalar lowerRed = Scalar(0, 100, 100);
    Scalar upperRed = Scalar(5, 255, 255);
    
    Mat hsv_img = myBgr2Hsv(src_img);
    
    Mat result_img;
    
    Mat mask = myInRange(hsv_img, lowerRed, upperRed);
    cout<<"Red"<<endl;
    
    return mask;
}

Mat isOrange(Mat src_img) {
    Scalar lowerOrange = Scalar(1, 190, 200);
    Scalar upperOrange = Scalar(18, 255, 255);
    
    Mat hsv_img = myBgr2Hsv(src_img);
    
    Mat result_img;
    
    Mat mask = myInRange(hsv_img, lowerOrange, upperOrange);
    cout<<"Orange"<<endl;
    
    return mask;
}

void prob1() {
    // 1. 바나나
    cout<<"banana.jpeg: ";
    Mat banana_img = imread("banana.jpeg", 1);
    Mat banana_hsv = myBgr2Hsv(banana_img);
    
    Mat banana_result;
    
    Mat banana_mask = isYellow(banana_img);
    cvtColor(banana_mask, banana_mask, COLOR_BGR2GRAY);
    
    bitwise_and(banana_hsv, banana_hsv, banana_result, banana_mask);
    
    cvtColor(banana_result, banana_result, COLOR_HSV2BGR_FULL);
    
    Mat result_img;
    hconcat(banana_img, banana_result, result_img);
    imshow("Test window", result_img);
    waitKey(0);
    destroyWindow("Test window");
    
    // 2. 사과
    cout<<"apple.jpeg: ";
    Mat apple_img = imread("apple.jpeg", 1);
    Mat apple_hsv = myBgr2Hsv(apple_img);
    
    Mat apple_result;
    
    Mat apple_mask = isRed(apple_img);
    cvtColor(apple_mask, apple_mask, COLOR_BGR2GRAY);
    
    bitwise_and(apple_hsv, apple_hsv, apple_result, apple_mask);
    
    cvtColor(apple_result, apple_result, COLOR_HSV2BGR_FULL);
    
    hconcat(apple_img, apple_result, result_img);
    imshow("Test window", result_img);
    waitKey(0);
    destroyWindow("Test window");
    
    // 3. 오렌지
    cout<<"orange.jpeg: ";
    Mat orange_img = imread("orange.jpeg", 1);
    Mat orange_hsv = myBgr2Hsv(orange_img);
    
    Mat orange_result;
    
    Mat orange_mask = isOrange(orange_img);
    
    
    cvtColor(orange_mask, orange_mask, COLOR_BGR2GRAY);
    
    bitwise_and(orange_hsv, orange_hsv, orange_result, orange_mask);
    
    cvtColor(orange_result, orange_result, COLOR_HSV2BGR_FULL);
    
    hconcat(orange_img, orange_result, result_img);
    imshow("Test window", result_img);
    waitKey(0);
    destroyWindow("Test window");
}

void prob2() {
    Mat src_img = imread("balls.jpg", 1);
    
    Mat result = MyKmeans(src_img, 5, 0);
    Mat randomResult = MyKmeans(src_img, 5, 1);
    
    Mat temp;
    hconcat(src_img, result, temp);

    Mat result_img;
    hconcat(temp, randomResult, result_img);
    
    imshow("Test window", result_img);
    waitKey(0);
    destroyWindow("Test window");
}

int main() {
    prob1();
    prob2();
    
    return 0;
}
