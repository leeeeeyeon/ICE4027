#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

// Harris Corner detection
void cvHarrisCorner() {
    Mat img = imread("ship.png");
    if(img.empty()) {
        cout << "Empty image!\n";
        exit(-1);
    }
    
    resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat harr;
    cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);
    
    int thresh = 125;
    Mat result = img.clone();
    
    for (int y=0; y<harr.rows; y++) {
        for (int x=0; x<harr.cols; x++) {
            if((int)harr.at<float>(y, x) > thresh) circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
        }
    }
    
    imshow("Source image", img);
    imshow("Harris image", harr_abs);
    imshow("Target image", result);
    
    waitKey(0);
    destroyAllWindows();
}

// SIFT
void cvFeatureSIFT() {
    Mat img = imread("church.jpg", 1);
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints);
    
    Mat result;
    drawKeypoints(img, keypoints, result);
    imwrite("sift_result.jpg", result);
    imshow("Sift result", result);
    
    waitKey(0);
    destroyAllWindows();
    
}

// Blob detection
void cvBlobDetection() {
    Mat img = imread("coin.png", IMREAD_COLOR);
    
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 9000;
    params.filterByCircularity = true;
    params.minCircularity = 0.8064;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    Mat result;
    drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cout << "동전의 개수: " << keypoints.size() << endl; // 14개
    imshow("keypoints", result);
    waitKey(0);
    destroyAllWindows();
}

// 투시 변환을 수행하는 함수
Mat warpImage(Mat src) {
    Mat dst;
    Point2f src_p[4], dst_p[4];
    
    int row = src.rows;
    int col = src.cols;
    
    src_p[0] = Point2f(0, 0);
    src_p[1] = Point2f(col, 0);
    src_p[2] = Point2f(0, row);
    src_p[3] = Point2f(col, row);
    
    dst_p[0] = Point2f(0, 0);
    dst_p[1] = Point2f(col, 0);
    dst_p[2] = Point2f(0, row);
    dst_p[3] = Point2f(col - 200, row - 200);
    
    Mat perspect_mat = getPerspectiveTransform(src_p, dst_p);
    
    warpPerspective(src, dst, perspect_mat, Size(col, row));
    
    return dst;
}

// 밝기를 변화시키는 함수
Mat changeBright(Mat src) {
    Mat dst = src + Scalar(30, 30, 30);
    
    return dst;
}

// 원본 영상과 투시 변환 및 밝기 변화가 수행된 영상의 SIFT 특징점 비교
void warpFeatureSIFT() {
    // 원본 영상
    Mat img = imread("church.jpg", 1);
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints);
    
    Mat result;
    drawKeypoints(img, keypoints, result);
    
    // ----------------------------------------------------
    // 투시변환과 밝기 수행을 적용한 영상
    
    Mat warp_img = changeBright(img);
    warp_img = warpImage(warp_img);
    
    Mat warp_gray;
    cvtColor(warp_img, warp_gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> warp_detector = SiftFeatureDetector::create();
    vector<KeyPoint> warp_keypoints;
    detector->detect(warp_gray, warp_keypoints);
    
    Mat warp_result;
    drawKeypoints(warp_img, warp_keypoints, warp_result);
    
    Mat final;
    hconcat(result, warp_result, final);
    
    imshow("compare", final);
    
    waitKey(0);
    destroyAllWindows();
    
}

// corner dectection을 이용하여 꼭짓점에 원을 그림
Mat detectAngle(Mat img) {
    if(img.empty()) {
        cout << "Empty image!\n";
        exit(-1);
    }
    
    resize(img, img, Size(900, 900), 0, 0, INTER_CUBIC);
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat harr;
    cornerHarris(gray, harr, 5, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);
    
    int thresh = 120; // threshold
    Mat result = img.clone();
    
    for (int y=0; y<harr.rows; y++) {
        for (int x=0; x<harr.cols; x++) {
            if((int)harr.at<float>(y, x) > thresh) {
                circle(result, Point(x, y), 7, Scalar(0, 0, 255), -1, 4, 0);
            }
        }
    }
    
    imshow("Target image", result);
    
    for (int y=0; y<result.rows; y++) {
        for (int x=0; x<result.cols; x++) { // B G R 순서
            if((int)result.at<Vec3b>(y, x)[2] != 255) {
                result.at<Vec3b>(y, x)[0] = 255;
                result.at<Vec3b>(y, x)[1] = 255;
                result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }
    
    
    return result;
}

// blob detection을 사용하여 원의 개수를 세는 함수
int countCircle(Mat img) {
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 1;
    params.filterByCircularity = true;
    params.minCircularity = 0.6;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    Mat dst;
    drawKeypoints(img, keypoints, dst, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cout << "꼭짓점의 개수: " << keypoints.size() << endl;
    
    return (int)keypoints.size();
}

// 몇 각형인지 출력
void detectPolygon(int angle) {
    switch (angle) {
        case 3:
            cout << "삼각형입니다." << endl;
            break;
        case 4:
            cout << "사각형입니다." << endl;
            break;
        case 5:
            cout << "오각형입니다." << endl;
            break;
        case 6:
            cout << "육각형입니다." << endl;
            break;
        default:
            cout << "삼각형, 사각형, 오각형, 육각형 중 하나가 아닙니다." << endl;
            break;
    }
}

// main 함수에서 실행할 함수
void printPolygon() {
    // 삼각형
    Mat img = imread("triangle.png", IMREAD_COLOR);
    Mat temp = detectAngle(img);
    int angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 사각형
    img = imread("rectangle.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 오각형
    img = imread("pentagon.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 육각형
    img = imread("hexagon.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
}

int main() {
//    실습 때 한 내용
//    cvHarrisCorner();
//    cvFeatureSIFT();
    cvBlobDetection();
    warpFeatureSIFT();
    printPolygon();
   
    return 0;
}
