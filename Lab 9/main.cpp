#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;

void cvRotation() {
    Mat src = imread("Lenna.png", 1);
    Mat dst, matrix;
    
    Point center = Point(src.cols/2, src.rows/2); // 중심점
    // 중심점을 기준으로 회전
    matrix = getRotationMatrix2D(center, 45.0, 1.0);

    warpAffine(src, dst, matrix, src.size());
    
    imshow("nonrot", src);
    imshow("rot - cvRotation", dst);
    waitKey(0);
    
    destroyAllWindows();
}

// 2D transformation matrix 제작
Mat getMyRotationMatrix2D(Point center, double angle){
    Mat matrix1, matrix2, matrix3;
    
    // 원점으로 이동
    matrix1 = (Mat_<double>(3,3) <<
               1, 0, -center.x,
               0, 1, -center.y,
               0, 0, 1
                   );
    // 회전
    matrix2 = (Mat_<double>(3,3) <<
                   cos(angle*CV_PI/180), sin(angle*CV_PI/180), 0,
                   -sin(angle*CV_PI/180), cos(angle*CV_PI/180), 0,
                   0, 0, 1
                   );
    
    // 원래 좌표로 이동
    matrix3 = (Mat_<double>(3,3) <<
               1, 0, center.x,
               0, 1, center.y,
               0, 0, 1
                   );

    // 하나의 행렬로 합침
    Mat matrix = matrix3 * matrix2 * matrix1;
    
    return matrix;
    
}

// 직접 구현한 transformation matrix를 사용하여 영상을 회전시키는 함수
void myRotation() {
    Mat src = imread("Lenna.png", 1);
    Mat dst, matrix;
    
    Point center = Point(src.cols/2, src.rows/2);
    matrix = getMyRotationMatrix2D(center, 45.0);
    warpPerspective(src, dst, matrix, src.size());
    
    imshow("nonrot", src);
    imshow("rot - myRotation", dst);
    waitKey(0);
    
    destroyAllWindows();
}

// corner dectection을 이용하여 꼭짓점을 검출
// 검출한 꼭짓점을 원으로 표시
Mat detectAngle(Mat img) {
    if(img.empty()) {
        cout << "Empty image!\n";
        exit(-1);
    }
    
    resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);
    
    // 꼭짓점을 제외한 점이 corner로 검출되어 내부 색을 채워주었음
    for (int y=0; y<img.rows; y++) {
        for (int x=0; x<img.cols; x++) {
            bool black = (int)img.at<Vec3b>(y, x)[0] == 0 && (int)img.at<Vec3b>(y, x)[1] == 0 && (int)img.at<Vec3b>(y, x)[2] == 0;
            if(!black) {
                img.at<Vec3b>(y, x)[0] = 196;
                img.at<Vec3b>(y, x)[1] = 114;
                img.at<Vec3b>(y, x)[2] = 68;
            }
        }
    }
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat harr;
    cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);
    
    int thresh = 120; // threshold
    Mat result = img.clone();
    
    for (int y=0; y<harr.rows; y++) {
        for (int x=0; x<harr.cols; x++) {
            if((int)harr.at<float>(y, x) > thresh) {
                circle(result, Point(x, y), 3, Scalar(255, 0, 255), -1, 4, 0);
            }
        }
    }
    
    for (int y=0; y<result.rows; y++) {
        for (int x=0; x<result.cols; x++) {
            bool pink = (int)result.at<Vec3b>(y, x)[0] == 255 && (int)result.at<Vec3b>(y, x)[1] == 0 && (int)result.at<Vec3b>(y, x)[2] == 255;
            if(!pink) {
                result.at<Vec3b>(y, x)[0] = 255;
                result.at<Vec3b>(y, x)[1] = 255;
                result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }
    return result;
}

// blob detection을 사용하여 원의 개수를 세는 함수
vector<Point2f> countCircle(Mat img) {
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 10;
    params.filterByCircularity = true;
    params.minCircularity = 0.895;
    params.filterByConvexity = true;
    params.minConvexity = 0.4;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    Mat dst;
    drawKeypoints(img, keypoints, dst, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    vector<Point2f> centers;
    for (auto keypoint : keypoints) {
        centers.push_back(keypoint.pt);
    }
    
    return centers;
}

// 네 꼭짓점을 직접 찾고 시점 변환 수행
void cvPerspective() {
    Mat src = imread("card_per.png", 1);
    Mat dst, matrix;
    
    Point2f srcQuad[4];
    srcQuad[0] = Point2f(124.f, 132.f);
    srcQuad[1] = Point2f(375.f, 177.f);
    srcQuad[2] = Point2f(72.f, 355.f);
    srcQuad[3] = Point2f(427.f, 320.f);
    
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(0.f, 0.f);
    dstQuad[1] = Point2f(src.cols - 1.f, 0.f);
    dstQuad[2] = Point2f(0.f, src.rows - 1.f);
    dstQuad[3] = Point2f(src.cols - 1.f, src.rows - 1.f);

    matrix = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(src, dst, matrix, src.size());
    
    resize(dst, dst, Size(500, 300));
    
    imshow("nonrot", src);
    imshow("rot", dst);
    waitKey(0);
    
    destroyAllWindows();
}

// 네 꼭짓점을 자동으롤 탐색한 뒤 시점 변환 수행
void autoPerspective() {
    // 직접 측정한 네 꼭짓점의 좌표들
    // 124 132
    // 375 177
    // 72 355
    // 427 320
    Mat src = imread("card_per.png", 1);
    Mat dst, matrix;
    
    // 꼭짓점 검출
    Mat img = detectAngle(src);
    
    // 검출한 원의 중심 4개를 centers 벡터에 저장
    vector<Point2f> centers = countCircle(img);
    
    // centers 벡터의 원소를 시점 변환에 사용
    Point2f srcQuad[4];
    srcQuad[0] = centers[3];
    srcQuad[1] = centers[2];
    srcQuad[2] = centers[0];
    srcQuad[3] = centers[1];
    
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(0.f, 0.f);
    dstQuad[1] = Point2f(src.cols - 1.f, 0.f);
    dstQuad[2] = Point2f(0.f, src.rows - 1.f);
    dstQuad[3] = Point2f(src.cols - 1.f, src.rows - 1.f);

    matrix = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(src, dst, matrix, src.size());
    
    resize(dst, dst, Size(500, 300));
    
    imshow("nonrot", src);
    imshow("rot", dst);
    waitKey(0);
    
    destroyAllWindows();
}

int main() {
    cvRotation();
    myRotation();
    autoPerspective();
    
    return 0;
}
