#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void foxGrabCut() {
    Mat img = imread("fox.jpg", 1);
    imshow("src_img", img);
    
// bounding box 확인
//    rectangle(img, Point(240, 20), Point(520, 340), Scalar(255, 0, 0), 3);
//    imshow("Test window", img);
    
    // 전경 객체를 포함하는 직사각형
    Rect rect = Rect(Point(240, 20), Point(520, 340));

    Mat result;
    Mat bgModel, fgModel;
    // Grab Cut을 진행
    grabCut(img, result, rect, bgModel, fgModel, 5, GC_INIT_WITH_RECT);

    // 직사각형 내부의 화소 초기값과 전경일 가능성이 있는 화소를 비교하여
    //동일한 값을 갖는 화소를 추출해 분할한 이진 영상을 얻음
    compare(result, GC_PR_FGD, result, CMP_EQ);
    // GC_PR_FGD: GrabCut class foreground 픽셀
    // CMP_EQ: compare 옵션, equal
    
    Mat mask(img.size(), CV_8UC3, Scalar(255, 255, 255));
    img.copyTo(mask, result);

    // 결과 영상 출력
    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}

void zebraGrabCut() {
    Mat img = imread("zebra.jpg", 1);
    imshow("src_img", img);
    
// bounding box 확인
//    rectangle(img, Point(160, 50), Point(800, 550), Scalar(255, 0, 0), 3);
//    imshow("Test window", img);
    
    Rect rect = Rect(Point(160, 50), Point(800, 550));

    Mat result;
    Mat bgModel, fgModel;
    grabCut(img, result, rect, bgModel, fgModel, 5, GC_INIT_WITH_RECT);

    compare(result, GC_PR_FGD, result, CMP_EQ);

    Mat mask(img.size(), CV_8UC3, Scalar(255, 255, 255));
    img.copyTo(mask, result);

    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}

void playerGrabCut() {
    Mat img = imread("player.png", 1);
    imshow("src_img", img);
    
// bounding box 확인
//    rectangle(img, Point(150, 30), Point(500, 350), Scalar(255, 0, 0), 3);
//    imshow("Test window", img);
    
    Rect rect = Rect(Point(150, 30), Point(500, 350));

    Mat result;
    Mat bgModel, fgModel;
    grabCut(img, result, rect, bgModel, fgModel, 5, GC_INIT_WITH_RECT);

    compare(result, GC_PR_FGD, result, CMP_EQ);

    Mat mask(img.size(), CV_8UC3, Scalar(255, 255, 255));
    img.copyTo(mask, result);

    imshow("mask", mask);
    imshow("result", result);

    waitKey(0);
}

int main() {
    foxGrabCut();
    zebraGrabCut();
    playerGrabCut();
    return 0;
}
