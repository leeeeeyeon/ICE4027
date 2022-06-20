#include <iostream>
#include <ctime>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void prob1(Mat frame) {
    vector<Scalar> colors;
    
    // 랜덤 컬러 생성
    RNG rng;
    for (int i=0; i<100; i++) {
    int r = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int b = rng.uniform(0, 256);
    colors.push_back(Scalar(r, g, b));
    }

    Mat old_frame, old_gray;
    old_frame = imread("cream.jpeg", IMREAD_COLOR);
    vector<Point2f> p0, p1;

    // 첫 번재 프레임에서 특징점 추출
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // 마스크 생성
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    while (true) {
        Mat frame_gray;
            
        if (frame.empty()) {
            break;
        }
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // optical flow 계산
        vector<uchar> status;
        vector<float> err;

        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(50,50), 2, criteria);

        // 특징점들 중 좋은 특징점들을 추림
        vector<Point2f> good_new;
        for (uint i=0; i<p0.size(); i++) {
            if (status[i]==1) {
                good_new.push_back(p1[i]);
                
                // 경로 그리기
                line(mask, p1[i], p0[i], colors[i], 2);
                circle(frame, p1[i], 5, colors[i], -1);
            }
        }
        Mat img;
        
        // 결과 영상 화면에 띄움
        add(frame, mask, img);
        addWeighted(img, 0.7, old_frame, 0.3, 0, img);
        imshow("Frame", img);

        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27) break;
        
        // 이전 프레임을 업데이트
        old_gray = frame_gray.clone();
        p0 = good_new;
    }
}

void prob2() {
    // 영상을 불러온다
    VideoCapture cap(samples::findFile("5479.mov"));
    
    // 영상이 열리지 않을 경우, 에러 메시지 출력
    if (!cap.isOpened()) {
        cerr << "Unable to open file!" << endl;
    }
    
    Mat flow, frame;
    UMat  flowUmat, prev;
    while(true) {
        bool Is = cap.grab();
        if (Is == false) {
        cout << "Video Capture Fail" << endl;
        break;
        }

        Mat next;
        Mat original;
        cap.retrieve(next, CAP_OPENNI_BGR_IMAGE);
        resize(next, next, Size(640, 800));
        next.copyTo(original);
        cvtColor(next, next, COLOR_BGR2GRAY);

        // 파네백 알고리즘을 사용하여 flow 계산
        if (prev.empty() == false) {
            calcOpticalFlowFarneback(prev, next, flowUmat, 0.5, 3, 15, 3, 5, 1.2, 0);
            flowUmat.copyTo(flow);
            
            // 결과 시각화
            for (int y = 0; y < original.rows; y += 8) {
                for (int x = 0; x < original.cols; x += 8)  {
                    const Point2f flowatxy = flow.at<Point2f>(y, x);
                    line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(0,255,0));
                }
            }

            // 결과를 창에 띄움
            namedWindow("prev", WINDOW_AUTOSIZE);
            imshow("prev", original);
            next.copyTo(prev);
        }
        else {
            next.copyTo(prev);
        }
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27) {
             break;
        }
    }
}

int main() {
    Mat frame1 = imread("cream2.jpeg", IMREAD_COLOR);
    Mat frame2 = imread("cream3.jpeg", IMREAD_COLOR);
    prob1(frame1);
    prob1(frame2);
    prob2();
}
