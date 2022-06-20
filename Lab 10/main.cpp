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

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


// Stitcher를 이용한 파노라마 영상 생성
void ex_panorama_simple() {
    Mat img;
    vector<Mat> imgs;

    img = imread("one.jpg", IMREAD_COLOR);
    imgs.push_back(img);

    img = imread("two.jpg", IMREAD_COLOR);
    imgs.push_back(img);
    
    img = imread("three.jpg", IMREAD_COLOR);
    imgs.push_back(img);

    Mat result;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Stitcher::Status status = stitcher->stitch(imgs, result);
    if (status != Stitcher::OK) {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        exit(-1);
    }

    imshow("ex_panorama_simple_result", result);
    waitKey();
}

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
    
    Mat img_gray_l, img_gray_r;
    cvtColor(img_l, img_gray_l, COLOR_BGR2GRAY);
    cvtColor(img_r, img_gray_r, COLOR_BGR2GRAY);
    
    Ptr<SurfFeatureDetector> Detector = SURF::create(300);
    vector<KeyPoint> kpts_obj, kpts_scene;
    Detector->detect(img_gray_l, kpts_obj);
    Detector->detect(img_gray_r, kpts_scene);
    
    Mat img_kpts_l, img_kpts_r;
    drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    imshow("img_kpts_l.png", img_kpts_l);
    imshow("img_kpts_r.png", img_kpts_r);
    waitKey();
    
    Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
    
    Mat img_des_obj, img_des_scene;
    Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
    Extractor->compute(img_gray_r, kpts_scene, img_des_scene);
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img_des_obj, img_des_scene, matches);
    
    Mat img_matches;
    drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches.png", img_matches);
    waitKey();
    
    // 대각선으로 매칭 된 것은 잘못된 매칭이므로 정제가 필요
    double dist_max = matches[0].distance;
    double dist_min = matches[0].distance;
    double dist;
    
    for (int i=0; i<img_des_obj.rows; i++) {
        dist = matches[i].distance;
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
    }
    
    cout << "max dist: " << dist_max << endl;
    cout << "min dist: " << dist_min << endl;
    
    vector<DMatch> matches_good;
    do {
        vector<DMatch> good_matches2;
        for (int i=0; i<img_des_obj.rows; i++) {
            if (matches[i].distance < thresh_dist * dist_min) {
                good_matches2.push_back(matches[i]);
            }
        }
        matches_good = good_matches2;
        thresh_dist -= 1;
    } while (thresh_dist != 2 && matches_good.size() > min_matches);
    
    Mat img_matches_good;
    drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches_good.png", img_matches_good);
    waitKey();
    
    vector<Point2f> obj, scene;
    for (int i=0; i<matches_good.size(); i++) {
        obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
        scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
    }
    
    Mat mat_homo = findHomography(scene, obj, RANSAC);
    
    Mat img_result;
    warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);

    Mat img_pano;
    img_pano = img_result.clone();
    imshow("connect image - img_pano", img_pano);
    waitKey();
    imshow("connect image - img_l", img_l);
    waitKey();
    
    Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
    
    Rect rect = Rect(img_l.cols-30, 0, 60, img_l.rows);

    
    img_l.copyTo(roi);
    medianBlur(img_pano(rect), img_pano(rect), 19);
    
    // 검은 부분 제거
    int cut_x = 0, cut_y = 0;
    for (int y=0; y<img_pano.rows; y++) {
        for (int x=0; x<img_pano.cols; x++) {
            if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
                img_pano.at<Vec3b>(y, x)[1] == 0 &&
                img_pano.at<Vec3b>(y, x)[2] == 0) {
                continue;
            }
            if (cut_x < x) cut_x = x;
            if (cut_y < y) cut_y = y;
        }
    }
    
    Mat img_pano_cut;
    
    img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
    imshow("img_pano_cut.png", img_pano_cut);
    waitKey();
    destroyAllWindows();
    
    return img_pano_cut;
    
}

void ex_panorama() {
    Mat matImage1 = imread("two.jpg", IMREAD_COLOR);
    Mat matImage2 = imread("one.jpg", IMREAD_COLOR);
    Mat matImage3 = imread("three.jpg", IMREAD_COLOR);
    
    if (matImage1.empty() || matImage2.empty() || matImage3.empty()) exit(-1);
    
    Mat result;
    flip(matImage1, matImage1, 1);
    flip(matImage2, matImage2, 1);
    
    result = makePanorama(matImage1, matImage2, 3, 60);
    flip(result, result, 1);
    result = makePanorama(result, matImage3, 3, 60);
    
    imshow("ex_panorama_result", result);
    waitKey();
}

void findBook(Mat img) {
    Mat scene = imread("Scene.jpg", IMREAD_COLOR);
    
    // 특징점 추출
    Mat img_gray;
    Mat scene_gray;
    
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    cvtColor(scene, scene_gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    vector<KeyPoint> img_keypoints, scene_keypoints;
    detector->detect(img_gray, img_keypoints);

    detector = SiftFeatureDetector::create();
    detector->detect(scene_gray, scene_keypoints);
    
    Mat img_result;
    drawKeypoints(img, img_keypoints, img_result);
    
    Mat scene_result;
    drawKeypoints(scene, scene_keypoints, scene_result);
    
    imshow("sift_img_result", img_result);
    waitKey();
    
    imshow("sift_img_result", scene_result);
    waitKey();
    
    // Brute Force 매칭
    Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    Mat img_des_obj, img_des_scene;
    
    extractor->compute(img_gray, img_keypoints, img_des_obj);
    extractor->compute(scene_gray, scene_keypoints, img_des_scene);
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img_des_obj, img_des_scene, matches);
    
    Mat img_matches;
    drawMatches(img_gray, img_keypoints, scene_gray, scene_keypoints, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches", img_matches);
    waitKey();
    
    // 매칭 결과 정제
    double thresh_dist = 3;
    int min_matches = 60;
    
    double dist_max = matches[0].distance;
    double dist_min = matches[0].distance;
    double dist;
    
    for (int i=0; i<img_des_obj.rows; i++) {
        dist = matches[i].distance;
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
    }
    
    cout << "max dist: " << dist_max << endl;
    cout << "min dist: " << dist_min << endl;
    
    vector<DMatch> matches_good;
    do {
        vector<DMatch> good_matches2;
        for (int i=0; i<img_des_obj.rows; i++) {
            if (matches[i].distance < thresh_dist * dist_min) {
                good_matches2.push_back(matches[i]);
            }
        }
        matches_good = good_matches2;
        thresh_dist -= 1;
    } while (thresh_dist != 2 && matches_good.size() > min_matches);
    
    Mat img_matches_good;
    drawMatches(img_gray, img_keypoints, scene_gray, scene_keypoints, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches_good.png", img_matches_good);
    waitKey();
    
    
    vector<Point2f> imgVector, sceneVector;
    if (matches_good.size()>=4) {
        cout << "Object found" << endl;
        for (int i=0; i<matches_good.size(); i++) {
            imgVector.push_back(img_keypoints[matches_good[i].queryIdx].pt);
            sceneVector.push_back(scene_keypoints[matches_good[i].trainIdx].pt);
        }
    }
    
    Mat H = findHomography(imgVector, sceneVector, RANSAC);
    
    // 책의 윤곽 그리기
    vector<Point2f> imgCorners, sceneCorners;
    imgCorners.push_back(Point2f(0.f, 0.f));
    imgCorners.push_back(Point2f(img.cols - 1.f, 0.f));
    imgCorners.push_back(Point2f(img.cols - 1.f, img.rows - 1.f));
    imgCorners.push_back(Point2f(0.f, img.rows - 1.f));
    perspectiveTransform(imgCorners, sceneCorners, H);
    
    vector<Point> dstCorners;
    for (Point2f pt : sceneCorners) {
        dstCorners.push_back(Point(cvRound(pt.x + img.cols), cvRound(pt.y)));
    }
    
    polylines(img_matches_good, dstCorners, true, Scalar(0, 255, 0), 3, LINE_AA);
    
    imshow("result", img_matches_good);
    waitKey(0);
    destroyAllWindows();
    
}

int main() {
    ex_panorama_simple();
    ex_panorama();
    
    Mat book1 = imread("Book1.jpg", IMREAD_COLOR);
    Mat book2 = imread("Book2.jpg", IMREAD_COLOR);
    Mat book3 = imread("Book3.jpg", IMREAD_COLOR);
    findBook(book1);
    findBook(book2);
    findBook(book3);
    return 0;
}
