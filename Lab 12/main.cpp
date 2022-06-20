#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

#include "opencv2/video/tracking.hpp"

#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

// Histogram을 구하는 함수
Mat getHistogram(Mat& src) {
    Mat histogram;
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 255;

    // 히스토그램 계산
    calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

    // 히스토그램 plot
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / number_bins);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
    // 정규화
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < number_bins; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
    }

    return histImage;
}

// 다양한 노출의 영상과 노출 시간 읽기
void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
    int numImages = 4;
//    static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
    static const float timesArray[] = { 1 / 326.0f, 1 / 120.0f, 1 / 60.0f, 1 / 30.0f };
    times.assign(timesArray, timesArray + numImages);

//    static const char* filenames[] = { "img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg" };
    static const char* filenames[] = { "9609.jpg", "9610.jpg", "9611.jpg", "9612.jpg" };
    
    for (int i=0; i<numImages; i++) {
        Mat im = imread(filenames[i]);
        images.push_back(im);
    }
}


int main() {
    // 이미지 불러오기
//    Mat img1 = imread("img_0.033.jpg", 0);
//    Mat img2 = imread("img_0.25.jpg", 0);
//    Mat img3 = imread("img_2.5.jpg", 0);
//    Mat img4 = imread("img_15.jpg", 0);
    
    Mat img1 = imread("9609.jpg", 0);
    Mat img2 = imread("9610.jpg", 0);
    Mat img3 = imread("9611.jpg", 0);
    Mat img4 = imread("9612.jpg", 0);
    
    // 히스토그램 생성
    Mat hist1 = getHistogram(img1);
    Mat hist2 = getHistogram(img2);
    Mat hist3 = getHistogram(img3);
    Mat hist4 = getHistogram(img4);
    
    // 히스토그램 결과를 창에 띄움
//    imshow("img_0.033", hist1);
//    imshow("img_0.25", hist2);
//    imshow("img_2.5", hist3);
//    imshow("img_15", hist4);
    
    imshow("1", hist1);
    imshow("2", hist2);
    imshow("3", hist3);
    imshow("4", hist4);
//    waitKey(0);
    
    // 다양한 노출의 영상과 노출 시간 읽기
    cout << "Reading images and exposure times ..." << endl;
    vector<Mat> images;
    vector<float> times;
    readImagesAndTimes(images, times);
    cout << "finished " << endl;
    
    // 영상 정렬
    // 삼각대 등을 이용해 완전히 고정된 상태에서 영상을 취득하여야 함
    cout << "Algining images ..." << endl;
    Ptr<AlignMTB> alignMTB = createAlignMTB();
    alignMTB->process(images, images);
    
    // Camera response function 복원
    cout << "Calculating Camera Response Function ..." << endl;
    Mat responseDebevec;
    Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
    calibrateDebevec->process(images, responseDebevec, times);
    cout << "----- CRF -----" << endl;
    cout << responseDebevec << endl;
    
    // 이미지 병합
    cout << "Merging images into one HDR image ..." << endl;
    Mat hdrDebevec;
    Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
    mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
    imwrite("hdrDebevec.hdr", hdrDebevec);
    cout << "saved hdrDebevec.hdr" << endl;
    
    // 24 -> 8 bit 톤 매핑
    // 24bit은 컴퓨터가 출력할 수 없기 때문
    cout << "Tonemaping using Drago's method ..." << endl;
    Mat IdrDrago;
    Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
    tonemapDrago->process(hdrDebevec, IdrDrago);
    IdrDrago = 3 * IdrDrago;
    imwrite("Idr-Drago.jpg", IdrDrago * 255);
    cout << "saved Idr-Drago.jpg" << endl;
    
    cout << "Tonemaping using Reinhard's method ..." << endl;
    Mat IdrReinhard;
    Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
    tonemapReinhard->process(hdrDebevec, IdrReinhard);
    imwrite("Idr-Reinhard.jpg", IdrReinhard * 255);
    cout << "saved Idr-Reinhard.jpg" << endl;
    
    cout << "Tonemaping using Mantiuk's method ..." << endl;
    Mat IdrMantiuk;
    Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
    tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
    IdrMantiuk = 3 * IdrMantiuk;
    imwrite("Idr-Mantiuk.jpg", IdrMantiuk * 255);
    cout << "saved Idr-Mantiuk.jpg" << endl;
    
    Mat drago = imread("Idr-Drago.jpg", 0);
    Mat mantiuk = imread("Idr-Mantiuk.jpg", 0);
    Mat reinhard = imread("Idr-Reinhard.jpg", 0);
    
    Mat histDrago = getHistogram(drago);
    Mat histMantiuk = getHistogram(mantiuk);
    Mat histReinhard = getHistogram(reinhard);
    
    imshow("histDrago", histDrago);
    imshow("histMantiuk", histMantiuk);
    imshow("histReinhard", histReinhard);
    waitKey(0);
    
}
