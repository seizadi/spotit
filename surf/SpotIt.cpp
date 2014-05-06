#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "SpotIt.h"

using namespace std;
using namespace cv;

Mat SpotIt::processMat(Mat& mat) {
    int centerX = mat.cols/2;
    
    Mat grayImage;
    cvtColor(mat, grayImage, CV_BGR2GRAY);
    
    // threshold(grayImage, grayImage, 200, 0, THRESH_TRUNC);
    
    cv::Rect leftRect(0, 0, centerX, mat.rows);
    cv::Rect rightRect(centerX, 0, centerX, mat.rows);
    cerr << "rightRect" << rightRect << endl;
    
    Mat leftMat(grayImage, leftRect);
    Features leftFeatures = processSubMat(leftMat);
    Mat rightMat(grayImage, rightRect);
    Features rightFeatures = processSubMat(rightMat);
    
//    BFMatcher matcher(NORM_L2);
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(leftFeatures.descriptors, rightFeatures.descriptors, matches);
    
    drawMatches(leftMat, leftFeatures.keypoints, rightMat, rightFeatures.keypoints, matches, mat,
                Scalar::all(-1), Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::DRAW_OVER_OUTIMG);
    
    Scalar blue(255, 0, 0);
    line(mat, Point(centerX, 0), Point(centerX, mat.cols), blue);
    
    return mat;
}

Features SpotIt::processSubMat(Mat& mat) {
    Features features;
    SurfFeatureDetector detector;
    detector.detect(mat, features.keypoints);
    SurfDescriptorExtractor extractor;
    Mat descriptors;
    extractor.compute(mat, features.keypoints, features.descriptors);
    
    return features;
}
