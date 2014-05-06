#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;


class SpotIt {
public:
    cv::Mat processMat(cv::Mat& mat);
    cv::Mat processSubMat(cv::Mat& mat);
    cv::Mat processSubMat2(cv::Mat& mat);
    cv::Mat processSubMat3(cv::Mat& mat);
    cv::Mat processSubMat4(cv::Mat& mat);
    cv::Mat processSubMat5(cv::Mat& mat);
    cv::Mat processSubMat6(cv::Mat& mat);
    cv::Mat processSubMat7(cv::Mat& mat);
    cv::Mat processSubMat8(cv::Mat& mat);
    cv::Mat processSubMat9(cv::Mat& mat);
    cv::Mat processSubMat10(cv::Mat& mat);
    cv::Mat processSubMat11(cv::Mat& mat);
    cv::Mat processSubMat12(cv::Mat& mat);
    cv::Mat processSubMat13(cv::Mat& mat);
    cv::Mat processSubMat14(cv::Mat& mat);
    string type2str(int type);
    void getImoments(double hu[7], double result[3]);
    void findInvariantMoments(std::vector<cv::Point>& contour, double invariantMoment[2]);
    Mat refineObjects(Mat& mat);
};
