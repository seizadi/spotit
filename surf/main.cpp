#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SpotIt.h"

using namespace std;
using namespace cv;

const char*const WindowName("spot-it");

int main(int argc, char** argv) {
    Scalar red(255, 0, 0);
    Scalar green(0, 255, 0);
    Scalar blue(0, 0, 255);
    
    Mat image;
    SpotIt spotIt;

    VideoCapture *cap = 0;
    namedWindow(WindowName, WINDOW_NORMAL);
    resizeWindow(WindowName, 800, 600);

    if (argc > 1) {
        image = imread(argv[1]);
    } else {
        cap = new VideoCapture(0);
        if(!cap->isOpened()) {
            return -1;
        }
    }
    
    for (;;) {
        if(cap) {
            *cap >> image;
        }

        image = spotIt.processMat(image);

        cv::imshow(WindowName, image);
        cv::waitKey(1);
        if (!cap) {
            waitKey(0);
            break;
        }
    }
    
    if (cap) {
        delete cap;
    }

    return 0;
}
