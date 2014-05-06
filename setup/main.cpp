#include "SpotIt.h"

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

	Mat origImage = image.clone();
#if 0
	spotIt.processMat(image);
	image = origImage.clone();
	destroyAllWindows();
	//image = spotIt.processSubMat(image);
	//image = origImage.clone();
	spotIt.processSubMat2(image);
	image = origImage.clone();
	destroyAllWindows();
	//spotIt.processSubMat3(image);
	//image = spotIt.processSubMat4(image);
	image = spotIt.processSubMat5(image);
	image = origImage.clone();
	destroyAllWindows();
	//image = spotIt.processSubMat6(image);
	//image = spotIt.processSubMat7(image);
	//image = spotIt.processSubMat8(image);
	//image = spotIt.processSubMat9(image);
	//image = spotIt.processSubMat10(image);
	//image = spotIt.processSubMat11(image);
	//image = spotIt.processSubMat12(image);
	//image = spotIt.processSubMat13(image);
	//image = origImage.clone();
#endif
	image=spotIt.processSubMat14(image);
	destroyAllWindows();
        image = imread("../disc-2.jpg");
	image=spotIt.processSubMat14(image);

        //cv::imshow(WindowName, image);
        //cv::waitKey(1);
        //if (!cap) {
        //    waitKey(0);
        //    break;
        //}
    }
    
    if (cap) {
        delete cap;
    }

    return 0;
}
