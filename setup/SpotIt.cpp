#include "SpotIt.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/photo/photo.hpp"

void SpotIt::getImoments(double hu[7], double result[2])
{
  double ma[7];
  int i, sma;
  double eps = 1.e-5;
  
  result[0] = 0;
  result[1] = 0;

  ma[0] = hu[0];
  ma[1] = hu[1];
  ma[2] = hu[2];
  ma[3] = hu[3];
  ma[4] = hu[4];
  ma[5] = hu[5];
  ma[6] = hu[6];

  for (int method=1; method < 3; method++) {
    switch (method) {
      case 1:
        {
	  for( i = 0; i < 7; i++ )
            {
	      double ama = fabs( ma[i] );

	      if( ma[i] > 0 )
		sma = 1;
	      else if( ma[i] < 0 )
		sma = -1;
	      else
		sma = 0;

	      if( ama > eps )
                {
		  ama = 1. / (sma * log10( ama ));
		  result[0] += fabs( ama );
                }
            }
	  break;
        }

    case 2:
      {
	for( i = 0; i < 7; i++ )
	  {
	    double ama = fabs( ma[i] );

	    if( ma[i] > 0 )
	      sma = 1;
	    else if( ma[i] < 0 )
	      sma = -1;
	    else
	      sma = 0;

	    if( ama > eps )
	      {
		ama = sma * log10( ama );
		result[1] += fabs( ama );
	      }
	  }
	break;
      }
    }
  }
}

void SpotIt::findInvariantMoments(std::vector<cv::Point>& contour, double im[2])
{
  Moments mu;
  Point2f mc;
  double hu [7];

  im[0] = 0;
  im[1] = 0;

  //  if (!contour.empty() {
  {
      // FIXME should the binaryImage flag be true or false for contours? For now set to true!
      // Get the moments
      mu = moments( contour, true );
      //  Get the mass centers:
      // TODO - Not really using it but calculate it in case we need it later!
      mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );
      HuMoments(mu, hu);
      getImoments(hu, im);
  }
}

Mat SpotIt::refineObjects(Mat& mat) {

  cv::Mat origImage = mat.clone();
  cv::Mat image = origImage.clone();

  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);
  blur( image, image, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image, image, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  vector<Vec4i> hierarchy;
  //  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );
  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  for (size_t idx = 0; idx < contours.size(); idx++) {
    double area = contourArea(contours[idx]);
    if (area > 1) {
      //printf("Contour Area %f \n", contourArea(contours[idx]));
      drawContours(contourImage, contours, idx, cv::Scalar(255,255,255),4,8);
      //cv::Rect brect = cv::boundingRect(contours[idx]);
      //cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
    }  
  }

  return (contourImage);
}


string SpotIt::type2str(int type) 
{
  string r;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  

  switch ( depth ) {
      
  case CV_8U:  r = "8U";
    break;
      
  case CV_8S:  r = "8S";
    break;
      
  case CV_16U: r = "16U";
    break;
      
  case CV_16S: r = "16S";
    break;
      
  case CV_32S: r = "32S";
    break;
      
  case CV_32F: r = "32F";
    break;
      
  case CV_64F: r = "64F";
    break;
      
  default:     r = "User";
    break;
      
  }
  
  r += "C";
  r += (chans+'0');
  return r;
}

Mat SpotIt::processMat(Mat& mat) {
    int centerX = mat.cols/2;
    
    Scalar blue(255, 0, 0);
    line(mat, Point(centerX, 0), Point(centerX, mat.cols), blue);
    
    cv::Mat img1 = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
    cv::Mat img2 = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img1, keypoints_1 );
    detector.detect( img2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img1, keypoints_1, descriptors_1 );
    extractor.compute( img2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    BFMatcher matcher(NORM_L2);
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints_1, img2, keypoints_2, matches, img_matches );

    //-- Show detected matches
    imshow("Matches", img_matches );

    waitKey(0);

    return mat;
}

Mat SpotIt::processSubMat(Mat& mat) {

  int centerX = mat.cols/2;
  
  cv::Mat img1 = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  cv::Mat img2 = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  int N = 200;

  for (int r = 0; r < img1.rows; r += N)
    for (int c = 0; c < img1.cols; c += N)
      {
	cv::Mat tile1 = img1(cv::Range(r, min(r + N, img1.rows)),
			     cv::Range(c, min(c + N, img1.cols)));
        cv::Mat tile2 = img2(cv::Range(r, min(r + N, img2.rows)),
			     cv::Range(c, min(c + N, img2.cols)));

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect( tile1, keypoints_1 );
	detector.detect( tile2, keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute( tile1, keypoints_1, descriptors_1 );
	extractor.compute( tile2, keypoints_2, descriptors_2 );

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match( descriptors_1, descriptors_2, matches );

	//-- Draw matches
	Mat img_matches;
	drawMatches( tile1, keypoints_1, tile2, keypoints_2, matches, img_matches );

	//-- Show detected matches
	imshow("Matches", img_matches );

	waitKey(0);
      }

    return mat;

}

Mat SpotIt::processSubMat2(Mat& mat) {

  int centerX = mat.cols/2;
  
  cv::Mat image = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //  cv::Mat img1 = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //  cv::Mat img2 = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);
  string ty =  type2str( image.type() );
  //printf("Matrix: %s %dx%d \n", ty.c_str(), image.cols, image.rows );
  
  cv::threshold(image, image, 128, 255, CV_THRESH_BINARY);
  ty =  type2str( image.type() );
  //printf("Matrix: %s %dx%d \n", ty.c_str(), image.cols, image.rows );

  // std::vector<Vec3f> circles;
  // HoughCircles( image, circles, HOUGH_GRADIENT, 1, src_gray.rows/8, cannyThreshold, accumulatorThreshold, 0, 0 );
  



  // Did not find any benefit to additional compensation
  // double threshold = 128; // needs adjustment.
  // int n_erode_dilate = 1; // needs adjustment.
  // cv::blur(image, image, cv::Size(5,5));
  // cv::threshold(image, image, threshold, 255,CV_THRESH_BINARY_INV);
  // cv::erode(image, image, cv::Mat(),cv::Point(-1,-1),n_erode_dilate);
  // cv::dilate(image, image, cv::Mat(),cv::Point(-1,-1),n_erode_dilate);

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
  // cv::findContours( contourOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
  // cv::findContours( contourOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
  // cv::findContours( contourOutput, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
  // cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_LINK_RUNS );

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);
  cout << contours.size();
  for (size_t idx = 0; idx < contours.size(); idx++) {
    cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
    //    cv::Rect brect = cv::boundingRect(contours[idx]);
    //    cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
  }

  //  std::vector<cv::Point> points;
  //  for (size_t i=0; i<contours.size(); i++) {
  //    for (size_t j = 0; j < contours[i].size(); j++) {
  //      cv::Point p = contours[i][j];
  //      points.push_back(p);
  //    }
  //  }
  
  // And process the points or contours to pick up specified object.
  // for example: draws rectangle on original image.
  //  if(points.size() > 0) {
  //    cv::Rect brect = cv::boundingRect(cv::Mat(points).reshape(2));
  //    cv::rectangle(image, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
  //  }
  
  cv::imshow("Input Image", image);
  cvMoveWindow("Input Image", 0, 0);
  cv::imshow("Contours", contourImage);
  cvMoveWindow("Contours", 600, 0);
  cv::waitKey(0);

  return mat;

}


Mat SpotIt::processSubMat3(Mat& mat) {

  int centerX = mat.cols/2;
  
  cv::Mat src = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //  cv::Mat img1 = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //  cv::Mat img2 = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  blur(src, src, Size(15,15));
  
  imshow("blurred", src);
  
  Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
  
  Mat bestLabels, centers, clustered;
  
  vector<Mat> bgr;
  
  cv::split(src, bgr);
  
  // i think there is a better way to split pixel bgr color
  for(int i=0; i<src.cols*src.rows; i++) {
      p.at<float>(i,0) = (i/src.cols) / src.rows;
      p.at<float>(i,1) = (i%src.cols) / src.cols;
      p.at<float>(i,2) = bgr[0].data[i] / 255.0;
      p.at<float>(i,3) = bgr[1].data[i] / 255.0;
      p.at<float>(i,4) = bgr[2].data[i] / 255.0;
  }
  

  int K = 8;
  
  cv::kmeans(p, K, bestLabels,
	     TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
	     3, KMEANS_PP_CENTERS, centers);
  
  int colors[K];
  
  for(int i=0; i<K; i++) {
      colors[i] = 255/(i+1);
  }
  
  // i think there is a better way to do this mayebe some Mat::reshape?
  clustered = Mat(src.rows, src.cols, CV_32F);
  
  for(int i=0; i<src.cols*src.rows; i++) {
      clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
      //      cout << bestLabels.at<int>(0,i) << " " << 
      //              colors[bestLabels.at<int>(0,i)] << " " << 
      //              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
      //              endl;
  }

  clustered.convertTo(clustered, CV_8U);
  imshow("clustered", clustered);
  waitKey(0);

  return mat;
}

Mat SpotIt::processSubMat4(Mat& mat) {

  Mat image;

  // fastNlMeansDenoisingColored(mat, image, 3, 3, 7, 21 );
  //  fastNlMeansDenoisingColored(mat, image, 10, 19, 7, 21 );
  fastNlMeansDenoisingColored(mat, image, 7, 3, 7, 21 );

  int centerX = image.cols/2;
  
  //cv::Mat src = image(cv::Range(0,image.rows), cv::Range(0,centerX));
  cv::Mat src = image(cv::Range(0,image.rows), cv::Range(centerX,image.cols));
  //  cv::Mat img1 = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //  cv::Mat img2 = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  // blur(src, src, Size(15,15));
  // blur(src, src, Size(3,3));
  // blur(src, src, Size(7,7));
  // blur(src, src, Size(11,11));
  blur(src, src, Size(13,13));
  
  imshow("blurred", src);
  waitKey(0);
  
  Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
  
  Mat bestLabels, centers, clustered;
  
  vector<Mat> bgr;
  
  cv::split(src, bgr);
  
  // i think there is a better way to split pixel bgr color
  for(int i=0; i<src.cols*src.rows; i++) {
      p.at<float>(i,0) = (i/src.cols) / src.rows;
      p.at<float>(i,1) = (i%src.cols) / src.cols;
      p.at<float>(i,2) = bgr[0].data[i] / 255.0;
      p.at<float>(i,3) = bgr[1].data[i] / 255.0;
      p.at<float>(i,4) = bgr[2].data[i] / 255.0;
  }
  

  int K = 8;
  
  cv::kmeans(p, K, bestLabels,
	     TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
	     3, KMEANS_PP_CENTERS, centers);
  
  int colors[K];
  
  for(int i=0; i<K; i++) {
      colors[i] = 255/(i+1);
  }
  
  // i think there is a better way to do this mayebe some Mat::reshape?
  clustered = Mat(src.rows, src.cols, CV_32F);
  
  for(int i=0; i<src.cols*src.rows; i++) {
      clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
      //      cout << bestLabels.at<int>(0,i) << " " << 
      //              colors[bestLabels.at<int>(0,i)] << " " << 
      //              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
      //              endl;
  }

  clustered.convertTo(clustered, CV_8U);
  imshow("clustered", clustered);
  waitKey(0);

  return (clustered);
}

Mat SpotIt::processSubMat5(Mat& mat) {

  int centerX = mat.cols/2;
  
  //  cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  cv::Mat image = origImage.clone();

  //  vector<Mat> chan;
  //  cv::split(image, chan);
  //  cv::threshold(chan[0], chan[0], 200, 255, CV_THRESH_TOZERO);
  //  cv::threshold(chan[1], chan[1], 200, 255, CV_THRESH_TOZERO);
  //  cv::threshold(chan[2], chan[2], 200, 255, CV_THRESH_TOZERO);
  //  cv::merge(chan,image);
  //  cv::imshow("Filter Image", image);
  //  cv::waitKey(0);


  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);

  cv::threshold(image, image, 10, 255, CV_THRESH_TOZERO);
  // cv::threshold(image, image, 128, 255, CV_THRESH_BINARY);
  //cv::threshold(image, image, 160, 255, CV_THRESH_BINARY);
  cv::threshold(image, image, 190, 255, CV_THRESH_BINARY);

  // fastNlMeansDenoising(image, image, 3, 3, 7, 21 );
  //  fastNlMeansDenoising(image, image, 10, 19, 7, 21 );
  // fastNlMeansDenoising(image, image, 7, 3, 7, 21 );

  //  const int LAPLACIAN_FILTER_SIZE = 3;
  //  Laplacian(image, image, CV_8U, LAPLACIAN_FILTER_SIZE);

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  cv::findContours( contourOutput, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);
  //cout << "Contour Size: " << contours.size() << endl;
  for (size_t idx = 0; idx < contours.size(); idx++) {
    double area = contourArea(contours[idx]);
    if (area > 20) {
      // printf("Contour Area %f \n", contourArea(contours[idx]));
      cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
      //cv::Rect brect = cv::boundingRect(contours[idx]);
      //cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
    }  
  }

  cv::imshow("Input Image", image);
  cvMoveWindow("Input Image", 0, 0);
  cv::imshow("Contours", contourImage);
  cvMoveWindow("Contours", 600, 0);
  cv::waitKey(0);

  cv::Mat src = contourImage(cv::Range(0,image.rows), cv::Range(0,centerX));

  blur(src, src, Size(11,11));
  //blur(src, src, Size(13,13));
  //blur(src, src, Size(15,15));
  
  Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
  
  Mat bestLabels, centers, clustered;
  
  vector<Mat> bgr;
  
  cv::split(src, bgr);
  
  // i think there is a better way to split pixel bgr color
  for(int i=0; i<src.cols*src.rows; i++) {
      p.at<float>(i,0) = (i/src.cols) / src.rows;
      p.at<float>(i,1) = (i%src.cols) / src.cols;
      p.at<float>(i,2) = bgr[0].data[i] / 255.0;
      p.at<float>(i,3) = bgr[1].data[i] / 255.0;
      p.at<float>(i,4) = bgr[2].data[i] / 255.0;
  }

  int K = 10;
  
  cv::kmeans(p, K, bestLabels,
	     TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
	     3, KMEANS_PP_CENTERS, centers);
  
  int kmcolors[K];
  
  for(int i=0; i<K; i++) {
      kmcolors[i] = 255/(i+1);
  }
  
  // i think there is a better way to do this mayebe some Mat::reshape?
  clustered = Mat(src.rows, src.cols, CV_32F);
  
  for(int i=0; i<src.cols*src.rows; i++) {
      clustered.at<float>(i/src.cols, i%src.cols) = (float)(kmcolors[bestLabels.at<int>(0,i)]);
      //      cout << bestLabels.at<int>(0,i) << " " << 
      //              colors[bestLabels.at<int>(0,i)] << " " << 
      //              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
      //              endl;
  }

  clustered.convertTo(clustered, CV_8U);
  imshow("clustered", clustered);
  cvMoveWindow("clustered", 0, 200);

  //cout << "Clustered Image = "<< endl << " "  << clustered << endl << endl;

  vector<Mat> channels;
  // splitting.
  split(origImage, channels);

  Mat clusterImage = origImage.clone();

  for (int r = 0; r < clusterImage.rows; r += 1)
    for (int c = 0; c < clusterImage.cols; c += 1)
      {
	if (clustered.at<unsigned char>(r,c) == 255) {
	  clusterImage.at<Vec3b>(r,c)[0] = 255;
	  clusterImage.at<Vec3b>(r,c)[1] = 255;
	  clusterImage.at<Vec3b>(r,c)[2] = 255;
	}
      }
  imshow("Color Cluster Filter", clusterImage);
  cvMoveWindow("Color Cluster Filter", 600, 200);
  waitKey(0);

  //Prepare the image for findContours
  cv::cvtColor(clusterImage, clusterImage, CV_BGR2GRAY);
  cv::threshold(clusterImage, clusterImage, 200, 255, CV_THRESH_BINARY_INV);

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > fcontours;
  cv::Mat fcontourOutput = clusterImage.clone();
  cv::findContours( fcontourOutput, fcontours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

  //Draw the contours
  cv::Mat fcontourImage(clusterImage.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar fcolors[3];
  fcolors[0] = cv::Scalar(255, 0, 0);
  fcolors[1] = cv::Scalar(0, 255, 0);
  fcolors[2] = cv::Scalar(0, 0, 255);
  //cout << "Final Contour Size: " << fcontours.size() << endl;
  for (size_t idx = 0; idx < fcontours.size(); idx++) {
    double area = contourArea(fcontours[idx]);
    if (area > 30) {
      cv::drawContours(fcontourImage, fcontours, idx, fcolors[idx % 3]);
      cv::Rect fbrect = cv::boundingRect(fcontours[idx]);
      cv::rectangle(fcontourImage, fbrect.tl(), fbrect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
    }
  }

  cv::imshow("Final Input Image", clusterImage);
  cvMoveWindow("Final Input Image", 0, 300);
  cv::imshow("Final Contours", fcontourImage);
  cvMoveWindow("Final Contours", 600, 300);
  cv::waitKey(0);


  // channels[0].mul(clustered);
  // channels[1].mul(clustered);
  // channels[2].mul(clustered);
  // Mat mergeImage;
  // merge(channels,mergeImage);
  // imshow("Merge", mergeImage);

  return (clustered);
}


Mat SpotIt::processSubMat6(cv::Mat& mat) {
  cvNamedWindow("Control",CV_WINDOW_AUTOSIZE); //create a window called "Control"

  int iLowH = 0;
  int iHighH = 255;

  int iLowS = 0; 
  int iHighS = 255;

  int iLowV = 0;
  int iHighV = 255;

  //Create trackbars in "Control" window
  cvCreateTrackbar("LowH", "Control", &iLowH, 255);
  cvCreateTrackbar("HighH", "Control", &iHighH, 255);

  cvCreateTrackbar("LowS", "Control", &iLowS, 255);
  cvCreateTrackbar("HighS", "Control", &iHighS, 255);

  cvCreateTrackbar("LowV", "Control", &iLowV, 255);
  cvCreateTrackbar("HighV", "Control", &iHighV, 255);

  while (true)
    {
      Mat imgOriginal = mat;

      Mat imgHSV;

      cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
 
      Mat imgThresholded;

      inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
      
      dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_RECT, Size(3, 3)) ); //dilate the image to get rid of holes

      imshow("Thresholded Image", imgThresholded); //show the thresholded image
      imshow("Original", imgOriginal); //show the original image

      if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
	{
	  cout << "esc key is pressed by user" << endl;
	  break; 
	}
    }

  return mat;
}

Mat SpotIt::processSubMat7(Mat& mat) {

  int centerX = mat.cols/2;
  
  //cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  cv::Mat image = origImage.clone();

  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);
  blur( image, image, Size(3,3) );
  //cv::imshow("Blur Image", image);
  //cv::waitKey(0);

  //cv::threshold(image, image, 10, 255, CV_THRESH_TOZERO);
  // cv::threshold(image, image, 128, 255, CV_THRESH_BINARY);
  //cv::threshold(image, image, 160, 255, CV_THRESH_BINARY);
  //cv::threshold(image, image, 190, 255, CV_THRESH_BINARY);

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image, image, thresh, thresh*2, 3 );
  //cv::imshow("Canny Edge Image", image);
  //cv::waitKey(0);

  //blur( image, image, Size(3,3) );
  //cv::imshow("Canny Edge Blur Image", image);
  //cv::waitKey(0);

  //  const int LAPLACIAN_FILTER_SIZE = 3;
  //  Laplacian(image, image, CV_8U, LAPLACIAN_FILTER_SIZE);

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  vector<Vec4i> hierarchy;
  //  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );
  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  for (size_t idx = 0; idx < contours.size(); idx++) {
    double area = contourArea(contours[idx]);
    if (area > 1) {
      //printf("Contour Area %f \n", contourArea(contours[idx]));
      drawContours(contourImage, contours, idx, cv::Scalar(255,255,255),4,8);
      //cv::Rect brect = cv::boundingRect(contours[idx]);
      //cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
    }  
  }

  blur( contourImage, contourImage, Size(3,3) );
  std::vector<std::vector<cv::Point> > ccontours;
  cv::Mat contourOut = contourImage.clone();
  vector<Vec4i> chierarchy;
  cv::findContours( contourOut, ccontours, chierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  cv::Mat cImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  for (size_t idx = 0; idx < ccontours.size(); idx++) {
    //double area = contourArea(ccontours[idx]);
    //printf("Contour Area %f \n", area);
    drawContours(cImage, ccontours, idx, cv::Scalar(255,255,255),4,8);
    //cv::Rect brect = cv::boundingRect(ccontours[idx]);
    //cv::rectangle(cImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
  }  

  cv::imshow("Input Image", contourImage);
  cvMoveWindow("Input Image", 0, 0);
  cv::imshow("Contours", cImage);
  cvMoveWindow("Contours", 200, 0);
  cv::waitKey(0);

  RNG rng(12345);
  /// Get the moments
  vector<Moments> mu(ccontours.size() );
  for( int i = 0; i < ccontours.size(); i++ )
    { mu[i] = moments( ccontours[i], false ); }

  ///  Get the mass centers:
  vector<Point2f> mc( ccontours.size() );
  for( int i = 0; i < ccontours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

  /// Draw contours
  Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
  for( int i = 0; i< ccontours.size(); i++ )
    {
      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      drawContours( drawing, ccontours, i, color, 2, 8, hierarchy, 0, Point() );
      circle( drawing, mc[i], 4, color, -1, 8, 0 );
    }

  /// Show in a window
  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  cv::waitKey(0);

  drawing = Mat::zeros( image.size(), CV_8UC3 );

  /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
  printf("\t Info: Area and Contour Length \n");
  for( int i = 0; i< ccontours.size(); i++ )
    {
      //printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
      //printf("%d, %.2f, %.2f, %.2f, %i, %i, %i, %i \n", i, mu[i].m00, contourArea(contours[i]), arcLength(contours[i],true), hierarchy[i][0], hierarchy[i][0], hierarchy[i][1], hierarchy[i][2], hierarchy[i][3]);
      // hierarchy[idx][0] = next contour at the same hierarchical level
      // hierarchy[idx][1] = previous contour at the same hierarchical level
      // hierarchy[idx][2] = denotes its first child contour
      // hierarchy[idx][3] = denotes index of its parent contour

      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

      drawContours( drawing, ccontours, i, color, 3, 8, hierarchy, 3, Point() );
      //circle( drawing, mc[i], 4, color, -1, 8, 0 );
      std::string title = std::to_string(i);
      imshow( title, drawing );
      cv::waitKey(0);
      drawing = Mat::zeros( image.size(), CV_8UC3 );
    }

  imshow( "Contours", drawing );
  cv::waitKey(0);

  return (contourImage);
}

Mat SpotIt::processSubMat8(Mat& mat) {

  int centerX = mat.cols/2;
  
  cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  //cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  cv::Mat image = origImage.clone();

  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);
  cv::threshold(image, image, 190, 255, CV_THRESH_BINARY);
  blur( image, image, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image, image, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  vector<Vec4i> hierarchy;
  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  cout << "Contour Size: " << contours.size() << endl;
  if (!contours.empty() && !hierarchy.empty()) {
      // Loop through the contours/hierarchy
    for (size_t idx = 0; idx < contours.size(); idx++) {
      double area = contourArea(contours[idx]);
      // printf("%f, %d, %i, %i, %i, %i \n", area, idx, hierarchy[idx][0], hierarchy[idx][0], hierarchy[idx][1], hierarchy[idx][2], hierarchy[idx][3]);
      // hierarchy[idx][0] = next contour at the same hierarchical level
      // hierarchy[idx][1] = previous contour at the same hierarchical level
      // hierarchy[idx][2] = denotes its first child contour
      // hierarchy[idx][3] = denotes index of its parent contour

      if (area > 20) {
	// look for hierarchy[i][3]!=-1, ie hole boundaries
	if ( hierarchy[idx][3] != -1 ) {
	  cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
	  //cv::Rect brect = cv::boundingRect(contours[idx]);
	  //cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
	}
      }
    }  
  }

  cv::imshow("Input Image", image);
  cvMoveWindow("Input Image", 0, 0);
  cv::imshow("Contours", contourImage);
  cvMoveWindow("Contours", 200, 0);
  cv::waitKey(0);

  return (contourImage);
}

Mat SpotIt::processSubMat9(Mat& mat) {

  int centerX = mat.cols/2;
  
  //cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(0,centerX));
  cv::Mat origImage = mat(cv::Range(0,mat.rows), cv::Range(centerX,mat.cols));

  cv::Mat image = origImage.clone();

  //Prepare the image for findContours
  cv::cvtColor(image, image, CV_BGR2GRAY);
  blur( image, image, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image, image, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  vector<Vec4i> hierarchy;
  //  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0) );
  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  for (size_t idx = 0; idx < contours.size(); idx++) {
    double area = contourArea(contours[idx]);
    if (area > 1) {
      //printf("Contour Area %f \n", contourArea(contours[idx]));
      drawContours(contourImage, contours, idx, cv::Scalar(255,255,255),4,8);
      //cv::Rect brect = cv::boundingRect(contours[idx]);
      //cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
    }  
  }

  return (contourImage);
}

Mat SpotIt::processSubMat10(Mat& mat) {

  cv::Mat image = mat.clone();

  //Prepare the image for findContours
  blur( image, image, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image, image, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours;
  cv::Mat contourOutput = image.clone();
  vector<Vec4i> hierarchy;
  cv::findContours( contourOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  cout << "Contour Size: " << contours.size() << endl;

  vector<Moments> mu( contours.size() );
  vector<Point2f> mc( contours.size() );
  double hu [contours.size()][7];
  //double hu[1000][7];
  double im [contours.size()][3];

  if (!contours.empty() && !hierarchy.empty()) {
      // Loop through the contours/hierarchy
    for (int i = 0; i < contours.size(); i++) {
      // FIXME should the binaryImage flag be true or false for contours? For now set to true!
      // Get the moments
      mu[i] = moments( contours[i], true );
      //  Get the mass centers:
      mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
      HuMoments(mu[i], hu[i]);
      getImoments(hu[i], im[i]);
      
      if (hierarchy[i][3] == -1) {
	printf("%i, %i, %i, %i, %i, ", i, hierarchy[i][0], hierarchy[i][1], hierarchy[i][2], hierarchy[i][3]);
	printf("%f, %f, %f, %f, %f, %f, %f, ", hu[i][0],hu[i][1],hu[i][2],hu[i][3],hu[i][4],hu[i][5],hu[i][6]);
	printf("%f, %f \n", im[i][0],im[i][1]);
	// hierarchy[i][0] = next contour at the same hierarchical level
	// hierarchy[i][1] = previous contour at the same hierarchical level
	// hierarchy[i][2] = denotes its first child contour
	// hierarchy[i][3] = denotes index of its parent contour

	cv::drawContours(contourImage, contours, i, colors[i % 3]);
	//cv::Rect brect = cv::boundingRect(contours[i]);
	//cv::rectangle(contourImage, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
	std::string title = std::to_string(i);
	imshow( title, contourImage );
	cv::waitKey(0);
	contourImage = Mat::zeros( contourImage.size(), CV_8UC3 );
      }
    }  
  }

  cv::imshow("Input Image", image);
  cvMoveWindow("Input Image", 0, 0);
  cv::imshow("Contours", contourImage);
  cvMoveWindow("Contours", 200, 0);
  cv::waitKey(0);

  return (contourImage);
}



Mat SpotIt::processSubMat11(Mat& mat) {

  Mat image;

  if ((mat.rows > 1024) || (mat.cols > 1024)) {
    image.create(512,1024, CV_8UC3);
    cv::resize(mat, image, image.size(), 0, 0, CV_INTER_LINEAR);
  } else
    image = mat.clone();

  int centerX = image.cols/2;
  
  cv::Mat image1 = image(cv::Range(0,image.rows), cv::Range(0,centerX));
  cv::Mat image2 = image(cv::Range(0,image.rows), cv::Range(centerX,image.cols));

  image1 = refineObjects(image1);
  image2 = refineObjects(image2);

  //Prepare the image for findContours
  blur( image1, image1, Size(3,3) );
  blur( image2, image2, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image1, image1, thresh, thresh*2, 3 );
  Canny( image2, image2, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours1;
  std::vector<std::vector<cv::Point> > contours2;
  vector<Vec4i> hierarchy1;
  vector<Vec4i> hierarchy2;
  cv::Mat contourOutput1 = image1.clone();
  cv::Mat contourOutput2 = image2.clone();
  cv::findContours( contourOutput1, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));
  cv::findContours( contourOutput2, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage1(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat contourImage2(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  //cout << "Contour Size: " << contours1.size() << endl;

  double im1 [contours1.size()][3];
  double im2 [contours2.size()][3];

  if (!contours1.empty() && !hierarchy1.empty()) {
      // Loop through the contours/hierarchy
    for (int i = 0; i < contours1.size(); i++) {
      if (hierarchy1[i][3] == -1) {
	findInvariantMoments(contours1[i], im1[i]);
	//printf("%i, %i, %i, %i, %i, ", i, hierarchy1[i][0], hierarchy1[i][1], hierarchy1[i][2], hierarchy1[i][3]);
	//printf("%f, %f, %f \n", im1[i][0],im1[i][1],im1[i][2]);
	//cv::drawContours(contourImage1, contours1, i, colors[i % 3]);
	//cv::Rect brect = cv::boundingRect(contours1[i]);
	//cv::rectangle(contourImage1, brect.tl(), brect.br(), cv::Scalar(100, 100, 200), 2, CV_AA);
	//std::string title = std::to_string(i);
	//imshow( title, contourImage1 );
	//cv::waitKey(0);
	//contourImage1 = Mat::zeros( contourImage1.size(), CV_8UC3 );
      }
    }  
  }

  if (!contours2.empty() && !hierarchy2.empty()) {
      // Loop through the contours/hierarchy
    for (int i = 0; i < contours2.size(); i++) {
      if (hierarchy2[i][3] == -1) {
	findInvariantMoments(contours2[i], im2[i]);
	// cv::drawContours(contourImage2, contours2, i, colors[i % 3]);
      }
    }  
  }

  int image1Sel[2], image2Sel[2];
  double score[2];
  score[0] = 0;
  score[1] = 0;
  for (int i = 0; i < contours1.size(); i++) {
    for (int j = 0; j < contours2.size(); j++) {
      if (hierarchy1[i][3] == -1) {
	for (int k=0; k<2; k++) {
	  if (!score[k] || fabs (im1[i][k]-im2[j][k]) < score[k]) {
	    image1Sel[k] = i;
	    image2Sel[k] = j;
	    score[k] = fabs (im1[i][k]-im2[j][k]);
	    //printf("i=%i, j=%i, k=%i, im1[i][k]=%f, im2[j][k]=%f, score[k]=%f\n", i,j,k,im1[i][k],im2[j][k],score[k]);
	  }
	}
      }
    }
  }

  if (score[1] > score[0]) {
    cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );
  } else {
    cv::drawContours(contourImage1, contours1, image1Sel[1], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[1], colors[1], 2, 8, hierarchy2, 0, Point() );
  }

    cv::imshow("Input Image1", image1);
    cv::imshow("Input Image2", image2);
    cvMoveWindow("Input Image1", 0, 0);
    cvMoveWindow("Input Image2", 0, 0);
  cv::imshow("Contours1", contourImage1);
  cv::imshow("Contours2", contourImage2);
  cvMoveWindow("Contours1", 0, 0);
  cvMoveWindow("Contours2", 200, 0);
  cv::waitKey(0);

  return (contourImage2);
}

Mat SpotIt::processSubMat12(Mat& mat) {

  Mat image;

  if ((mat.rows > 1024) || (mat.cols > 1024)) {
    image.create(512,1024, CV_8UC3);
    cv::resize(mat, image, image.size(), 0, 0, CV_INTER_LINEAR);
  } else
    image = mat.clone();

  int centerX = image.cols/2;
  
  cv::Mat image1 = image(cv::Range(0,image.rows), cv::Range(0,centerX));
  cv::Mat image2 = image(cv::Range(0,image.rows), cv::Range(centerX,image.cols));

  image1 = refineObjects(image1);
  image2 = refineObjects(image2);

  //Prepare the image for findContours
  blur( image1, image1, Size(3,3) );
  blur( image2, image2, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image1, image1, thresh, thresh*2, 3 );
  Canny( image2, image2, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours1;
  std::vector<std::vector<cv::Point> > contours2;
  vector<Vec4i> hierarchy1;
  vector<Vec4i> hierarchy2;
  cv::Mat contourOutput1 = image1.clone();
  cv::Mat contourOutput2 = image2.clone();
  cv::findContours( contourOutput1, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));
  cv::findContours( contourOutput2, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage1(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat contourImage2(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  //cout << "Contour Size: " << contours1.size() << endl;

  int image1Sel[3], image2Sel[3];
  double score[3];
  double tempScore = 0;
  int method[3];
  score[0] = 0;
  score[1] = 0;
  score[2] = 0;
  method[0] = CV_CONTOURS_MATCH_I1;
  method[1] = CV_CONTOURS_MATCH_I2;
  method[2] = CV_CONTOURS_MATCH_I3;

  // Loop through the contours/hierarchy
  for (int i = 0; i < contours1.size(); i++) {
    if (hierarchy1[i][3] == -1) {
      cv::drawContours(contourImage1, contours1, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage1 );
      cv::waitKey(0);
      contourImage1 = Mat::zeros( contourImage1.size(), CV_8UC3 );
    }
  }  
  // Loop through the contours/hierarchy
  for (int i = 0; i < contours2.size(); i++) {
    if (hierarchy2[i][3] == -1) {
      cv::drawContours(contourImage2, contours2, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage2 );
      cv::waitKey(0);
      contourImage2 = Mat::zeros( contourImage2.size(), CV_8UC3 );
    }
  }

  for (int i = 0; i < contours1.size(); i++) {
    for (int j = 0; j < contours2.size(); j++) {
      if ( (hierarchy1[i][3] == -1) && (hierarchy2[j][3] == -1)) {
	for (int k=0; k<3; k++) {
	  tempScore = cv::matchShapes(contours1[i], contours2[j], method[k], 0);
	  printf("i=%i, j=%i, k=%i, tempScore=%f,score[k]=%f\n", i,j,k,tempScore,score[k]);
	  if ((score[k] == 0) || tempScore < score[k]) {
	    image1Sel[k] = i;
	    image2Sel[k] = j;
	    score[k] = tempScore;
	    printf("i=%i, j=%i, k=%i, score[k]=%f\n", i,j,k,score[k]);
	  }
	}
      }
    }
  }

#if 0
  if ((score[1] > score[0]) && (score[2] > score[0])) {
    cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );
  } else if (score[2] > score[1]) {
    cv::drawContours(contourImage1, contours1, image1Sel[1], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[1], colors[1], 2, 8, hierarchy2, 0, Point() );
  } else {
    cv::drawContours(contourImage1, contours1, image1Sel[2], colors[2], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[2], colors[2], 2, 8, hierarchy2, 0, Point() );
  }
#else
    cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );
#endif

  cv::imshow("Input Image1", image1);
  cv::imshow("Input Image2", image2);
  cvMoveWindow("Input Image1", 0, 0);
  cvMoveWindow("Input Image2", 0, 0);
  cv::imshow("Contours1", contourImage1);
  cv::imshow("Contours2", contourImage2);
  cvMoveWindow("Contours1", 0, 0);
  cvMoveWindow("Contours2", 200, 0);
  cv::waitKey(0);

  return (contourImage2);
}

Mat SpotIt::processSubMat13(Mat& mat) {

  //  Mat image;

  //  if ((mat.rows > 1024) || (mat.cols > 1024)) {
  //    image.create(512,1024, CV_8UC3);
  //    cv::resize(mat, image, image.size(), 0, 0, CV_INTER_LINEAR);
  //  } else
  //    image = mat.clone();

  //  int centerX = image.cols/2;
  
  //  cv::Mat image1 = image(cv::Range(0,image.rows), cv::Range(0,centerX));
  //  cv::Mat image2 = image(cv::Range(0,image.rows), cv::Range(centerX,image.cols));
  Mat image1 = imread("../spot-it-small-clean-left-target.jpg");
  Mat image2 = imread("../spot-it-small-clean-right-target.jpg");

  image1 = refineObjects(image1);
  image2 = refineObjects(image2);

  //Prepare the image for findContours
  blur( image1, image1, Size(3,3) );
  blur( image2, image2, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image1, image1, thresh, thresh*2, 3 );
  Canny( image2, image2, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours1;
  std::vector<std::vector<cv::Point> > contours2;
  vector<Vec4i> hierarchy1;
  vector<Vec4i> hierarchy2;
  cv::Mat contourOutput1 = image1.clone();
  cv::Mat contourOutput2 = image2.clone();
  cv::findContours( contourOutput1, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));
  cv::findContours( contourOutput2, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage1(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat contourImage2(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  //cout << "Contour Size: " << contours1.size() << endl;

  int image1Sel[3], image2Sel[3];
  double score[3];
  double tempScore = 0;
  int method[3];
  score[0] = 0;
  score[1] = 0;
  score[2] = 0;
  method[0] = CV_CONTOURS_MATCH_I1;
  method[1] = CV_CONTOURS_MATCH_I2;
  method[2] = CV_CONTOURS_MATCH_I3;

#if 0
  // Loop through the contours/hierarchy
  for (int i = 0; i < contours1.size(); i++) {
    if (hierarchy1[i][3] == -1) {
      cv::drawContours(contourImage1, contours1, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage1 );
      cv::waitKey(0);
      contourImage1 = Mat::zeros( contourImage1.size(), CV_8UC3 );
    }
  }  
  // Loop through the contours/hierarchy
  for (int i = 0; i < contours2.size(); i++) {
    if (hierarchy2[i][3] == -1) {
      cv::drawContours(contourImage2, contours2, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage2 );
      cv::waitKey(0);
      contourImage2 = Mat::zeros( contourImage2.size(), CV_8UC3 );
    }
  }
#endif

  for (int i = 0; i < contours1.size(); i++) {
    for (int j = 0; j < contours2.size(); j++) {
      if ( (hierarchy1[i][3] == -1) && (hierarchy2[j][3] == -1)) {
	for (int k=0; k<3; k++) {
	  tempScore = cv::matchShapes(contours1[i], contours2[j], method[k], 0);
	  printf("i=%i, j=%i, k=%i, tempScore=%f,score[k]=%f\n", i,j,k,tempScore,score[k]);
	  if ((score[k] == 0) || tempScore < score[k]) {
	    image1Sel[k] = i;
	    image2Sel[k] = j;
	    score[k] = tempScore;
	    printf("i=%i, j=%i, k=%i, score[k]=%f\n", i,j,k,score[k]);
	  }
	}
      }
    }
  }

  cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
  cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );

  cv::imshow("Input Image1", image1);
  cv::imshow("Input Image2", image2);
  cvMoveWindow("Input Image1", 0, 0);
  cvMoveWindow("Input Image2", 0, 0);
  cv::imshow("Contours1", contourImage1);
  cv::imshow("Contours2", contourImage2);
  cvMoveWindow("Contours1", 0, 0);
  cvMoveWindow("Contours2", 200, 0);
  cv::waitKey(0);

  return (contourImage2);
}

Mat SpotIt::processSubMat14(Mat& mat) {

  Mat image;

  if ((mat.rows > 1024) || (mat.cols > 1024)) {
    image.create(512,1024, CV_8UC3);
    cv::resize(mat, image, image.size(), 0, 0, CV_INTER_LINEAR);
  } else
    image = mat.clone();

  int centerX = image.cols/2;
  
  cv::Mat image1 = image(cv::Range(0,image.rows), cv::Range(0,centerX));
  cv::Mat image2 = image(cv::Range(0,image.rows), cv::Range(centerX,image.cols));

  image1 = refineObjects(image1);
  image2 = refineObjects(image2);

  //Prepare the image for findContours
  blur( image1, image1, Size(3,3) );
  blur( image2, image2, Size(3,3) );

  /// Detect edges using canny
  int thresh = 80; // max 255
  Canny( image1, image1, thresh, thresh*2, 3 );
  Canny( image2, image2, thresh, thresh*2, 3 );

  //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
  std::vector<std::vector<cv::Point> > contours1;
  std::vector<std::vector<cv::Point> > contours2;
  vector<Vec4i> hierarchy1;
  vector<Vec4i> hierarchy2;
  cv::Mat contourOutput1 = image1.clone();
  cv::Mat contourOutput2 = image2.clone();
  cv::findContours( contourOutput1, contours1, hierarchy1, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));
  cv::findContours( contourOutput2, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0,0));

  //Draw the contours
  cv::Mat contourImage1(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Mat contourImage2(image1.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::Scalar colors[3];
  colors[0] = cv::Scalar(255, 0, 0);
  colors[1] = cv::Scalar(0, 255, 0);
  colors[2] = cv::Scalar(0, 0, 255);

  //cout << "Contour Size: " << contours1.size() << endl;

  int image1Sel[3], image2Sel[3];
  double score[3];
  double tempScore = 0;
  int method[3];
  score[0] = 0;
  score[1] = 0;
  score[2] = 0;
  method[0] = CV_CONTOURS_MATCH_I1;
  method[1] = CV_CONTOURS_MATCH_I2;
  method[2] = CV_CONTOURS_MATCH_I3;

#if 0
  // Loop through the contours/hierarchy
  for (int i = 0; i < contours1.size(); i++) {
    if (hierarchy1[i][3] == -1) {
      vector<Point> approxShape1;
      double length1 = cv::arcLength(contours1[i], false);
      cv::approxPolyDP(contours1[i],approxShape1, 0.01*length1, false);
      double approxLength1 = cv::arcLength(approxShape1, false);
      printf("Length1 %f - %f\n", length1, approxLength1);
      cv::drawContours(contourImage1, contours1, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage1 );
      cv::waitKey(0);
      contourImage1 = Mat::zeros( contourImage1.size(), CV_8UC3 );
    }
  }  
  // Loop through the contours/hierarchy
  for (int i = 0; i < contours2.size(); i++) {
    if (hierarchy2[i][3] == -1) {
      vector<Point> approxShape2;
      double length2 = cv::arcLength(contours2[i], false);
      cv::approxPolyDP(contours2[i],approxShape2, 0.01*length2, false);
      double approxLength2 = cv::arcLength(approxShape2, false);
      printf("Length2 %f - %f\n", length2, approxLength2);
      cv::drawContours(contourImage2, contours2, i, colors[i % 3]);
      std::string title = std::to_string(i);
      imshow( title, contourImage2 );
      cv::waitKey(0);
      contourImage2 = Mat::zeros( contourImage2.size(), CV_8UC3 );
    }
  }
#endif

  for (int i = 0; i < contours1.size(); i++) {
    for (int j = 0; j < contours2.size(); j++) {
      if ( (hierarchy1[i][3] == -1) && (hierarchy2[j][3] == -1)) {
	for (int k=0; k<3; k++) {
	  tempScore = cv::matchShapes(contours1[i], contours2[j], method[k], 0);
	  //printf("i=%i, j=%i, k=%i, tempScore=%f,score[k]=%f\n", i,j,k,tempScore,score[k]);
	  if ((score[k] == 0) || tempScore < score[k]) {
	    image1Sel[k] = i;
	    image2Sel[k] = j;
	    score[k] = tempScore;
	    //printf("i=%i, j=%i, k=%i, score[k]=%f\n", i,j,k,score[k]);
	  }
	}
      }
    }
  }

#if 0
  if ((score[1] > score[0]) && (score[2] > score[0])) {
    cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );
  } else if (score[2] > score[1]) {
    cv::drawContours(contourImage1, contours1, image1Sel[1], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[1], colors[1], 2, 8, hierarchy2, 0, Point() );
  } else {
    cv::drawContours(contourImage1, contours1, image1Sel[2], colors[2], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[2], colors[2], 2, 8, hierarchy2, 0, Point() );
  }
#else
    cv::drawContours(contourImage1, contours1, image1Sel[0], colors[1], 2, 8, hierarchy1, 0, Point() );
    cv::drawContours(contourImage2, contours2, image2Sel[0], colors[1], 2, 8, hierarchy2, 0, Point() );
#endif

  cv::imshow("Input Image1", image1);
  cv::imshow("Input Image2", image2);
  cvMoveWindow("Input Image1", 0, 0);
  cvMoveWindow("Input Image2", 600, 0);
  cv::waitKey(0);
  cv::imshow("Match1", contourImage1);
  cv::imshow("Match2", contourImage2);
  cvMoveWindow("Match1", 200, 200);
  cvMoveWindow("Match2", 800, 200);
  cv::waitKey(0);

  return (contourImage2);
}
