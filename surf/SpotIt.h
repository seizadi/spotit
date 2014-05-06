#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

struct Features {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

class SpotIt {
public:
    cv::Mat processMat(cv::Mat& mat);
    
private:
    Features processSubMat(cv::Mat& mat);
};
