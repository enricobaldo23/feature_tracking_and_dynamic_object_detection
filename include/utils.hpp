#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Utils {
public:
    cv::Rect readGroundTruth(std::string path);

    float computeIoU(cv::Rect prediction, cv::Rect groundTruth);
};
#endif