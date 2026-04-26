#ifndef MOTION_HPP
#define MOTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class MotionDetector {
public:
    cv::Rect detectMovingObject(std::vector<cv::Mat> frames);
};
#endif