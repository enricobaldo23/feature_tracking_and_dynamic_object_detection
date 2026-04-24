#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class Utils {
public:
    Rect groundTruth(string path);
    float computeIoU(Rect pred_box, Rect gt_box);
};

#endif