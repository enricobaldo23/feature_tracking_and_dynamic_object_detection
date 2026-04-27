#ifndef LOADER_HPP
#define LOADER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

class Loader{
public: 
    static std::vector<cv::Mat> getImageSequence(const std::string& dataPath);

    static std::vector<std::string> getLabelPath(const std::string& labelsPath, const std::string& type);
};
#endif