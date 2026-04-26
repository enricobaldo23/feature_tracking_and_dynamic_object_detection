#include <iostream>
#include <fstream> 
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include "motion.hpp"
#include "utils.hpp"
#include "loader.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main() {
    string dataPath = "../data";
    string labelsPath = "../labels";
    string outputPath = "../detections.txt"; 

    if (!fs::exists(dataPath)) {
        cerr << "Error: data not found!" << endl;
        return -1;
    }
    if (!fs::exists(labelsPath)) {
        cerr << "Error: labels not found!" << endl;
        return -1;
    }

 
    ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        cerr << "Could not create output file!" << endl;
        return -1;
    }

    MotionDetector detector;
    Utils utils;

    float totalIoU = 0.0f;
    int totalCategories = 0; //counter for directories in dataSet
    int correctDetections = 0;  

    for (const fs::directory_entry& entry : fs::directory_iterator(dataPath)) {
        if (!entry.is_directory()) continue; //without this, on MacOS error due to the .DS_store file

        string category = entry.path().filename().string();
        cout << "object detected: " << category << endl;

        vector<Mat> sequence = Loader::getImageSequence(entry.path().string()); 

        if (sequence.empty()) {
            cout << "No valid images found" << endl;
            continue;
        }
        
        Rect detectedRect = detector.detectMovingObject(sequence);

    
        string groundTruthPath = Loader::getLabelPath(labelsPath, category);
        Rect groundTruthRect = utils.readGroundTruth(groundTruthPath);
        
        float currentIoU = utils.computeIoU(detectedRect, groundTruthRect);
            cout << "IoU corrente: " << (currentIoU * 100.0f) << "%" << endl;
            if (currentIoU > 0.5f) {
            correctDetections++;
            }
        int xmin = detectedRect.x;
        int ymin = detectedRect.y;
        int xmax = detectedRect.x + detectedRect.width;
        int ymax = detectedRect.y + detectedRect.height;
        outFile << xmin << " " << ymin << " " << xmax << " " << ymax << endl; 

        totalIoU = totalIoU + currentIoU;
        totalCategories++;

        Mat showImage = sequence[0].clone(); //we show the rectangles on the first frame of the sequence

        rectangle(showImage, detectedRect, Scalar(0, 255, 0), 2); // green for the prediction
        rectangle(showImage, groundTruthRect, Scalar(0, 0, 255), 1); //red fot the ground truth

        imshow("Result: " + category, showImage);
        
        cout << "  (Press any key for next category...)" << endl;
        waitKey(0); 
        destroyAllWindows(); 
    }

    outFile.close();

    if (totalCategories == 0) {
        cout << "No categories processed." << endl; //can't divide by zero for the median IoU
        return 0;
    }

    float mIoU = (totalIoU / totalCategories) * 100.0f;
    float detectionAccuracy = (static_cast<float>(correctDetections) / totalCategories) * 100.0f;

    cout << "mIoU: " << mIoU << "%" << endl;
    cout << "Detection Accuracy: " << detectionAccuracy << "% (" << correctDetections << "/" << totalCategories << ")" << endl;
    cout << "Coordinates saved to: " << outputPath << endl;

    return 0;
}