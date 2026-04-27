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

    float totalIoU = 0;
    int totalCategories = 0;   //counter for the directories in dataSet
    int correctDetections = 0;  

    for (const fs::directory_entry& entry : fs::directory_iterator(dataPath)) {

        if (!entry.is_directory()) {
            continue; //without this was giving error on MacOS due to the .DS_store file
        }
        string category = entry.path().filename().string();
        cout << "object detected: " << category << endl;

        vector<Mat> sequence = Loader::getImageSequence(entry.path().string()); 

        if (sequence.empty()) {
            cout << "No valid images found" << endl;
            continue;
        }
        
        Rect detectedRect = detector.detectMovingObject(sequence);

    
        vector<string> groundTruthPath = Loader::getLabelPath(labelsPath, category);
        float maxIoU = 0;
        Rect groundTruthRect;
        
        /*if there are different ground truth files, like in the squirrel case we need to consider the 
        ground truth values with the highest IoU */
        for (const string& path : groundTruthPath) {
             Rect currentGroundTruth = utils.readGroundTruth(path);
             float currentIoU = utils.computeIoU(detectedRect, currentGroundTruth);
    
                if (currentIoU >= maxIoU) {
                maxIoU = currentIoU;
                groundTruthRect = currentGroundTruth;
                    }               
        }
        
        float currentIoU = maxIoU;
            cout << "IoU: " << (currentIoU * 100) << "%" << endl;
            if (currentIoU > 0.5) {
            correctDetections++;
            }
        int xmin = detectedRect.x;
        int ymin = detectedRect.y;
        int xmax = detectedRect.x + detectedRect.width;
        int ymax = detectedRect.y + detectedRect.height;
        outFile << xmin << " " << ymin << " " << xmax << " " << ymax << endl; 

        totalIoU = totalIoU + currentIoU;
        totalCategories++;

        Mat showImage = sequence[0].clone(); //we show in output the rectangles
                                            //on the first frame of the sequence

        rectangle(showImage, detectedRect, Scalar(0, 255, 0), 2); // green for the prediction
        rectangle(showImage, groundTruthRect, Scalar(0, 0, 255), 1); //red fot the ground truth

        imshow("Result: " + category, showImage);
        
        cout << "Give any input for next category..." << endl;
        waitKey(0); 
        destroyAllWindows(); 
    }

    outFile.close();

    if (totalCategories == 0) {                     //control if there is no input, because
        cout << "No categories processed." << endl; //can't divide by zero for the median IoU
        return 0;
    }

    float mIoU = (totalIoU / totalCategories) * 100;
    float detectionAccuracy = (float(correctDetections)/ totalCategories) * 100; //without cast was always 0 

    cout << "median IoU: " << mIoU << "%" << endl;
    cout << "Detection Accuracy: " << detectionAccuracy << "% (" << correctDetections << "/" << totalCategories << ")" << endl;
    cout << "Values saved to: " << outputPath << endl;

    return 0;
}