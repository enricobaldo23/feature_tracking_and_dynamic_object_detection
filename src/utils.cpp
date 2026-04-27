#include "utils.hpp"
#include <fstream>  
#include <iostream>

using namespace std;
using namespace cv;

Rect Utils::readGroundTruth(string filePath) {
    
    ifstream file(filePath);
    
    if (!file.is_open()) {
        return Rect(0, 0, 0, 0); 
    }

    
    int xmin, ymin, xmax, ymax;
    
 
    if (!(file >> xmin >> ymin >> xmax >> ymax)) {
        file.close();
        return Rect(0, 0, 0, 0);
    }
    file.close();

    Point p1 = Point(xmin, ymin);
    Point p2 = Point(xmax, ymax);
    return Rect(p1, p2);
}

float Utils::computeIoU(Rect prediction, Rect groundTruth) {

    if (prediction.area() == 0 || groundTruth.area() == 0) { 
        return 0; //if one of the rectangles is empty, IoU must be 0
    }

    Rect intersectionRect = prediction & groundTruth; //the and operator calculates the intersection

    float intersectionValue = intersectionRect.area(); 
       /*intersectionValue contains the area in common between our prediction rectangle and the
    ground truth rectangle */
    
    float totalArea = prediction.area() + groundTruth.area() - intersectionValue;
    /*totalArea is the area covered by both rectangles, calculated as the sum of their areas,
    minus the intersection to avoid double counting.*/
    
    if (totalArea == 0) {
        return 0; 
    }
    return intersectionValue/totalArea; //IoU
}



