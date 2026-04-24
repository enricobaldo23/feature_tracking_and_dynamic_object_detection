#include "motion.hpp"
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>


using namespace std;
using namespace cv;

Rect motion::detectMovingObject(vector<Mat> frames) {
    if (frames.size() < 2) return Rect();

    Mat prevGray;
    cvtColor(frames[0], prevGray, COLOR_BGR2GRAY);

    // 1. Detect interesting points on the first frame
    vector<Point2f> initialPoints;
    goodFeaturesToTrack(prevGray, initialPoints, 500, 0.01, 10);

    if (initialPoints.empty()) return Rect();

    vector<Point2f> currentPoints = initialPoints;
    vector<float> totalMovement(initialPoints.size(), 0.0f);
    vector<bool> pointIsLost(initialPoints.size(), false);

    // 2. Track points across the sequence using Lucas-Kanade
    for (int i = 1; i < frames.size(); i++) {
        Mat currentGray;
        cvtColor(frames[i], currentGray, COLOR_BGR2GRAY);

        vector<Point2f> nextPoints;
        vector<uchar> status;
        vector<float> err;

        calcOpticalFlowPyrLK(prevGray, currentGray, currentPoints, nextPoints, status, err);

        for (int k = 0; k < currentPoints.size(); k++) {
            if (status[k] == 1 && !pointIsLost[k]) {
                // Measure how much the point moved in this step
                float stepDist = norm(nextPoints[k] - currentPoints[k]);
                totalMovement[k] += stepDist;
                
                // Update position for the next frame
                currentPoints[k] = nextPoints[k];
            } else {
                pointIsLost[k] = true; 
            }
        }
        prevGray = currentGray.clone();
    }

    // 3. Filter points that actually moved significantly
    vector<Point2f> movingPoints;
    float movementThreshold = 15.0f; 

    for (int j = 0; j < initialPoints.size(); j++) {
        if (!pointIsLost[j] && totalMovement[j] > movementThreshold) {
            // Use the original coordinates from the FIRST frame
            movingPoints.push_back(initialPoints[j]); 
        }
    }

    if (movingPoints.empty()) return Rect();

    // Return the box containing all moving points
    return boundingRect(movingPoints);
}