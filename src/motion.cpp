#include "motion.hpp"
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

Rect MotionDetector::detectMovingObject(vector<Mat> frames) {

    if (frames.size() < 2) {
        return Rect(0, 0, 0, 0); //if we have only one frame we can't detect motion
    }

    Mat grayFirst, grayMid; //we consider the first and middle frame

    cvtColor(frames.front(), grayFirst, COLOR_BGR2GRAY);       //convert in grayscale
    cvtColor(frames[frames.size()/2], grayMid, COLOR_BGR2GRAY);
    

    vector<Point2f> pointsFirst, pointsMid; //contain points tracked in the frames we consider

    goodFeaturesToTrack(grayFirst, pointsFirst, 1500, 0.005, 3, Mat(), 3, true, 0.03); //Shi-Tomasi corner detection in first frame

     if (pointsFirst.empty()) { //no points tracked
        return Rect(0, 0, 0, 0);
    }

    //now calculate optical flow 

    vector<uchar> status; 
    vector<float> err; 
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(grayFirst, grayMid, pointsFirst, pointsMid, status, err,Size(20, 20), 3,criteria);

    vector<Point2f> movingPoints; 

    for (int i = 0; i < pointsFirst.size(); i++) {           
        if (status[i]) {                                        //for the points tracked, we check how mich they moved
            float normal = norm(pointsMid[i] - pointsFirst[i]);
            
            if (normal > 5.5 && normal < 40) { //if the movement is between 5.5 and 40 pixels, we consider it as motion
                movingPoints.push_back(pointsFirst[i]);
            }
        }
    }

    if (movingPoints.size() < 5) { //5 is our lowerbound for motion detection
        return Rect(0, 0, 0, 0); 
    }  

return boundingRect(movingPoints);}
