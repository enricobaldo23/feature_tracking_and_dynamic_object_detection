#include "utils.hpp"
using namespace std;
using namespace cv;


Rect Utils::readGroundTruth(string path)
{   int xmin, int ymin, int xmax, int ymax;
    ifstream file(path);

    if (!file.is_open()) {                                  //if we can't open the file, we return an empty rectangle
        cerr << "Error opening file: " << path << endl;
        return Rect(0,0,0,0);
    }
    file >> xmin >> ymin >> xmax >> ymax;
    file.close();
    
    Point p_1(xmin, ymin);
    Point p_2(xmax, ymax);
    Rect r(p_1, p_2);
    return r;
}


float Utils::computeIoU(cv::Rect pred_box, cv::Rect truth_box) 
{
    cv::Rect intersection = pred_box & truth_box; //compute intersection

    float intersection_area = intersection.area(); 
 
    float union_area = pred_box.area() + truth_box.area() - intersection_area; //need to remove intersection in the sum
    if(union_area==0){
        return 0; //can't divide by zero
    } 
    return intersection_area / union_area;
}




