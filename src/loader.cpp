#include "loader.hpp"
#include <algorithm>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

vector<Mat> Loader::getImageSequence(const string& dataPath) { //reads dataset
    vector<Mat> sequence; 

    if (!fs::exists(dataPath)) { 
        return sequence; //firstly, control if path exists
    }

    vector<string> imgFiles; 
    for (const fs::directory_entry& file : fs::directory_iterator(dataPath)) {
        
        /*firstly, need to control if the file is an image or there would 
        be error when reading the sequence*/

    string extension = file.path().extension().string();
        if (extension == ".jpg" || extension == ".png") {
            imgFiles.push_back(file.path().string()); 
        }
    }
    sort(imgFiles.begin(), imgFiles.end());   //had some issues with optical flow if images weren't ordered

    for (const string& p : imgFiles) {
        Mat f = imread(p);
        if (!f.empty()) sequence.push_back(f); 
    }
    
    return sequence;
}

vector<string> Loader::getLabelPath(const string& labelsPath, const string& type) { 
    /*had troubles with the double directories and ground truth for the squirrel, 
    with only strings, so used vector of strings and the recursive directory iterator from
    filesystem library*/
    
    vector<string> labels;
    string createLabelPath = labelsPath + "/" + type;

    if (fs::exists(createLabelPath)) {
        for (const fs::directory_entry& entry : fs::recursive_directory_iterator(createLabelPath)) {
            if (entry.path().extension() == ".txt") {
                labels.push_back(entry.path().string());
            }
        }
    }
    return labels;
}

   
