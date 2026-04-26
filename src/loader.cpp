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
        string extension = file.path().extension().string();

        if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
            imgFiles.push_back(file.path().string()); //we want only images
        }
    }
    sort(imgFiles.begin(), imgFiles.end());   //had some issues with optical flow if images weren't ordered

    for (const string& p : imgFiles) {
        Mat f = imread(p);
        if (!f.empty()) sequence.push_back(f); 
    }

    return sequence;
}

string Loader::getLabelPath(const string& labelsPath, const string& type) { //reads labels
 
    string createLabelPath = labelsPath + "/" + type;

    if (fs::exists(createLabelPath)){
    for (const fs::directory_entry& f : fs::directory_iterator(createLabelPath)) {

            if (f.path().extension() == ".txt"){
                return f.path().string(); 
            } 
        }
    }
    return "";
}

   
