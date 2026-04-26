# feature_tracking_and_dynamic_object_detection
This project, receives as an input different sequences of frames, and predicts correctly the position of the moving object across the different frames; to measure the correctness of the idea, it's used the median IoU between my detections and the results contained in a label set. 

The program gives in output a .txt file containing the coordinates of the prediction boxes and shows the first frame of each category of the data set whith the correct box in red and the detected in green.

The detection pipeline is the following:
- feature selection, using Shi-Tomasi algorithm;
- optical flow, using Lucas-Kenade method;
- motion filtering, cause we want to consider only points with a significant movement;
- bounding box generation.

Project Structure:
data/: Input directory containing folders of image sequences.
labels/: Ground truth annotations for each sequence.
src/: C++ source files (Motion detection logic, loaders, and utilities).
detections.txt: Output file containing coordinates in <xmin> <ymin> <xmax> <ymax> format.

the data set can be found at this link:
https://drive.google.com/drive/folders/193LYvOr5Z0LAmVjv1kSWkkdUB2uaSAS1

the label set can be found at this link: 
https://drive.google.com/drive/folders/19ymfHkcEQnW6-iI1KPHfhs8zn0D7KVYX

the project works with cmake