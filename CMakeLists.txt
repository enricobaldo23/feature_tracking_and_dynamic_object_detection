cmake_minimum_required(VERSION 3.10)
project("Feature Tracking and Dynamic Object Detection")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(feature_tracking_and_dynamic_object_detection src/main.cpp src/utils.cpp src/motion.cpp src/loader.cpp)
target_link_libraries(feature_tracking_and_dynamic_object_detection ${OpenCV_LIBS})