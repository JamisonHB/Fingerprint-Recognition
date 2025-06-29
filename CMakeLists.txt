cmake_minimum_required(VERSION 3.10)

project(FingerprintRecognition)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV configuration
set(OpenCV_DIR "C:/Users/jamis/source/C++ Libs/opencv/build")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories("C:/opencv/build/x64/vc16/lib")

# Add dlib subdirectory
add_subdirectory(dlib)

#include user-defined directories
include_directories(include)

#Add source files
set(SOURCES main.cpp src/imageProcessing.cpp)

# Add your executable
add_executable(FingerPrintRecognition main.cpp src/imageProcessing.cpp)

# Link to dlib
target_link_libraries(FingerPrintRecognition dlib::dlib)
target_link_libraries(FingerPrintRecognition ${OpenCV_LIBS})