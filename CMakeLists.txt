cmake_minimum_required(VERSION 3.10)

project(FingerprintRecognition)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add dlib subdirectory
add_subdirectory(dlib)

# Add your executable
add_executable(FingerPrintRecognition main.cpp)

# Link to dlib
target_link_libraries(FingerPrintRecognition dlib::dlib)
