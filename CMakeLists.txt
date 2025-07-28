cmake_minimum_required(VERSION 3.10)

project(FingerprintRecognition)

# Use C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- OpenCV Configuration ---
# For this to work, you must set the OpenCV_DIR environment variable
# pointing to your OpenCV build directory (e.g., C:/opencv/build).
find_package(OpenCV REQUIRED)

# Include the necessary directories for headers
include_directories(include ${OpenCV_INCLUDE_DIRS})

# Group all shared library source files into a single variable
set(LIB_SOURCES
    src/imageProcessing.cpp
    src/MinutiaePoint.cpp
    src/matching.cpp
)

# --- Executable for the main demonstration program ---
add_executable(FingerprintRecognition
    main.cpp
    ${LIB_SOURCES}
)
# Link the main executable against OpenCV
target_link_libraries(FingerprintRecognition PRIVATE ${OpenCV_LIBS})

# --- Executable for the testing and validation program ---
add_executable(FingerprintRecognitionTest
    testMain.cpp
    ${LIB_SOURCES}
)
# Link the testing executable against OpenCV
target_link_libraries(FingerprintRecognitionTest PRIVATE ${OpenCV_LIBS})