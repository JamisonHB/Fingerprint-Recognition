#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Function to display a grid of fingerprint images
void displayFingerprintGrid(const std::vector<std::string>& imagePaths);

// Function to binarize an image using a specified threshold
cv::Mat binarizeImage(const cv::Mat& img, int threshold);