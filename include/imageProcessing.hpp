#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

// Function to display a grid of fingerprint images
void displayFingerprintGrid(const std::vector<std::string>& imagePaths);

void displayFingerprintObjectsGrid(const std::vector<cv::Mat>& images);

// Function to binarize an image
void binarizeFingerprint(const cv::Mat& img);

std::vector<uchar> findNeighbors(const cv::Mat& img, int x, int y);