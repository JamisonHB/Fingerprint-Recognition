#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "include/MinutiaePoint.hpp"

// Function to display a grid of fingerprint images
void displayFingerprintGrid(const std::vector<std::string>& imagePaths);

void displayFingerprintObjectsGrid(const std::vector<cv::Mat>& images);

// Function to binarize an image
void binarizeFingerprint(cv::Mat& img);

// Function to find neighboring pixels
std::vector<uchar> findNeighbors(const cv::Mat& img, int i, int j);

int findTransitions(std::vector<uchar> neighbors);

// Function to thin a fingerprint image
cv::Mat thinFingerprint(const cv::Mat& img);

// Function to find minutiae points in a fingerprint image
std::vector<MinutiaePoint> findMinutiae(const cv::Mat& thinnedImage);

cv::Mat overlayMinutiae(const cv::Mat& img, const std::vector<MinutiaePoint>& minutiaePoints);