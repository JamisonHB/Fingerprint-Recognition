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

// Function to denoise a fingerprint image
// Note: this function was written by AI
cv::Mat segmentFingerprint(const cv::Mat& img);

// Helper function to find neighboring pixels
std::vector<uchar> findNeighbors(const cv::Mat& img, int i, int j);

// Helper function to find the number of transitions in a vector of neighbors
int findTransitions(std::vector<uchar> neighbors);

// Function to thin a fingerprint image
cv::Mat thinFingerprint(const cv::Mat& img);

// Helper function to trace the minutiae and get the angle
double traceAndGetAngle(const cv::Mat& thinnedImage, cv::Point minutiaeCenter, cv::Point startNode);

// Function to find minutiae points in a fingerprint image
std::vector<MinutiaePoint> findMinutiae(const cv::Mat& thinnedImage);

// Function to remove false minutiae points based on image dimensions
void removeFalseMinutiae(std::vector<MinutiaePoint>& minutiaePoints, int imgWidth, int imgHeight);

// Function to overlay minutiae points on an image
cv::Mat overlayMinutiae(const cv::Mat& img, const std::vector<MinutiaePoint>& minutiaePoints);