#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "MinutiaePoint.hpp"

// Finds the best possible alignment between two prints
double calculateMatchScore(const std::vector<MinutiaePoint>& minutiaeA, const cv::Mat& thinnedA, const std::vector<MinutiaePoint>& minutiaeB, const cv::Mat& thinnedB);