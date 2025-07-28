#include <opencv2/opencv.hpp>
#include <iostream>
#include "imageProcessing.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void displayFingerprintGrid(const std::vector<std::string>& imagePaths) {
    if (imagePaths.empty()) {
        std::cerr << "No images to display.\n";
        return;
    }

    std::vector<cv::Mat> processedImages;

    // Uniform resize
    const cv::Size targetSize(200, 200);
    const int borderSize = 2;

    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load: " << path << std::endl;
            continue;
        }

        // Resize and add black border
        cv::resize(img, img, targetSize);
        cv::copyMakeBorder(img, img, borderSize, borderSize, borderSize, borderSize,
            cv::BORDER_CONSTANT, cv::Scalar(0));
        processedImages.push_back(img);
    }

    if (processedImages.empty()) {
        std::cerr << "No valid images loaded.\n";
        return;
    }

    // Determine grid layout: square or close to square
    int total = static_cast<int>(processedImages.size());
    int cols = static_cast<int>(std::ceil(std::sqrt(total)));
    int rows = static_cast<int>(std::ceil(total / static_cast<double>(cols)));

    // Fill missing cells with blank images (optional)
    while (processedImages.size() < rows * cols) {
        cv::Mat blank = cv::Mat::zeros(targetSize.height + 2 * borderSize,
            targetSize.width + 2 * borderSize,
            CV_8UC1);
        processedImages.push_back(blank);
    }

    // Assemble the grid row by row
    std::vector<cv::Mat> rowsImages;
    for (int r = 0; r < rows; ++r) {
        std::vector<cv::Mat> row;
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            row.push_back(processedImages[idx]);
        }
        cv::Mat rowMat;
        cv::hconcat(row, rowMat);
        rowsImages.push_back(rowMat);
    }

    cv::Mat finalDisplay;
    cv::vconcat(rowsImages, finalDisplay);

    cv::imshow("Fingerprint Grid", finalDisplay);
    cv::waitKey(0);
}

void displayFingerprintObjectsGrid(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        std::cerr << "No images to display.\n";
        return;
    }
    // Uniform resize
    const cv::Size targetSize(200, 200);
    const int borderSize = 2;
    std::vector<cv::Mat> processedImages;
    int imageType = -1;

    for (const auto& img : images) {
        if (img.empty()) {
            std::cerr << "Empty image encountered.\n";
            continue;
        }
        if (imageType == -1) imageType = img.type();

        // Resize and add black border
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, targetSize);
        cv::copyMakeBorder(resizedImg, resizedImg, borderSize, borderSize, borderSize, borderSize,
            cv::BORDER_CONSTANT, imageType == CV_8UC1 ? cv::Scalar(255) : cv::Scalar(255,255,255));
        processedImages.push_back(resizedImg);
    }
    if (processedImages.empty()) {
        std::cerr << "No valid images loaded.\n";
        return;
    }
    // Determine grid layout: square or close to square
    int total = static_cast<int>(processedImages.size());
    int cols = static_cast<int>(std::ceil(std::sqrt(total)));
    int rows = static_cast<int>(std::ceil(total / static_cast<double>(cols)));

    // Fill missing cells with blank images (optional)
    while (processedImages.size() < rows * cols) {
        cv::Mat blank = cv::Mat::zeros(targetSize.height + 2 * borderSize,
            targetSize.width + 2 * borderSize,
            imageType);
        if (imageType == CV_8UC1)
            blank.setTo(255);
        else
            blank.setTo(cv::Scalar(255,255,255));
        processedImages.push_back(blank);
    }
    // Assemble the grid row by row
    std::vector<cv::Mat> rowsImages;
    for (int r = 0; r < rows; ++r) {
        std::vector<cv::Mat> row;
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            row.push_back(processedImages[idx]);
        }
        cv::Mat rowMat;
        cv::hconcat(row, rowMat);
        rowsImages.push_back(rowMat);
    }
    cv::Mat finalDisplay;
    cv::vconcat(rowsImages, finalDisplay);
    cv::imshow("Fingerprint Objects Grid", finalDisplay);
    cv::waitKey(0);
}

void binarizeFingerprint(cv::Mat& img) {
    const int blockSize = 16;

	// Iterates over the image in blocks
    for (int y = 0; y < img.rows; y += blockSize) {
        for (int x = 0; x < img.cols; x += blockSize) {
            // Handle boundary blocks (image edge cases)
            int blockWidth = std::min(blockSize, img.cols - x);
            int blockHeight = std::min(blockSize, img.rows - y);

            // Define the current block
            cv::Rect blockRect(x, y, blockWidth, blockHeight);
            cv::Mat block = img(blockRect);

            // Compute mean intensity of the block
            double meanVal = cv::mean(block)[0];

			// Apply thresholding within the block by iterating over each pixel in the block
            for (int j = 0; j < blockHeight; ++j) {
                for (int i = 0; i < blockWidth; ++i) {
                    uchar& pixel = block.at<uchar>(j, i);
                    pixel = (pixel < meanVal) ? 255 : 0;
                }
            }
        }
    }
}

void adaptiveBinarizeFingerprint(cv::Mat& img) {
    cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);
}

std::vector<uchar> findNeighbors(const cv::Mat& img, int i, int j) {
    if (i <= 0 || j <= 0 || i >= img.rows - 1 || j >= img.cols - 1) {
        return std::vector<uchar>(8, 0);
    }
    return {
        img.at<uchar>(i - 1, j),     // P2 (top)
        img.at<uchar>(i - 1, j + 1), // P3 (top-right)
        img.at<uchar>(i,     j + 1), // P4 (right)
        img.at<uchar>(i + 1, j + 1), // P5 (bottom-right)
        img.at<uchar>(i + 1, j),     // P6 (bottom)
        img.at<uchar>(i + 1, j - 1), // P7 (bottom-left)
        img.at<uchar>(i,     j - 1), // P8 (left)
        img.at<uchar>(i - 1, j - 1)  // P9 (top-left)
    };
}

int findTransitions(std::vector<uchar> neighbors) {
    int transitions = 0;
    for (size_t i = 0; i < neighbors.size(); ++i) {
        if (neighbors[i] == 0 && neighbors[(i + 1) % neighbors.size()] == 1) {
            transitions++;
        }
    }
    return transitions;
}

cv::Mat thinFingerprint(const cv::Mat& img) {
    // Convert to binary: 0 or 1 for thinning logic
    cv::Mat thinnedImage;
    img.convertTo(thinnedImage, CV_8UC1);
    thinnedImage /= 255;

    bool changing = true;

    while (changing) {
        changing = false;
        std::vector<cv::Point> toDelete;

        // --- Step 1 ---
        for (int j = 1; j < thinnedImage.rows - 1; ++j) {
            for (int i = 1; i < thinnedImage.cols - 1; ++i) {
                if (thinnedImage.at<uchar>(j, i) == 1) {
                    std::vector<uchar> neighbors = findNeighbors(thinnedImage, j, i);
                    int transitions = findTransitions(neighbors);
                    int numOfNeighbors = 0;
                    for (uchar n : neighbors) {
                        numOfNeighbors += (n > 0);
                    }

                    if (2 <= numOfNeighbors && numOfNeighbors <= 6 &&
                        transitions == 1 &&
                        neighbors[0] * neighbors[2] * neighbors[4] == 0 &&
                        neighbors[2] * neighbors[4] * neighbors[6] == 0) {
                        toDelete.emplace_back(i, j); // Store as (col, row)
                    }
                }
            }
        }

        for (const auto& pt : toDelete) {
            thinnedImage.at<uchar>(pt.y, pt.x) = 0;
        }
        if (!toDelete.empty()) changing = true;

        toDelete.clear();

        // --- Step 2 ---
        for (int j = 1; j < thinnedImage.rows - 1; ++j) { // Iterate rows
            for (int i = 1; i < thinnedImage.cols - 1; ++i) { // Iterate columns
                if (thinnedImage.at<uchar>(j, i) == 1) {
                    std::vector<uchar> neighbors = findNeighbors(thinnedImage, j, i);
                    int transitions = findTransitions(neighbors);
                    int numOfNeighbors = 0;
                    for (uchar n : neighbors) {
                        numOfNeighbors += (n > 0);
                    }

                    if (2 <= numOfNeighbors && numOfNeighbors <= 6 &&
                        transitions == 1 &&
                        neighbors[0] * neighbors[2] * neighbors[6] == 0 &&
                        neighbors[0] * neighbors[4] * neighbors[6] == 0) {
                        toDelete.emplace_back(i, j); // Store as (col, row)
                    }
                }
            }
        }

        for (const auto& pt : toDelete) {
            thinnedImage.at<uchar>(pt.y, pt.x) = 0;
        }
        if (!toDelete.empty()) changing = true;
    }

    // Convert back to 0/255 format for output
    thinnedImage *= 255;
    return thinnedImage;
}

double traceAndGetAngle(const cv::Mat& thinnedImage, cv::Point minutiaCenter, cv::Point startNode) {
    cv::Point currentPos = startNode;
    cv::Point prevPos = minutiaCenter;

	const int TRACE_LENGTH = 7; // Walk distance (adjustable)
    for (int i = 0; i < TRACE_LENGTH; ++i) {
        bool foundNext = false;
        // Check 8 neighbors to find the next step
        for (int ny = -1; ny <= 1; ++ny) {
            for (int nx = -1; nx <= 1; ++nx) {
                if (ny == 0 && nx == 0) continue;

                cv::Point nextPos(currentPos.x + nx, currentPos.y + ny);

                // Ensure the next point is within bounds
                if (nextPos.x < 0 || nextPos.y < 0 || nextPos.y >= thinnedImage.rows || nextPos.x >= thinnedImage.cols) continue;

                // Move to the next ridge pixel that isn't the one we just came from
                if ((nextPos != prevPos) && (thinnedImage.at<uchar>(nextPos) > 0)) {
                    prevPos = currentPos;
                    currentPos = nextPos;
                    foundNext = true;
                    goto next_trace_step; // Exit neighbor loops
                }
            }
        }
    next_trace_step:
        if (!foundNext) break; // Stop if the path ends
    }

    // Return the angle of the vector from the minutia's center to the end of our trace
    return atan2(static_cast<double>(currentPos.y - minutiaCenter.y), static_cast<double>(currentPos.x - minutiaCenter.x));
}

std::vector<MinutiaePoint> findMinutiae(const cv::Mat& thinnedImage) {
    std::vector<MinutiaePoint> minutiaePoints;
    
	// Iterate through the thinned image, skipping the border pixels
    for (int y = 1; y < thinnedImage.rows - 1; y++) {
        for (int x = 1; x < thinnedImage.cols - 1; x++) {

			// Ridge pixel check
            if (thinnedImage.at<uchar>(y, x) > 0) {

                std::vector<uchar> neighbors = findNeighbors(thinnedImage, y, x);

				// Normalize to 0 or 1
                for (auto& val : neighbors) {
                    val = val > 0 ? 1 : 0;
                }

                // Calculate the Crossing Number (sum of absolute differences / 2)
                int transitions = 0;
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    transitions += std::abs(neighbors[i] - neighbors[(i + 1) % neighbors.size()]);
                }
                int crossingNumber = transitions / 2;

                // Check for minutiae types and add them
                cv::Point center(x, y);

                // Check for Endings
                if (crossingNumber == 1) {
                    // Find the single neighboring ridge pixel to start the trace
                    for (int ny = -1; ny <= 1; ++ny) {
                        for (int nx = -1; nx <= 1; ++nx) {
                            if (ny == 0 && nx == 0) continue;
                            cv::Point neighbor(x + nx, y + ny);
                            if (thinnedImage.at<uchar>(neighbor) > 0) {
                                double angle = traceAndGetAngle(thinnedImage, center, neighbor);
                                minutiaePoints.push_back(MinutiaePoint(x, y, "ending", angle));
								break; // Neighbor found, no need to check others
                            }
                        }
                    }
                }
                // Check for Bifurcations
                else if (crossingNumber == 3) {
                    std::vector<double> angles;
                    // Find the 3 neighboring ridge pixels
                    for (int ny = -1; ny <= 1; ++ny) {
                        for (int nx = -1; nx <= 1; ++nx) {
                            if (ny == 0 && nx == 0) continue;
                            cv::Point neighbor(x + nx, y + ny);
                            if (thinnedImage.at<uchar>(neighbor) > 0) {
                                angles.push_back(traceAndGetAngle(thinnedImage, center, neighbor));
                            }
                        }
                    }

                    // If we found 3 angles, find the orientation of the stem
                    if (angles.size() == 3) {
                        double d1 = std::abs(angles[0] - angles[1]);
                        double d2 = std::abs(angles[1] - angles[2]);
                        double d3 = std::abs(angles[2] - angles[0]);

                        if (d1 > M_PI) d1 = 2 * M_PI - d1;
                        if (d2 > M_PI) d2 = 2 * M_PI - d2;
                        if (d3 > M_PI) d3 = 2 * M_PI - d3;

                        double bifurcation_angle = 0;
                        if (d1 <= d2 && d1 <= d3) {
                            bifurcation_angle = angles[2] + M_PI;
                        }
                        else if (d2 <= d1 && d2 <= d3) {
                            bifurcation_angle = angles[0] + M_PI;
                        }
                        else {
                            bifurcation_angle = angles[1] + M_PI;
                        }

                        bifurcation_angle = atan2(sin(bifurcation_angle), cos(bifurcation_angle));
                        minutiaePoints.push_back(MinutiaePoint(x, y, "bifurcation", bifurcation_angle));
                    }
                }
            }
        }
    }
    return minutiaePoints;
}

void removeFalseMinutiae(std::vector<MinutiaePoint>& minutiaePoints, int imgWidth, int imgHeight) {
	const int BORDER_MARGIN = 15; // 15 pixels from the border (adjustable)
	const double MIN_DISTANCE_SQ = std::pow(11.0, 2); // Minimum distance squared between minutiae points (12 pixels, adjustable)

    // Rule 1: Mark minutiae too close to the border for removal
    std::vector<bool> to_remove(minutiaePoints.size(), false);
    for (size_t i = 0; i < minutiaePoints.size(); ++i) {
        auto pos = minutiaePoints[i].getPosition();
        if (pos.first < BORDER_MARGIN || pos.first > imgWidth - BORDER_MARGIN ||
            pos.second < BORDER_MARGIN || pos.second > imgHeight - BORDER_MARGIN) {
            to_remove[i] = true;
        }
    }

    // Rule 2: Mark minutiae that are too close to each other
    for (size_t i = 0; i < minutiaePoints.size(); ++i) {
        if (to_remove[i]) continue;
        for (size_t j = i + 1; j < minutiaePoints.size(); ++j) {
            if (to_remove[j]) continue;

            auto pos1 = minutiaePoints[i].getPosition();
            auto pos2 = minutiaePoints[j].getPosition();
            double distSq = std::pow(pos1.first - pos2.first, 2) + std::pow(pos1.second - pos2.second, 2);

            if (distSq < MIN_DISTANCE_SQ) {
                to_remove[i] = true;
                to_remove[j] = true;
            }
        }
    }

    // Move all items not marked for removal to the front,
    // then erase the rest of the vector.
    auto new_end = std::remove_if(minutiaePoints.begin(), minutiaePoints.end(),
        [&](const MinutiaePoint& point) {
            // Get the index of the current point to check our 'to_remove' list
            size_t index = &point - &minutiaePoints[0];
            return to_remove[index];
        });

    minutiaePoints.erase(new_end, minutiaePoints.end());
}

cv::Mat overlayMinutiae(const cv::Mat& img, const std::vector<MinutiaePoint>& minutiaePoints) {
    cv::Mat overlayedImage = img.clone();
    cv::cvtColor(img, overlayedImage, cv::COLOR_GRAY2BGR);

    for (const auto& point : minutiaePoints) {
        auto pos = point.getPosition();
        if (point.getType() == "ending") {
            cv::circle(overlayedImage, cv::Point(pos.first, pos.second), 5, cv::Scalar(0, 0, 255), -1);
        } else if (point.getType() == "bifurcation") {
            cv::circle(overlayedImage, cv::Point(pos.first, pos.second), 7, cv::Scalar(0, 255, 0), -1);
		}
    }
    return overlayedImage;
}