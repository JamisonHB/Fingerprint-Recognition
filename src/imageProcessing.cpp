#include <opencv2/opencv.hpp>
#include <iostream>
#include "include/imageProcessing.hpp"

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

// Function to find the 8 neighbors of a pixel
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


// Function to count transitions in the neighbors
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

std::vector<MinutiaePoint> findMinutiae(const cv::Mat& thinnedImage) {
    std::vector<MinutiaePoint> minutiaePoints;

    // Use a padded image to avoid border checks in the loop
    cv::Mat paddedImage;
    cv::copyMakeBorder(thinnedImage, paddedImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Iterate through each pixel of the original image dimensions
    for (int y = 1; y < paddedImage.rows - 1; y++) {
        for (int x = 1; x < paddedImage.cols - 1; x++) {

            // We only care about ridge pixels (black, value > 0)
            if (paddedImage.at<uchar>(y, x) > 0) {

                // Get 8 neighbors
                uchar p2 = paddedImage.at<uchar>(y - 1, x);
                uchar p3 = paddedImage.at<uchar>(y - 1, x + 1);
                uchar p4 = paddedImage.at<uchar>(y, x + 1);
                uchar p5 = paddedImage.at<uchar>(y + 1, x + 1);
                uchar p6 = paddedImage.at<uchar>(y + 1, x);
                uchar p7 = paddedImage.at<uchar>(y + 1, x - 1);
                uchar p8 = paddedImage.at<uchar>(y, x - 1);
                uchar p9 = paddedImage.at<uchar>(y - 1, x - 1);

                // Create an ordered list of neighbors and normalize to 0 or 1
                std::vector<uchar> neighbors = { p2, p3, p4, p5, p6, p7, p8, p9 };
                for (auto& val : neighbors) {
                    val = val > 0 ? 1 : 0;
                }

                // Calculate the Crossing Number
                int transitions = 0;
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    transitions += std::abs(neighbors[i] - neighbors[(i + 1) % neighbors.size()]);
                }
                int crossingNumber = transitions / 2;

                // Check for minutiae types
                if (crossingNumber == 1) {
                    // Minutiae coordinates must be adjusted back to the original image space (-1)
                    minutiaePoints.push_back(MinutiaePoint(x - 1, y - 1, "ending"));
                }
                else if (crossingNumber == 3) {
                    minutiaePoints.push_back(MinutiaePoint(x - 1, y - 1, "bifurcation"));
                }
            }
        }
    }
    return minutiaePoints;
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