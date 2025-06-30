#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
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
    for (const auto& img : images) {
        if (img.empty()) {
            std::cerr << "Empty image encountered.\n";
            continue;
        }
        // Resize and add black border
        cv::Mat resizedImg;
        cv::resize(img, resizedImg, targetSize);
        cv::copyMakeBorder(resizedImg, resizedImg, borderSize, borderSize, borderSize, borderSize,
            cv::BORDER_CONSTANT, cv::Scalar(255));
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

// Function to find the 8 neighbors of a pixel in a grayscale image
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