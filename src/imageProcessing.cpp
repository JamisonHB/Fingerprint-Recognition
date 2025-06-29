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

void binarizeFingerprint(const cv::Mat& img) {
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
std::vector<uchar> findNeighbors(const cv::Mat& img, int x, int y) {
    std::vector<uchar> neighbors = {
        img.at<uchar&>(y - 1, x - 1), img.at<uchar&>(y - 1, x), img.at<uchar&>(y - 1, x + 1),
        img.at<uchar&>(y, x - 1), img.at<uchar&>(y, x + 1),
        img.at<uchar&>(y + 1, x - 1), img.at<uchar&>(y + 1, x), img.at<uchar&>(y + 1, x + 1)
	};
	return neighbors;
}