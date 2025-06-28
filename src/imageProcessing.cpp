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

cv::Mat binarizeImage(const cv::Mat& img, int threshold) {
    cv::Mat binaryImg;
    cv::threshold(img, binaryImg, threshold, 255, cv::THRESH_BINARY);
    return binaryImg;
}
