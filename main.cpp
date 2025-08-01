#include <opencv2/opencv.hpp>
#include "include/imageProcessing.hpp"
#include "include/matching.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// Helper function to load all image paths from the dataset directory
std::vector<std::string> getDatabasePaths(const std::string& directory) {
    std::vector<std::string> paths;
    if (!std::filesystem::exists(directory)) {
        throw std::runtime_error("Dataset directory not found: " + directory);
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".png") {
            paths.push_back(entry.path().string());
        }
    }
    return paths;
}

int main() {
    // SETUP
	const std::string datasetDirectory = "./validation_imagery_enhanced/"; // Adjust this path to dataset directory
	const std::string probeImagePath = datasetDirectory + "a015_02.png"; // Adjust this path to your probe image
    const double ACCEPTANCE_THRESHOLD = 0.02; // Define a threshold for accepting matches

    std::vector<std::string> databaseImagePaths = getDatabasePaths(datasetDirectory);

    // PROCESS PROBE IMAGE
    std::cout << "Processing Probe Image: " << probeImagePath << std::endl;
    cv::Mat probeImage = cv::imread(probeImagePath, cv::IMREAD_GRAYSCALE);
    if (probeImage.empty()) {
        std::cerr << "Failed to load probe image." << std::endl;
        return -1;
    }
    binarizeFingerprint(probeImage);
    cv::Mat thinnedProbe = thinFingerprint(probeImage);
	cleanThinnedImage(thinnedProbe);
    std::vector<MinutiaePoint> probeMinutiae = findMinutiae(thinnedProbe);
    removeFalseMinutiae(probeMinutiae, thinnedProbe.cols, thinnedProbe.rows);
    std::cout << "Probe processing complete. Found " << probeMinutiae.size() << " minutiae.\n" << std::endl;

    // PROCESS AND MATCH AGAINST DATABASE
    std::cout << "Matching probe against database..." << std::endl;
    double bestScore = -1.0;
    int bestMatchIndex = -1;

    for (size_t i = 0; i < databaseImagePaths.size(); ++i) {
        const auto& dbPath = databaseImagePaths[i];
        // Don't match the probe against itself
        if (dbPath == probeImagePath) continue;

        cv::Mat dbImage = cv::imread(dbPath, cv::IMREAD_GRAYSCALE);
        if (dbImage.empty()) continue;

        binarizeFingerprint(dbImage);
        cv::Mat thinnedDb = thinFingerprint(dbImage);
		cleanThinnedImage(thinnedDb);
        std::vector<MinutiaePoint> dbMinutiae = findMinutiae(thinnedDb);
        removeFalseMinutiae(dbMinutiae, thinnedDb.cols, thinnedDb.rows);

        double currentScore = calculateMatchScore(probeMinutiae, thinnedProbe, dbMinutiae, thinnedDb);
        std::cout << "  - Score with " << dbPath << ": " << currentScore << std::endl;

        if (currentScore > bestScore) {
            bestScore = currentScore;
            bestMatchIndex = static_cast<int>(i);
        }
    }

    // SHOW RESULTS
    std::cout << "\n--- Best Match Found ---" << std::endl;
    std::cout << "Probe: " << probeImagePath << std::endl;
    std::cout << "Best Match: " << databaseImagePaths[bestMatchIndex] << std::endl;
    std::cout << "Score: " << bestScore << std::endl;

    if (bestScore >= ACCEPTANCE_THRESHOLD) {
        std::cout << "Result: Match Accepted" << std::endl;
    }
    else {
        std::cout << "Result: Match Rejected (Score below threshold of " << ACCEPTANCE_THRESHOLD << ")" << std::endl;
    }

    // Display the two images side-by-side for visual comparison
    cv::Mat bestMatchImage = cv::imread(databaseImagePaths[bestMatchIndex], cv::IMREAD_GRAYSCALE);
    displayFingerprintObjectsGrid({
        overlayMinutiae(cv::imread(probeImagePath, cv::IMREAD_GRAYSCALE), probeMinutiae),
        overlayMinutiae(bestMatchImage, {})
        });

    return 0;
}