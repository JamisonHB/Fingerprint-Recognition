#include <opencv2/opencv.hpp>
#include "include/imageProcessing.hpp"
#include "include/matching.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// A helper struct to hold all data for a single fingerprint
struct FingerprintData {
    std::string path;
    std::string personId;
    std::vector<MinutiaePoint> minutiae;
};

// Helper function to extract the person ID (e.g., "001") from a filename
std::string parsePersonId(const std::string& filename) {
    size_t start = 1;
    size_t end = filename.find('_');
    if (end != std::string::npos && end > start) {
        return filename.substr(start, end - start);
    }
    return "";
}

int main() {
    // --- 1. SETUP ---
	const std::string databaseDirectory = "./validation_imagery_enhanced/"; // Adjust this path to your dataset directory
    if (!std::filesystem::exists(databaseDirectory)) {
        throw std::runtime_error("Dataset directory not found: " + databaseDirectory);
    }
    std::vector<FingerprintData> fingerprints;

    // --- 2. LOAD AND PRE-PROCESS ALL IMAGES ---
    std::cout << "Loading and processing all fingerprints from: " << databaseDirectory << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(databaseDirectory)) {
        if (entry.path().extension() != ".png" && entry.path().extension() != ".bmp") continue; // Skip non-png/bmp files

        std::string path = entry.path().string();
        std::string filename = entry.path().filename().string();

        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Failed to load: " << filename << std::endl;
            continue;
        }

        binarizeFingerprint(image);
        cv::Mat thinnedImage = thinFingerprint(image);
		cleanThinnedImage(thinnedImage);
        std::vector<MinutiaePoint> minutiae = findMinutiae(thinnedImage);
        removeFalseMinutiae(minutiae, thinnedImage.cols, thinnedImage.rows);

        fingerprints.push_back({ path, parsePersonId(filename), minutiae });
    }
    std::cout << "Processing complete. Loaded " << fingerprints.size() << " fingerprints.\n" << std::endl;

    // --- 3. CALCULATE TOP-1 IDENTIFICATION ACCURACY ---
    std::cout << "Calculating Top-1 identification accuracy..." << std::endl;
    int correctTopMatches = 0;
    for (size_t i = 0; i < fingerprints.size(); ++i) {
        const auto& probe = fingerprints[i];
        double bestScore = -1.0;
        int bestMatchIndex = -1;

        for (size_t j = 0; j < fingerprints.size(); ++j) {
            if (i == j) continue; // Don't compare a print to itself
            const auto& candidate = fingerprints[j];
            double score = calculateMatchScore(probe.minutiae, candidate.minutiae);
            if (score > bestScore) {
                bestScore = score;
                bestMatchIndex = j;
            }
        }

        if (bestMatchIndex != -1 && probe.personId == fingerprints[bestMatchIndex].personId) {
            correctTopMatches++;
        }
        std::cout << "  Probe " << i + 1 << "/" << fingerprints.size() << " completed...\r";
    }
    std::cout << std::string(80, ' ') << "\r";


    // --- 4. PERFORM AUTOMATED THRESHOLD TESTING ---
    std::cout << "\nStarting automated threshold testing..." << std::endl;
    std::cout << "\n--- Threshold CSV Data ---" << std::endl;
    std::cout << "Threshold,FAR,FRR" << std::endl;

	// Test thresholds from 0.0 to 0.5 in increments of 0.01
    for (double threshold = 0.0; threshold <= 0.5; threshold += 0.01) {
        int falseAccepts = 0;
        int falseRejects = 0;
        int totalImpostorPairs = 0;
        int totalTruePairs = 0;

        for (size_t i = 0; i < fingerprints.size(); ++i) {
            for (size_t j = i + 1; j < fingerprints.size(); ++j) {
                bool isTruePair = (fingerprints[i].personId == fingerprints[j].personId);
                if (isTruePair) totalTruePairs++;
                else totalImpostorPairs++;

                double score = calculateMatchScore(fingerprints[i].minutiae, fingerprints[j].minutiae);

                if (isTruePair && score < threshold) falseRejects++;
                else if (!isTruePair && score >= threshold) falseAccepts++;
            }
        }

        double FAR = (totalImpostorPairs > 0) ? static_cast<double>(falseAccepts) / totalImpostorPairs : 0.0;
        double FRR = (totalTruePairs > 0) ? static_cast<double>(falseRejects) / totalTruePairs : 0.0;
        std::cout << threshold << "," << FAR * 100.0 << "," << FRR * 100.0 << std::endl;
    }

    // --- 5. DISPLAY FINAL STATS ---
    std::cout << "\n--- Testing Complete ---" << std::endl;
    double top1_accuracy = (fingerprints.size() > 0) ? static_cast<double>(correctTopMatches) / fingerprints.size() : 0.0;
    std::cout << "Correct Top-1 Matches: " << correctTopMatches << " / " << fingerprints.size()
        << " (" << top1_accuracy * 100.0 << "%)" << std::endl;

    return 0;
}