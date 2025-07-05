#include "include/matching.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double calculateMatchScore(const std::vector<MinutiaePoint>& minutiaeA, const std::vector<MinutiaePoint>& minutiaeB) {
	double bestScore = 0.0;
	if (minutiaeA.empty() || minutiaeB.empty()) {
		return 0.0;
	}

	// Test all possible alignments for closest match
	for (auto& mA : minutiaeA) {
		for (auto& mB : minutiaeB) {
			// Calculate the translation and rotation needed to align mB with mA
			// and apply it to all points in minutiaeB
			double dTheta = mA.getAngle() - mB.getAngle();
			std::vector<MinutiaePoint> transformedMinutiae;

			for (const auto& pB : minutiaeB) {
				// Applying transformation using rotation matrix
				int xT = static_cast<int>((pB.getPosition().first - mB.getPosition().first) * cos(dTheta) - (pB.getPosition().second - mB.getPosition().second) * sin(dTheta)) + mA.getPosition().first;
				int yT = static_cast<int>((pB.getPosition().first - mB.getPosition().first) * sin(dTheta) + (pB.getPosition().second - mB.getPosition().second) * cos(dTheta)) + mA.getPosition().second;

				// Calculate new angle
				double angleT = pB.getAngle() + dTheta;

				// Add transformed point to transformedMinutiae
				transformedMinutiae.push_back(MinutiaePoint(xT, yT, pB.getType(), angleT));
			}

			const double DISTANCE_THRESHOLD_SQ = std::pow(8, 2); // Distance threshold squared (8 pixels, adjustable)
			const double ANGLE_THRESHOLD = M_PI / 30; // Angle threshold (6 degrees, adjustable)
			int matchedPairs = 0;
			double score;

			// Check if best matching pair is within thresholds
			for (const auto& pA : minutiaeA) {
				double MIN_DISTANCE_SQ = -1.0;
				const MinutiaePoint* bestMatch = nullptr;

				for (const auto& mTB : transformedMinutiae) {
					double distanceSq = std::pow(pA.getPosition().first - mTB.getPosition().first, 2) +
										std::pow(pA.getPosition().second - mTB.getPosition().second, 2);

					if (MIN_DISTANCE_SQ == -1 || distanceSq < MIN_DISTANCE_SQ) {
						MIN_DISTANCE_SQ = distanceSq;
                        bestMatch = &mTB;
					}
				}

				if (bestMatch != nullptr && MIN_DISTANCE_SQ <= DISTANCE_THRESHOLD_SQ) {
					double angleDiff = std::abs(bestMatch->getAngle() - pA.getAngle());

					// Handle angle wraparound
					if (angleDiff > M_PI) {
						angleDiff = 2 * M_PI - angleDiff;
					}

					if (angleDiff <= ANGLE_THRESHOLD && bestMatch->getType() == pA.getType()) {
						matchedPairs++;
					}
				}
			}
			// Calculate the score based on matched pairs & update bestScore if higher
			double score = (static_cast<double>(std::pow(matchedPairs, 2)) / (minutiaeA.size() * minutiaeB.size()));
			if (score > bestScore) {
				bestScore = score;
			}
		}
	}
	return bestScore;
}