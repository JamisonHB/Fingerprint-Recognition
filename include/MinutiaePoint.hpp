#pragma once
#include <utility>
#include <string>


class MinutiaePoint {
private:
	std::pair<int, int> position; // (x, y) coordinates
	std::string type; // "ending" or "bifurcation"
	double angle; // angle of the minutiae point

public:
	MinutiaePoint();

	MinutiaePoint(int x, int y, const std::string& type, double angle);

	std::pair<int, int> getPosition() const;

	void setPosition(int x, int y);

	std::string getType() const;

	double getAngle() const;

	bool operator==(const MinutiaePoint& other) const;
};