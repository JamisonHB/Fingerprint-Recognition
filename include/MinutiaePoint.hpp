#pragma once
#include <utility>
#include <string>


class MinutiaePoint {
private:
	std::pair<int, int> position; // (x, y) coordinates
	std::string type; // "ending" or "bifurcation"

public:
	MinutiaePoint(int x, int y, const std::string& type);

	std::pair<int, int> getPosition() const;

	std::string getType() const;

	bool operator==(const MinutiaePoint& other) const;
};