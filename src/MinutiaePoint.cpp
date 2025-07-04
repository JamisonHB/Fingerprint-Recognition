#include "MinutiaePoint.hpp"
#include <stdexcept>

MinutiaePoint::MinutiaePoint(int x, int y, const std::string& type, double angle)
	: position(x, y), type(type), angle(angle) {
	if (type != "ending" && type != "bifurcation") {
		throw std::invalid_argument("Type must be 'ending' or 'bifurcation'");
	}
}

std::pair<int, int> MinutiaePoint::getPosition() const {
	return position;
}

std::string MinutiaePoint::getType() const {
	return type;
}

double MinutiaePoint::getAngle() const {
	return angle;
}

bool MinutiaePoint::operator==(const MinutiaePoint& other) const {
	return position == other.position && type == other.type;
}