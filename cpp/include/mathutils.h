#ifndef __MATHUTILS_H__
#define __MATHUTILS_H__

#include <cmath>
#include <stdexcept>

template <typename T>
struct point3d {
public:
    point3d() : x(0), y(0), z(0) {}
    point3d(const T x, const T y, const T z) : x(x), y(y), z(z) {}

    point3d operator+(const point3d& other) const {
        return point3d(x + other.x, y + other.y, z + other.z);
    }
    point3d operator+(const T scalar) const {
        return point3d(x + scalar, y + scalar, z + scalar);
    }
    point3d operator-(const point3d& other) const {
        return point3d(x - other.x, y - other.y, z - other.z);
    }
    point3d operator-(const T scalar) const {
        return point3d(x - scalar, y - scalar, z - scalar);
    }
    point3d operator*(const T scalar) const {
        return point3d(x * scalar, y * scalar, z * scalar);
    }
    point3d operator/(const T scalar) const {
        if(scalar == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return point3d(x / scalar, y / scalar, z /scalar);
    }
    point3d operator*(const point3d& other) const {
        return point3d(x * other.x, y * other.y, z * other.z);
    }
    point3d operator/(const point3d& other) const {
        if(other.x == 0 || other.y == 0 || other.z == 0){
            throw std::invalid_argument("Division by zero");
        }
        return point3d(x / other.x, y / other.y, z / other.z);
    }
    point3d& operator=(const point3d& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
    bool operator==(const point3d& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }
    bool operator!=(const point3d& other) const {
        return !(*this == other);
    }

    point3d floor() {
        return point3d(std::floor(x), std::floor(y), std::floor(z));
    }
    point3d ceil() {
        return point3d(std::ceil(x), std::ceil(y), std::ceil(z));
    }
    
    template <typename TargetType>
    point3d<TargetType> convertTo() const {
        return point3d<TargetType>(static_cast<TargetType>(x), static_cast<TargetType>(y), static_cast<TargetType>(z));
    }

    point3d<int32_t> sign() const {
        return point3d<int32_t>(x >= 0 ? 1 : -1, y >= 0 ? 1 : -1, z >= 0 ? 1 : -1 );
    }

    T x, y, z;
};

inline double logOdd(const double probability) {
    if (probability >= 0.99999) return 10;
    if (probability <= 1e-5) return -10;
    return std::log(probability / (1 - probability));
}

inline double probability(const double logodd) {
    return std::exp(logodd) / (1 + std::exp(logodd));
}

inline double entropy(const double probability) {
    return -probability * log(probability) - (1 - probability) * log(1-probability);
}

#endif // __MATHUTILS_H__