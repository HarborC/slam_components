#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>

typedef Eigen::Vector2d Point2D;

class Polygon2D {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    Polygon2D() = default;
    Polygon2D(const std::vector<Point2D> &points) : points_(points) {}
    
    void addPoint(const Point2D &point) { points_.push_back(point); }
    
    const std::vector<Point2D> &points() const { return points_; }
    
    size_t size() const { return points_.size(); }

    bool empty() const { return points_.empty(); }
    
    Point2D &operator[](size_t i) { return points_[i]; }
    
    const Point2D &operator[](size_t i) const { return points_[i]; }

    // Check if point P is inside polygon (Ray casting algorithm)
    bool pointInPolygon(const Point2D &P) const;

    // Calculate the area of a polygon using the shoelace formula
    double polygonArea() const;

    // Remove duplicate points
    void removeDuplicates();

    // Sort points by angle around centroid
    void sortPoints();

private:
    std::vector<Point2D> points_;
};

// Check if segments AB and CD intersect and find intersection point
bool segmentIntersect(const Point2D &A, const Point2D &B,
                      const Point2D &C, const Point2D &D,
                      Point2D &intersection);

// Calculate the area of the intersection of two polygons
double polyIntersectionArea(const Polygon2D &poly1, const Polygon2D &poly2);