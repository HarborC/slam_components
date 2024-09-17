#include "sparse_map/polygon2d.h"

// Check if point P is inside polygon (Ray casting algorithm)
bool Polygon2D::pointInPolygon(const Point2D &P) const {
  int n = points_.size();
  bool inPolygon = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    if (((points_[i].y() > P.y()) != (points_[j].y() > P.y())) &&
        (P.x() < (points_[j].x() - points_[i].x()) * (P.y() - points_[i].y()) /
                         (points_[j].y() - points_[i].y() + 1e-10) +
                     points_[i].x())) {
      inPolygon = !inPolygon;
    }
  }
  return inPolygon;
}

// Calculate the area of a polygon using the shoelace formula
double Polygon2D::polygonArea() const {
  double area = 0;
  int n = points_.size();
  for (int i = 0; i < n; ++i) {
    area += (points_[i].x() * points_[(i + 1) % n].y() -
             points_[(i + 1) % n].x() * points_[i].y());
  }
  return std::fabs(area) / 2.0;
}

// Remove duplicate points
void Polygon2D::removeDuplicates() {
  std::sort(points_.begin(), points_.end(),
            [](const Point2D &a, const Point2D &b) {
              if (std::fabs(a.x() - b.x()) > 1e-8)
                return a.x() < b.x();
              return a.y() < b.y();
            });
  points_.erase(unique(points_.begin(), points_.end(),
                       [](const Point2D &a, const Point2D &b) {
                         return std::fabs(a.x() - b.x()) < 1e-8 &&
                                std::fabs(a.y() - b.y()) < 1e-8;
                       }),
                points_.end());
}

// Sort points by angle around centroid
void Polygon2D::sortPoints() {
  // Compute centroid
  Point2D centroid(0, 0);
  for (const auto &p : points_) {
    centroid.x() += p.x();
    centroid.y() += p.y();
  }
  centroid.x() /= points_.size();
  centroid.y() /= points_.size();

  // Sort points by angle around centroid
  std::sort(points_.begin(), points_.end(),
            [&centroid](const Point2D &a, const Point2D &b) {
              double angleA =
                  std::atan2(a.y() - centroid.y(), a.x() - centroid.x());
              double angleB =
                  std::atan2(b.y() - centroid.y(), b.x() - centroid.x());
              return angleA < angleB;
            });
}

// Check if segments AB and CD intersect and find intersection point
bool segmentIntersect(const Point2D &A, const Point2D &B, const Point2D &C,
                      const Point2D &D, Point2D &intersection) {
  double a1 = B.y() - A.y();
  double b1 = A.x() - B.x();
  double c1 = a1 * A.x() + b1 * A.y();
  double a2 = D.y() - C.y();
  double b2 = C.x() - D.x();
  double c2 = a2 * C.x() + b2 * C.y();
  double determinant = a1 * b2 - a2 * b1;

  if (std::fabs(determinant) < 1e-10) {
    return false; // Parallel lines
  } else {
    double x = (b2 * c1 - b1 * c2) / determinant;
    double y = (a1 * c2 - a2 * c1) / determinant;
    if (std::min(A.x(), B.x()) - 1e-10 <= x &&
        x <= std::max(A.x(), B.x()) + 1e-10 &&
        std::min(A.y(), B.y()) - 1e-10 <= y &&
        y <= std::max(A.y(), B.y()) + 1e-10 &&
        std::min(C.x(), D.x()) - 1e-10 <= x &&
        x <= std::max(C.x(), D.x()) + 1e-10 &&
        std::min(C.y(), D.y()) - 1e-10 <= y &&
        y <= std::max(C.y(), D.y()) + 1e-10) {
      intersection = Point2D(x, y);
      return true;
    }
    return false;
  }
}

// Calculate the area of the intersection of two polygons
double polyIntersectionArea(const Polygon2D &poly1, const Polygon2D &poly2) {
  Polygon2D intersectionPoints;

  // Find intersection points between edges
  for (int i = 0; i < 4; ++i) {
    Point2D A1 = poly1[i];
    Point2D A2 = poly1[(i + 1) % 4];
    for (int j = 0; j < 4; ++j) {
      Point2D B1 = poly2[j];
      Point2D B2 = poly2[(j + 1) % 4];
      Point2D interPt;
      if (segmentIntersect(A1, A2, B1, B2, interPt)) {
        intersectionPoints.addPoint(interPt);
      }
    }
  }

  // Add vertices of poly1 inside poly2
  for (int i = 0; i < 4; ++i) {
    if (poly2.pointInPolygon(poly1[i])) {
      intersectionPoints.addPoint(poly1[i]);
    }
  }

  // Add vertices of poly2 inside poly1
  for (int i = 0; i < 4; ++i) {
    if (poly1.pointInPolygon(poly2[i])) {
      intersectionPoints.addPoint(poly2[i]);
    }
  }

  if (intersectionPoints.empty()) {
    return 0.0;
  }

  // Remove duplicate points
  intersectionPoints.removeDuplicates();

  // Sort points by angle around centroid
  intersectionPoints.sortPoints();

  // Calculate and output the area
  return intersectionPoints.polygonArea();
}
