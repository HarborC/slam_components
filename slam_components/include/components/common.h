#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <cereal/cereal_utils.hpp>

#define DEFAULT_DEPTH 1.0

typedef long long int FeatureIDType;
typedef long long int FrameIDType;

namespace cv {
// Define a new bool reader in order to accept "true/false"-like values.
inline void read_bool(const cv::FileNode &node, bool &value,
                      const bool &default_value) {
  std::string s(static_cast<std::string>(node));
  if (s == "y" || s == "Y" || s == "yes" || s == "Yes" || s == "YES" ||
      s == "true" || s == "True" || s == "TRUE" || s == "on" || s == "On" ||
      s == "ON") {
    value = true;
    return;
  }
  if (s == "n" || s == "N" || s == "no" || s == "No" || s == "NO" ||
      s == "false" || s == "False" || s == "FALSE" || s == "off" ||
      s == "Off" || s == "OFF") {
    value = false;
    return;
  }
  value = static_cast<int>(node);
}
// Specialize cv::operator>> for bool.
template <> inline void operator>>(const cv::FileNode &n, bool &value) {
  read_bool(n, value, false);
}
} // namespace cv