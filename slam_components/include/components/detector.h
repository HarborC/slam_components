#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace slam_components {

class Detector {
public:
  Detector() {}
  ~Detector() {}

  void detectORB(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                 cv::Mat &descriptors, const cv::Mat &mask = cv::Mat());

  void detectSIFT(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                  cv::Mat &descriptors, const cv::Mat &mask = cv::Mat());
};

} // namespace slam_components