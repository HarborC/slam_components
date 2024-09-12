#pragma once

#include <opencv2/opencv.hpp>

class Matcher {
public:
  Matcher() {}
  ~Matcher() {}

  void matchORB(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                std::vector<cv::DMatch> &matches);
};