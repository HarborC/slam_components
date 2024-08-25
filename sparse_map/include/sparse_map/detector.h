#pragma once

#include "sparse_map/common.h"

class Detector {
public:
  Detector() {}
  ~Detector() {}

  void detectORB(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints,
                 cv::Mat &descriptors, const cv::Mat &mask = cv::Mat());

};