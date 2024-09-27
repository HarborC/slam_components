#pragma once

#include <memory>
#include <torch/cuda.h>

#include <opencv2/opencv.hpp>

#include "components/network/superpoint/superpoint_impl.h"

namespace slam_components {

class SuperpointNet {
public:
  using Ptr = std::shared_ptr<SuperpointNet>;
  Ptr makeShared() { return std::make_shared<SuperpointNet>(*this); }

public:
  SuperpointNet() = default;
  ~SuperpointNet() {}

  bool initialize(const cv::FileNode &node);

  void warmup();

public:
  bool initialized_ = false;
  SuperPoint net_;
  torch::Device device_ = torch::Device(torch::kCUDA);
  torch::Tensor param_inputs_;

  int nms_radius_ = 4;
  int max_num_keypoints_ = -1;
  double detection_threshold_ = 0.0005;
  int remove_borders_ = 4;
};

} // namespace slam_components