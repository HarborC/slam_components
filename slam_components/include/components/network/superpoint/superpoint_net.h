#pragma once

#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

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
  torch::jit::script::Module net_;
  torch::Device device_ = torch::Device(torch::kCUDA);
};

} // namespace slam_components