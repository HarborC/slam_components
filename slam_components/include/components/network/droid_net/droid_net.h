#pragma once

#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

namespace slam_components {

class DroidNet {
public:
  using Ptr = std::shared_ptr<DroidNet>;
  Ptr makeShared() { return std::make_shared<DroidNet>(*this); }

public:
  DroidNet() = default;
  ~DroidNet() {}

  bool initialize(const cv::FileNode &node);

  void warmup();

public:
  bool initialized_ = false;
  torch::jit::script::Module droid_fnet_;
  torch::jit::script::Module droid_cnet_;
  torch::jit::script::Module droid_update_;
  torch::Device device_ = torch::Device(torch::kCUDA);
};

} // namespace slam_components