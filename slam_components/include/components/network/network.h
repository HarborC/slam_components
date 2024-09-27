#pragma once

#include <memory>
#include <torch/cuda.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

#include "components/network/droid_net/droid_net.h"
#include "components/network/superpoint/superpoint_net.h"

namespace slam_components {

class Network {
public:
  using Ptr = std::shared_ptr<Network>;
  Ptr makeShared() { return std::make_shared<Network>(*this); }

public:
  Network() = default;
  ~Network() {}

  bool initialize(const cv::FileNode &node);

  void warmup();

public:
  DroidNet::Ptr droid_net_;
  SuperpointNet::Ptr superpoint_net_;
};

} // namespace slam_components