#include "components/network/network.h"

#include <fstream>
#include <iostream>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "components/network/utils.h"
#include "utils/log_utils.h"

namespace slam_components {

bool Network::initialize(const cv::FileNode &node) {
  printLibtorchVersion();
  printCudaCuDNNInfo();
  torch::globalContext().setBenchmarkCuDNN(true);
  // torch::globalContext().setDeterministicCuDNN(true);
  torch::globalContext().setUserEnabledCuDNN(true);
  torch::autograd::GradMode::set_enabled(false);

  if (node["droid"].empty()) {
    SPDLOG_CRITICAL("droid is not provided");
    return false;
  }

  droid_net_.reset(new DroidNet());
  if (!droid_net_->initialize(node["droid"])) {
    SPDLOG_CRITICAL("Failed to initialize DroidNet");
    return false;
  }

  if (node["superpoint"].empty()) {
    SPDLOG_CRITICAL("superpoint is not provided");
    return false;
  }

  superpoint_net_.reset(new SuperpointNet());
  if (!superpoint_net_->initialize(node["superpoint"])) {
    SPDLOG_CRITICAL("Failed to initialize SuperpointNet");
    return false;
  }

  warmup();

  return true;
}

void Network::warmup() {
  // Implement warmup
  if (droid_net_)
    droid_net_->warmup();
  if (superpoint_net_)
    superpoint_net_->warmup();
}

} // namespace slam_components