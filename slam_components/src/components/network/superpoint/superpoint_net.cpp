#include "components/network/superpoint/superpoint_net.h"

#include <fstream>
#include <iostream>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "components/network/utils.h"
#include "utils/log_utils.h"

namespace slam_components {

bool SuperpointNet::initialize(const cv::FileNode &node) {
  torch::NoGradGuard no_grad;

  device_ = torch::Device(torch::kCUDA);
  if (!torch::cuda::is_available()) {
    SPDLOG_WARN("CUDA is not available, using CPU instead.");
    device_ = torch::Device(torch::kCPU);
  }

  std::string net_path;

  if (node["net_path"].empty()) {
    SPDLOG_CRITICAL("net_path is not provided");
    return false;
  }

  node["net_path"] >> net_path;

  try {
    net_ = torch::jit::load(std::string(PROJECT_DIR) + net_path);
  } catch (const c10::Error &e) {
    SPDLOG_CRITICAL("Error loading net_path");
    return false;
  }

  net_.to(device_);
  net_.eval();

  initialized_ = true;

  SPDLOG_INFO("SuperpointNet initialized");

  return true;
}

void SuperpointNet::warmup() {
  // Implement warmup
  // Warm up the network
  {
    torch::Tensor x0 = torch::randn({1, 1, 480, 752}).to(device_);
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(x0);
    net_.forward(input_tensors);
  }
}

} // namespace slam_components