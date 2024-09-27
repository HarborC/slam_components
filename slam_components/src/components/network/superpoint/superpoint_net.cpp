#include "components/network/superpoint/superpoint_net.h"

#include <fstream>
#include <iostream>
#include <torch/script.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "components/network/utils.h"
#include "utils/log_utils.h"

namespace slam_components {

bool SuperpointNet::initialize(const cv::FileNode &node) {
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
    torch::load(net_, std::string(PROJECT_DIR) + net_path);
  } catch (const c10::Error &e) {
    SPDLOG_CRITICAL("Error : {}", e.what());
    SPDLOG_CRITICAL("Error loading net_path");
    return false;
  }

  net_->to(device_);
  net_->eval();

  if (!node["nms_radius"].empty()) {
    node["nms_radius"] >> nms_radius_;
  }

  if (!node["max_num_keypoints"].empty()) {
    node["max_num_keypoints"] >> max_num_keypoints_;
  }

  if (!node["detection_threshold"].empty()) {
    node["detection_threshold"] >> detection_threshold_;
  }

  if (!node["remove_borders"].empty()) {
    node["remove_borders"] >> remove_borders_;
  }

  initialized_ = true;

  SPDLOG_INFO("SuperpointNet initialized");

  return true;
}

void SuperpointNet::warmup() {
  // Implement warmup
  // Warm up the network
  {
    torch::Tensor x0 = torch::randn({1, 1, 480, 752}).to(device_);
    net_->forward(x0, nms_radius_, max_num_keypoints_, detection_threshold_,
                  remove_borders_);
  }
}

} // namespace slam_components