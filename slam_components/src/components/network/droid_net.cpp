#include "components/network/droid_net.h"

#include <fstream>
#include <iostream>
#include <torch/serialize.h>
#include <torch/torch.h>

namespace slam_components {

bool DroidNet::initialize(const cv::FileNode &node) {
  torch::NoGradGuard no_grad;

  device_ = torch::Device(torch::kCUDA);
  if (!torch::cuda::is_available()) {
    std::cout << "CUDA is not available, using CPU instead.\n";
    device_ = torch::Device(torch::kCPU);
  }

  std::string fnet_path, cnet_path, update_path;

  if (node["fnet_path"].empty()) {
    std::cerr << "Error: fnet_path is not provided\n";
    return false;
  }

  node["fnet_path"] >> fnet_path;

  try {
    droid_fnet_ = torch::jit::load(fnet_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading fnet_path\n";
    return false;
  }

  droid_fnet_.to(device_);
	droid_fnet_.eval();

  if (node["cnet_path"].empty()) {
    std::cerr << "Error: cnet_path is not provided\n";
    return false;
  }

  node["cnet_path"] >> cnet_path;

  try {
    droid_cnet_ = torch::jit::load(cnet_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading cnet_path\n";
    return false;
  }

  droid_cnet_.to(device_);
	droid_cnet_.eval();

  if (node["update_path"].empty()) {
    std::cerr << "Error: update_path is not provided\n";
    return false;
  }

  node["update_path"] >> update_path;

  try {
    droid_update_ = torch::jit::load(update_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading update_path\n";
    return false;
  }

  droid_update_.to(device_);
	droid_update_.eval();

  initialized_ = true;
  warmup();

  return true;
}

void DroidNet::warmup() {
  // TODO: Implement warmup
  // Warm up the network
}

} // namespace slam_components