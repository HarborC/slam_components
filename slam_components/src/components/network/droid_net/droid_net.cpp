#include "components/network/droid_net/droid_net.h"

#include <fstream>
#include <iostream>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "components/network/droid_net/utils.h"

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
    droid_fnet_ = torch::jit::load(std::string(PROJECT_DIR) + fnet_path);
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
    droid_cnet_ = torch::jit::load(std::string(PROJECT_DIR) + cnet_path);
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
    droid_update_ = torch::jit::load(std::string(PROJECT_DIR) + update_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading update_path\n";
    return false;
  }

  droid_update_.to(device_);
  droid_update_.eval();

  initialized_ = true;
  warmup();

  std::cout << "DroidNet initialized\n";

  return true;
}

void DroidNet::warmup() {
  // TODO: Implement warmup
  // Warm up the network
  printLibtorchVersion();
  printCudaCuDNNInfo();

  torch::globalContext().setBenchmarkCuDNN(true);
  // torch::globalContext().setDeterministicCuDNN(true);
  torch::globalContext().setUserEnabledCuDNN(true);

  // Warm up the network
  {
    torch::Tensor x0 = torch::randn({1, 2, 3, 480, 752}).to(device_);
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(x0);
    droid_fnet_.forward(input_tensors);
    droid_cnet_.forward(input_tensors);
  } 
  
  {
    torch::Tensor x1 = torch::randn({1, 2, 128, 60, 94}).to(device_);
    torch::Tensor x2 = torch::randn({1, 2, 128, 60, 94}).to(device_);
    torch::Tensor x3 = torch::randn({1, 2, 196, 60, 94}).to(device_);
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(x1);
    input_tensors.push_back(x2);
    input_tensors.push_back(x3);
    droid_update_.forward(input_tensors);
  }
}

} // namespace slam_components