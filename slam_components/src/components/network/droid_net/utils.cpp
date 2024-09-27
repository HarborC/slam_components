#include "components/network/droid_net/utils.h"
#include "utils/log_utils.h"

namespace slam_components {

torch::Tensor getCoordsGrid(int64_t ht, int64_t wd, torch::Device device) {
  torch::Tensor y = torch::arange(
      ht, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  torch::Tensor x = torch::arange(
      wd, torch::TensorOptions().device(device).dtype(torch::kFloat32));

  std::vector<torch::Tensor> meshgrid = torch::meshgrid({x, y}, "xy");

  torch::Tensor x_grid = meshgrid[0];
  torch::Tensor y_grid = meshgrid[1];

  return torch::stack({x_grid, y_grid}, -1);
}

void printLibtorchVersion() {
  SPDLOG_INFO("PyTorch version: {}.{}.{}", TORCH_VERSION_MAJOR,
              TORCH_VERSION_MINOR, TORCH_VERSION_PATCH);
}

void printCudaCuDNNInfo() {
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  SPDLOG_INFO("The cuDNN version is {}", cudnn_version);
  int runtime_version;
  AT_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  SPDLOG_INFO("The CUDA runtime version is {}", runtime_version);
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  SPDLOG_INFO("The driver version is  {}", version);
}

} // namespace slam_components