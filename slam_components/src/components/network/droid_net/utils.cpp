#include "components/network/droid_net/utils.h"

namespace slam_components {

torch::Tensor getCoordsGrid(int64_t ht, int64_t wd, torch::Device device) {
  // 创建 ht 和 wd 范围的 arange 张量
  torch::Tensor y = torch::arange(
      ht, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  torch::Tensor x = torch::arange(
      wd, torch::TensorOptions().device(device).dtype(torch::kFloat32));

  // 使用 meshgrid 生成网格
  std::vector<torch::Tensor> meshgrid = torch::meshgrid({x, y}, "xy");

  // 获取 x 和 y 的网格
  torch::Tensor x_grid = meshgrid[0];
  torch::Tensor y_grid = meshgrid[1];

  // std::cout << "x_grid shape: " << x_grid.sizes() << std::endl;
  // std::cout << "y_grid shape: " << y_grid.sizes() << std::endl;

  // 按照最后一个维度堆叠 x 和 y
  torch::Tensor stacked_grid = torch::stack({x_grid, y_grid}, -1);

  return stacked_grid;
}

void printLibtorchVersion() {
  std::cout << "PyTorch version: " << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;
}

void printCudaCuDNNInfo() {
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  std::cout << "The cuDNN version is " << cudnn_version << "\n";
  int runtimeVersion;
  AT_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
  std::cout << "The CUDA runtime version is " << runtimeVersion << "\n";
  int version;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  std::cout << "The driver version is " << version << "\n";
}

} // namespace slam_components