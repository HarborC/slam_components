#include <iostream>
#include <torch/torch.h>

int main() {

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;

    // 检查是否支持 cuDNN
    if (torch::cuda::cudnn_is_available()) {
      std::cout << "cuDNN is available!" << std::endl;
    } else {
      std::cout << "cuDNN is not available." << std::endl;
    }
  } else {
    std::cout << "CUDA is not available." << std::endl;
  }

  // cudnnHandle_t cudnn;
  // cudnnStatus_t status = cudnnCreate(&cudnn);

  // if (status != CUDNN_STATUS_SUCCESS) {
  //     std::cerr << "cuDNN initialization failed: " <<
  //     cudnnGetErrorString(status) << std::endl; return 1;
  // }

  // int version = cudnnGetVersion();
  // std::cout << "cuDNN version: " << version << std::endl;

  // // Clean up
  // // cudnnDestroy(cudnn);

  return 0;
}
