#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <torch/torch.h>

namespace slam_components {

torch::Tensor getCoordsGrid(int64_t ht, int64_t wd,
                            torch::Device device = torch::kCPU);

void printLibtorchVersion();

void printCudaCuDNNInfo();

} // namespace slam_components