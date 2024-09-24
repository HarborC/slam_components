#pragma once

#include <torch/cuda.h>
#include <torch/torch.h>

namespace slam_components {

torch::Tensor getCoordsGrid(int64_t ht, int64_t wd,
                            torch::Device device = torch::kCPU);

}