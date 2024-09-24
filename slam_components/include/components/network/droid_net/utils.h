#pragma once

#include <torch/cuda.h>
#include <torch/torch.h>

torch::Tensor getCoordsGrid(int64_t ht, int64_t wd,
                            torch::Device device = torch::kCPU);