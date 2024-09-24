#pragma once

#include <torch/torch.h>

struct CorrSampler : public torch::autograd::Function<CorrSampler> {
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor volume, torch::Tensor coords,
                               int radius);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_outputs);
};

class CorrBlock {
public:
  CorrBlock(torch::Tensor fmap1, torch::Tensor fmap2, int num_levels = 4,
            int radius = 3);

  torch::Tensor operator()(torch::Tensor coords);

  static torch::Tensor corr(torch::Tensor fmap1, torch::Tensor fmap2);

private:
  int num_levels;
  int radius;
  std::vector<torch::Tensor> corr_pyramid;
};
