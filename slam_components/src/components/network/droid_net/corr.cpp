#include "components/network/droid_net/corr.h"
#include "utils/log_utils.h"

namespace slam_components {

torch::Tensor CorrSampler::forward(torch::autograd::AutogradContext *ctx,
                                   torch::Tensor volume, torch::Tensor coords,
                                   int radius) {
  ctx->save_for_backward({volume, coords});
  ctx->saved_data["radius"] = radius;

  auto corr = corr_index_forward(volume, coords, radius);

  return corr[0];
}

torch::autograd::tensor_list
CorrSampler::backward(torch::autograd::AutogradContext *ctx,
                      torch::autograd::tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  torch::Tensor volume = saved[0];
  torch::Tensor coords = saved[1];
  int radius = ctx->saved_data["radius"].toInt();

  torch::Tensor grad_output = grad_outputs[0].contiguous();

  auto grad_volume = corr_index_backward(volume, coords, grad_output, radius);

  return {grad_volume[0], torch::Tensor(), torch::Tensor()};
}

CorrBlock::CorrBlock(torch::Tensor fmap1, torch::Tensor fmap2, int num_levels,
                     int radius)
    : num_levels(num_levels), radius(radius) {
  corr_pyramid = std::vector<torch::Tensor>();

  SPDLOG_INFO("fmap1 size {}", fmap1.sizes());
  SPDLOG_INFO("fmap2 size {}", fmap2.sizes());

  // all pairs correlation
  torch::Tensor corr = CorrBlock::corr(fmap1, fmap2);

  SPDLOG_INFO("corr size {}", corr.sizes());

  auto sizes = corr.sizes();
  int64_t batch = sizes[0];
  int64_t num = sizes[1];
  int64_t h1 = sizes[2];
  int64_t w1 = sizes[3];
  int64_t h2 = sizes[4];
  int64_t w2 = sizes[5];

  corr = corr.reshape({batch * num * h1 * w1, 1, h2, w2});

  SPDLOG_INFO("corr size {}", corr.sizes());

  for (int i = 0; i < num_levels; ++i) {
    corr_pyramid.push_back(
        corr.view({batch * num, h1, w1, h2 / (1 << i), w2 / (1 << i)}));
    corr = torch::avg_pool2d(corr, 2, 2);
  }
}

torch::Tensor CorrBlock::operator()(torch::Tensor coords) {
  std::vector<torch::Tensor> out_pyramid;
  auto sizes = coords.sizes();
  int64_t batch = sizes[0];
  int64_t num = sizes[1];
  int64_t ht = sizes[2];
  int64_t wd = sizes[3];

  coords = coords.permute({0, 1, 4, 2, 3})
               .contiguous()
               .view({batch * num, 2, ht, wd});

  for (int i = 0; i < num_levels; ++i) {
    torch::Tensor corr =
        CorrSampler::apply(corr_pyramid[i], coords / std::pow(2, i), radius);
    out_pyramid.push_back(corr.view({batch, num, -1, ht, wd}));
  }

  return torch::cat(out_pyramid, 2);
}

torch::Tensor CorrBlock::corr(torch::Tensor fmap1, torch::Tensor fmap2) {
  auto sizes = fmap1.sizes();
  int64_t batch = sizes[0];
  int64_t num = sizes[1];
  int64_t dim = sizes[2];
  int64_t ht = sizes[3];
  int64_t wd = sizes[4];

  fmap1 = fmap1.reshape({batch * num, dim, ht * wd}) / 4.0;
  fmap2 = fmap2.reshape({batch * num, dim, ht * wd}) / 4.0;

  torch::Tensor corr = torch::matmul(fmap1.transpose(1, 2), fmap2);

  return corr.view({batch, num, ht, wd, ht, wd});
}

} // namespace slam_components