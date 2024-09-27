#include "components/network/superpoint/superpoint_impl.h"
#include "utils/log_utils.h"
#include <tuple>

SuperPointImpl::SuperPointImpl()
    : relu(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
      pool(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))) {
  int c1 = 64, c2 = 64, c3 = 128, c4 = 128, c5 = 256;
  int descriptor_dim = 256;

  // 初始化卷积层
  conv1a = register_module(
      "conv1a", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1)));
  conv1b = register_module(
      "conv1b", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)));
  conv2a = register_module(
      "conv2a", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)));
  conv2b = register_module(
      "conv2b", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)));
  conv3a = register_module(
      "conv3a", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)));
  conv3b = register_module(
      "conv3b", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)));
  conv4a = register_module(
      "conv4a", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)));
  conv4b = register_module(
      "conv4b", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)));

  // 初始化Pa和Pb
  convPa = register_module(
      "convPa", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
  convPb = register_module(
      "convPb", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)));

  // 初始化Da和Db
  convDa = register_module(
      "convDa", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
  convDb = register_module(
      "convDb",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(c5, descriptor_dim, 1)
                            .stride(1)
                            .padding(0)));
}

// maxPool function using max pooling operation
torch::Tensor maxPool(torch::Tensor x, int nms_radius) {
  int kernel_size = nms_radius * 2 + 1;
  // 直接传递 kernel_size, stride 和 padding
  return torch::max_pool2d(x,
                           /* kernel_size */ {kernel_size, kernel_size},
                           /* stride */ {1, 1},
                           /* padding */ {nms_radius, nms_radius});
}

torch::Tensor rgb2Grayscale(const torch::Tensor &image) {
  if (image.size(1) == 3) {
    return 0.2989 * image.select(1, 0) + // R 通道
           0.5870 * image.select(1, 1) + // G 通道
           0.1140 * image.select(1, 2);  // B 通道
  }
  return image;
}

// simpleNMS function to perform Non-maximum suppression (NMS)
torch::Tensor simpleNMS(torch::Tensor scores, int nms_radius) {
  TORCH_CHECK(nms_radius >= 0, "nms_radius must be non-negative");

  torch::Tensor zeros = torch::zeros_like(scores);
  torch::Tensor max_mask = scores == maxPool(scores, nms_radius);

  for (int i = 0; i < 2; i++) {
    torch::Tensor supp_mask =
        maxPool(max_mask.to(torch::kFloat32), nms_radius) > 0;
    torch::Tensor supp_scores = torch::where(supp_mask, zeros, scores);
    torch::Tensor new_max_mask =
        supp_scores == maxPool(supp_scores, nms_radius);
    max_mask = max_mask | (new_max_mask & (~supp_mask));
  }

  return torch::where(max_mask, scores, zeros);
}

std::tuple<torch::Tensor, torch::Tensor>
topKKeypoints(torch::Tensor keypoints, torch::Tensor scores, int k) {
  // 如果 k 大于或等于 keypoints 的数量，直接返回所有 keypoints 和 scores
  if (k >= keypoints.size(0)) {
    return std::make_tuple(keypoints, scores);
  }

  // 使用 torch::topk 获取 top k 的 scores 及其对应的索引
  // dim=0, largest=true, sorted=true
  auto topk_result = torch::topk(scores, k, 0, true, true);
  torch::Tensor top_scores = std::get<0>(topk_result); // top k scores
  torch::Tensor indices = std::get<1>(topk_result);    // top k 对应的索引

  // 根据 top k 的索引提取相应的 keypoints
  torch::Tensor top_keypoints = keypoints.index_select(0, indices);

  // 返回 keypoints 和 scores
  return std::make_tuple(top_keypoints, top_scores);
}

torch::Tensor sampleDescriptors(torch::Tensor keypoints,
                                torch::Tensor descriptors, int s = 8) {
  // 获取 descriptors 的形状信息
  auto sizes = descriptors.sizes();
  int64_t b = sizes[0]; // batch size
  int64_t c = sizes[1]; // channels
  int64_t h = sizes[2]; // height
  int64_t w = sizes[3]; // width

  // 调整 keypoints 坐标
  keypoints = keypoints - s / 2.0 + 0.5;
  keypoints /= torch::tensor({(w * s - s / 2.0 - 0.5), (h * s - s / 2.0 - 0.5)})
                   .to(keypoints.device())
                   .unsqueeze(0);

  // 将 keypoints 范围归一化到 (-1, 1)
  keypoints = keypoints * 2 - 1;

  // 使用 grid_sample 函数在描述符位置进行双线性插值
  descriptors = torch::nn::functional::grid_sample(
      descriptors,
      keypoints.view({b, 1, -1, 2}), // 调整 keypoints 形状为 (B, 1, N, 2)
      torch::nn::functional::GridSampleFuncOptions()
          .mode(torch::kBilinear)
          .align_corners(true));

  // 归一化描述符
  descriptors = torch::nn::functional::normalize(
      descriptors.view({b, c, -1}),
      torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));

  return descriptors;
}

// forward函数，执行前向传播
std::tuple<torch::Tensor, torch::Tensor>
SuperPointImpl::forward(torch::Tensor x, int nms_radius, int max_num_keypoints,
                        float detection_threshold, int remove_borders) {
  x = rgb2Grayscale(x);

  SPDLOG_INFO("SuperPoint forward");
  x = relu(conv1a(x));
  x = relu(conv1b(x));
  x = pool(x);
  x = relu(conv2a(x));
  x = relu(conv2b(x));
  x = pool(x);
  x = relu(conv3a(x));
  x = relu(conv3b(x));
  x = pool(x);
  x = relu(conv4a(x));
  x = relu(conv4b(x));

  // Apply ReLU activation and convPa
  torch::Tensor cPa = relu(convPa(x));

  // Apply convPb and softmax
  torch::Tensor scores = convPb(cPa);
  scores = torch::nn::functional::softmax(scores, 1).index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, -1)});

  // Get shape dimensions
  int64_t b = scores.size(0); // batch size
  int64_t h = scores.size(2); // height
  int64_t w = scores.size(3); // width

  // Permute and reshape scores
  scores = scores.permute({0, 2, 3, 1})
               .reshape({b, h, w, 8, 8})
               .permute({0, 1, 3, 2, 4})
               .reshape({b, h * 8, w * 8});

  // Apply simple NMS
  scores = simpleNMS(scores, nms_radius);

  // Discard keypoints near the image borders
  if (remove_borders > 0) {
    int64_t h1 = scores.size(1);
    int64_t w1 = scores.size(2);
    scores.index({torch::indexing::Slice(),
                  torch::indexing::Slice(0, remove_borders)}) = -1;
    scores.index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, remove_borders)}) = -1;
    scores.index({torch::indexing::Slice(),
                  torch::indexing::Slice(h1 - remove_borders,
                                         torch::indexing::None)}) = -1;
    scores.index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(w1 - remove_borders,
                                         torch::indexing::None)}) = -1;
  }

  // Extract keypoints based on threshold
  std::vector<torch::Tensor> best_kp =
      torch::where(scores > detection_threshold);
  torch::Tensor indices0 = best_kp[0]; // batch indices
  torch::Tensor indices1 = best_kp[1]; // y indices
  torch::Tensor indices2 = best_kp[2]; // x indices

  // Filter scores based on extracted keypoints
  scores = scores.index({indices0, indices1, indices2});

  // Separate into batches
  std::vector<torch::Tensor> keypoints;
  std::vector<torch::Tensor> scores_list;

  for (int i = 0; i < b; ++i) {
    auto kp_batch =
        torch::stack({best_kp[1], best_kp[2]}, -1).index({best_kp[0] == i});
    keypoints.push_back(kp_batch);
    auto score_batch = scores.index({best_kp[0] == i});
    scores_list.push_back(score_batch);
  }

  // Keep the top k keypoints with the highest scores
  if (max_num_keypoints > 0) {
    std::vector<torch::Tensor> keypoints_list, scores_top_list;
    for (size_t i = 0; i < keypoints.size(); ++i) {
      auto [kp, sc] =
          topKKeypoints(keypoints[i], scores_list[i], max_num_keypoints);
      keypoints_list.push_back(kp);
      scores_top_list.push_back(sc);
    }
    keypoints = keypoints_list;
    scores_list = scores_top_list;
  }

  // Convert (h, w) to (x, y)
  for (auto &kp : keypoints) {
    kp = torch::flip(kp, {1}).to(torch::kFloat32);
  }

  // Compute the dense descriptors
  torch::Tensor cDa = relu(convDa(x));
  torch::Tensor descriptors = convDb(cDa);
  descriptors = torch::nn::functional::normalize(
      descriptors, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

  std::vector<torch::Tensor> processed_descriptors;

  for (size_t i = 0; i < keypoints.size(); ++i) {
    torch::Tensor k = keypoints[i].unsqueeze(0);
    torch::Tensor d = descriptors[i].unsqueeze(0);
    torch::Tensor sampled_desc = sampleDescriptors(k, d, 8);
    processed_descriptors.push_back(sampled_desc[0]);
  }

  return std::make_tuple(
      torch::stack(keypoints, 0),
      torch::stack(processed_descriptors, 0).transpose(-1, -2).contiguous());
}
