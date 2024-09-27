#pragma once

#include <iostream>
#include <memory>
#include <torch/torch.h>

struct SuperPointImpl : public torch::nn::Module {
  using Ptr = std::shared_ptr<SuperPointImpl>;

  // 定义各个网络层
  torch::nn::Conv2d conv1a{nullptr}, conv1b{nullptr};
  torch::nn::Conv2d conv2a{nullptr}, conv2b{nullptr};
  torch::nn::Conv2d conv3a{nullptr}, conv3b{nullptr};
  torch::nn::Conv2d conv4a{nullptr}, conv4b{nullptr};

  torch::nn::Conv2d convPa{nullptr}, convPb{nullptr};
  torch::nn::Conv2d convDa{nullptr}, convDb{nullptr};

  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d pool{nullptr};

  // 构造函数，初始化各个层
  SuperPointImpl();

  // forward函数，执行前向传播
  std::tuple<torch::Tensor, torch::Tensor>
  forward(torch::Tensor x, int nms_radius = 4, int max_num_keypoints = -1,
          float detection_threshold = 0.0005, int remove_borders = 4);
};

// 定义模型的别名，方便后续使用
TORCH_MODULE(SuperPoint);
