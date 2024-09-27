#include "components/tracking.h"
#include "components/network/droid/corr.h"
#include "components/network/utils.h"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "components/utils/type_utils.h"
#include "utils/log_utils.h"

namespace slam_components {

bool Tracking::initialize(const cv::FileNode &node, const Network::Ptr &network,
                          const Calibration::Ptr &calibration,
                          const foxglove_viz::Visualizer::Ptr &viz_server) {
  network_ = network;
  calibration_ = calibration;
  viz_server_ = viz_server;

  if (node["motion_filter_thresh"].empty()) {
    SPDLOG_CRITICAL("motion_filter_thresh is not provided");
    return false;
  } else {
    node["motion_filter_thresh"] >> motion_filter_thresh_;
  }

  if (node["motion_model"].empty()) {
    SPDLOG_CRITICAL("motion_model is not provided");
    return false;
  } else {
    node["motion_model"] >> motion_model_;
  }

  // hyper parameters
  MEAN = torch::tensor({0.485, 0.456, 0.406})
             .view({3, 1, 1})
             .to(network_->droid_net_->device_);
  STDV = torch::tensor({0.229, 0.224, 0.225})
             .view({3, 1, 1})
             .to(network_->droid_net_->device_);

  printSetting();

  return true;
}

void Tracking::printSetting() {
  SPDLOG_INFO(
      "\nTracking Setting: \n - motion_filter_thresh: {} \n - motion_model: {}",
      motion_filter_thresh_, motion_model_);
}

Frame::Ptr Tracking::process(const TrackingInput &input) {
  static double process_total_time = 0.0;
  static int process_total_count = 0;

  TimeStatistics tracking_statistics("Tracking");

  tracking_statistics.tic();

  curr_frame_.reset(
      new Frame(next_frame_id_++, input.camera_data->images_.size()));
  curr_frame_->setTimestamp(input.camera_data->timestamp_);
  curr_frame_->addData(input.camera_data->images_);

  tracking_statistics.tocAndTic("initialize frame");

  estimateInitialPose();

  tracking_statistics.tocAndTic("estimate initial pose");

  this->propressImage(curr_frame_);

  tracking_statistics.tocAndTic("propress image");

  if (judgeKeyframe()) {
    tracking_statistics.tocAndTic("judge keyframe");
    curr_frame_->setKeyFrame(true);

    extractDenseFeature(curr_frame_);

    tracking_statistics.tocAndTic("extract dense feature");

    extractSparseFeature(curr_frame_);

    tracking_statistics.tocAndTic("extract sparse feature");

    publishRawImage();

    tracking_statistics.tocAndTic("publish raw image");

    last_keyframe_ = curr_frame_;
    last_frame_ = curr_frame_;

    tracking_statistics.logTimeStatistics(curr_frame_->id());

    return curr_frame_;
  } else {
    tracking_statistics.tocAndTic("judge nonkeyframe");
  }

  last_frame_ = curr_frame_;

  process_total_time +=
      tracking_statistics.logTimeStatistics(curr_frame_->id());
  process_total_count += 1;

  SPDLOG_INFO("Tracking process average time: {} ms",
              process_total_time / process_total_count);

  return nullptr;
}

void Tracking::estimateInitialPose() {
  if (last_frame_ == nullptr) {
    return;
  }

  if (motion_model_ == "fixed") {
    curr_frame_->setBodyPose(last_frame_->getBodyPose());
    curr_frame_->setVelocity(last_frame_->getVelocity());
  } else if (motion_model_ == "constant_velocity") {
    estimatePoseByConstantVelocity();
  } else if (motion_model_ == "imu") {
    if (!is_imu_initial_) {
      estimatePoseByConstantVelocity();
      return;
    }

    estimatePoseByIMU();
  } else {
    estimatePoseByConstantVelocity();
    SPDLOG_WARN("Unknown motion model: {}", motion_model_);
  }
}

void Tracking::estimatePoseByConstantVelocity() {}

void Tracking::estimatePoseByIMU() {}

bool Tracking::judgeKeyframe() {
  if (last_keyframe_ == nullptr) {
    return true;
  }

  // SPDLOG_INFO("judge keyframe");
  extractDenseFeature(curr_frame_, true);
  // SPDLOG_INFO("extract dense feature");
  if (motionFilter()) {
    return true;
  }

  return false;
}

void Tracking::propressImage(const Frame::Ptr &frame) {
  std::vector<torch::Tensor> images;
  for (size_t i = 0; i < frame->imgs().size(); ++i) {
    cv::Mat image = frame->imgs()[i].clone();

    if (image.channels() == 1) {
      cv::Mat temp;
      cv::cvtColor(image, temp, cv::COLOR_GRAY2BGR);
      image = temp;
    }

    torch::Tensor img_tensor =
        torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
    images.push_back(img_tensor);
  }

  torch::Tensor images_tensor =
      torch::stack(images)
          .permute({0, 3, 1, 2})
          .to(network_->droid_net_->device_, torch::kFloat32);
  images_tensor =
      images_tensor.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})})
          .to(network_->droid_net_->device_) /
      255.0;

  frame->images_superpoint_torch_ = images_tensor;
  frame->images_droid_torch_ = images_tensor.sub(MEAN).div(STDV).unsqueeze(0);
}

void Tracking::extractDenseFeature(const Frame::Ptr &frame,
                                   bool only_feature_map) {
  // 禁用梯度计算
  // torch::NoGradGuard no_grad;

  // 自动混合精度推理
  at::autocast::set_autocast_cache_enabled(true);

  // 检查图像是否已预处理
  if (!frame->images_droid_torch_.defined()) {
    SPDLOG_ERROR("images_droid_torch_ is not defined");
    return;
  }

  // 提取 feature map
  if (!frame->feature_map_.defined()) {
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(frame->images_droid_torch_);
    frame->feature_map_ =
        this->network_->droid_net_->droid_fnet_.forward(input_tensors)
            .toTensor()
            .squeeze(0);
  }

  // 如果需要提取 context map 和 net map
  if (!only_feature_map &&
      (!frame->context_map_.defined() || !frame->net_map_.defined())) {
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(frame->images_droid_torch_);
    torch::Tensor output =
        this->network_->droid_net_->droid_cnet_.forward(input_tensors)
            .toTensor();
    auto tensors = output.split_with_sizes({128, 128}, 2);

    // 保存 net_map 和 context_map
    frame->net_map_ = tensors[0].tanh().squeeze(0);
    frame->context_map_ = tensors[1].relu().squeeze(0);

    frame->images_droid_torch_.reset();
  }

  at::autocast::clear_cache();
  at::autocast::set_autocast_cache_enabled(false);
}

bool Tracking::motionFilter() {
  // SPDLOG_INFO("motion filter");

  // 禁用梯度计算
  // torch::NoGradGuard no_grad;

  // 计算 ht 和 wd
  int64_t ht =
      curr_frame_->images_droid_torch_.size(-2) / image_downsample_scale_;
  int64_t wd =
      curr_frame_->images_droid_torch_.size(-1) / image_downsample_scale_;

  // 自动混合精度推理
  at::autocast::set_autocast_cache_enabled(true);

  // 生成坐标网格
  torch::Tensor coords0 = getCoordsGrid(ht, wd, network_->droid_net_->device_)
                              .unsqueeze(0)
                              .unsqueeze(0);

  // SPDLOG_INFO("getCoordsGrid");

  // SPDLOG_INFO("coords0 shape: {}", coords0.sizes());

  // SPDLOG_INFO("last_keyframe_->feature_map_ shape: {}",
  //             last_keyframe_->feature_map_.sizes());

  // SPDLOG_INFO("curr_frame_->feature_map_ shape: {}",
  //             curr_frame_->feature_map_.sizes());

  // 计算相关性
  torch::Tensor corr = CorrBlock(
      last_keyframe_->feature_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0),
      curr_frame_->feature_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0))(coords0);

  // SPDLOG_INFO("CorrBlock");

  // SPDLOG_INFO("corr shape: {}", corr.sizes());

  // SPDLOG_INFO("last_keyframe_->net_map_ shape: {}",
  //             last_keyframe_->net_map_.sizes());

  // SPDLOG_INFO("last_keyframe_->context_map_ shape: {}",
  //             last_keyframe_->context_map_.sizes());

  // 使用 droid_update 计算
  std::vector<torch::jit::IValue> input_tensors;
  input_tensors.push_back(
      last_keyframe_->net_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0));
  input_tensors.push_back(
      last_keyframe_->context_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0));
  input_tensors.push_back(corr);

  // SPDLOG_INFO("push data");

  auto output =
      this->network_->droid_net_->droid_update_.forward(input_tensors);

  // SPDLOG_INFO("update");

  auto outputs = output.toTuple();

  at::autocast::clear_cache();
  at::autocast::set_autocast_cache_enabled(false);

  torch::Tensor delta = outputs->elements()[1].toTensor();

  // SPDLOG_INFO("output");

  if (delta.norm(2, -1).mean().item<float>() > motion_filter_thresh_) {
    return true;
  }

  return false;
}

void Tracking::extractSparseFeature(const Frame::Ptr &frame) {
  if (0)
    frame->extractFeature();
  else {
    std::vector<std::vector<Eigen::Vector2d>> keypoints_vec;
    std::vector<cv::Mat> descriptors_vec;
    for (size_t i = 0; i < frame->imgs().size(); ++i) {
      auto sparse_result = network_->superpoint_net_->net_->forward(
          frame->images_superpoint_torch_[i].unsqueeze(0));

      torch::Tensor keypoint = std::get<0>(sparse_result);
      torch::Tensor descriptor = std::get<1>(sparse_result);

      std::vector<Eigen::Vector2d> keypoints;
      cv::Mat descriptors;
      // TODO: convert tensor to cv::Mat
    }
    frame->images_superpoint_torch_.reset();
  }
}

void Tracking::publishRawImage() {
  if (viz_server_) {
    cv::Mat raw_img = curr_frame_->drawRawImage();
    viz_server_->showImage("curr_keyframe/raw_images",
                           int64_t(curr_frame_->timestamp() * 1e6), raw_img);
  }
}

} // namespace slam_components