#include "components/tracking.h"
#include "components/network/droid_net/corr.h"
#include "components/network/droid_net/utils.h"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "utils/log_utils.h"

namespace slam_components {

bool Tracking::initialize(const cv::FileNode &node,
                          const DroidNet::Ptr &droid_net,
                          const Calibration::Ptr &calibration,
                          const foxglove_viz::Visualizer::Ptr &viz_server) {
  droid_net_ = droid_net;
  calibration_ = calibration;
  viz_server_ = viz_server;

  if (node["motion_filter_thresh"].empty()) {
    std::cerr << "Error: motion_filter_thresh is not provided\n";
    return false;
  } else {
    node["motion_filter_thresh"] >> motion_filter_thresh_;
  }

  if (node["motion_model"].empty()) {
    std::cerr << "Error: motion_model is not provided\n";
    return false;
  } else {
    node["motion_model"] >> motion_model_;
  }

  // hyper parameters
  MEAN = torch::tensor({0.485, 0.456, 0.406})
             .view({3, 1, 1})
             .to(droid_net_->device_);
  STDV = torch::tensor({0.229, 0.224, 0.225})
             .view({3, 1, 1})
             .to(droid_net_->device_);

  return true;
}

bool Tracking::track(const TrackingInput &input) {

  curr_frame_.reset(
      new Frame(next_frame_id_++, input.camera_data->images_.size()));
  curr_frame_->setTimestamp(input.camera_data->timestamp_);
  curr_frame_->addData(input.camera_data->images_);

  SPDLOG_INFO("Tracking frame: {}", curr_frame_->id());

  publishRawImage();

  SPDLOG_INFO("publish raw image");

  // return true;

  estimateInitialPose();

  SPDLOG_INFO("estimate initial pose");

  estimateInitialIdepth();

  SPDLOG_INFO("estimate initial idepth");

  if (judgeKeyframe()) {
    SPDLOG_INFO("keyframe");
    extractDenseFeature(curr_frame_);
    SPDLOG_INFO("extract dense feature");
    extractSparseFeature(curr_frame_);
    SPDLOG_INFO("extract sparse feature");
    curr_frame_->setKeyFrame(true);
    last_keyframe_ = curr_frame_;
  } else {
    SPDLOG_INFO("not keyframe");
  }

  last_frame_ = curr_frame_;

  return true;
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
    std::cerr << "Unknown motion model: {" << motion_model_ << "}\n";
  }
}

void Tracking::estimatePoseByConstantVelocity() {}

void Tracking::estimatePoseByIMU() {}

void Tracking::estimateInitialIdepth() {}

bool Tracking::judgeKeyframe() {
  if (last_keyframe_ == nullptr) {
    return true;
  }

  SPDLOG_INFO("judge keyframe");
  extractDenseFeature(curr_frame_, true);
  SPDLOG_INFO("extract dense feature");
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

  torch::Tensor images_tensor = torch::stack(images)
                                    .permute({0, 3, 1, 2})
                                    .to(droid_net_->device_, torch::kFloat32);
  images_tensor =
      images_tensor.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})})
          .to(droid_net_->device_) /
      255.0;

  frame->images_lightglue_torch_ = images_tensor;
  frame->images_droid_torch_ = images_tensor.sub(MEAN).div(STDV).unsqueeze(0);
}

void Tracking::extractDenseFeature(const Frame::Ptr &frame,
                                   bool only_feature_map) {
  // 禁用梯度计算
  torch::NoGradGuard no_grad;

  // 自动混合精度推理
  at::autocast::set_autocast_cache_enabled(true);

  // 检查图像是否已预处理
  if (!frame->images_droid_torch_.defined()) {
    this->propressImage(frame);
  }

  // 提取 feature map
  if (!frame->feature_map_.defined()) {
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(frame->images_droid_torch_);
    frame->feature_map_ = this->droid_net_->droid_fnet_.forward(input_tensors)
                              .toTensor()
                              .squeeze(0);
  }

  // 如果需要提取 context map 和 net map
  if (!only_feature_map &&
      (!frame->context_map_.defined() || !frame->net_map_.defined())) {
    std::vector<torch::jit::IValue> input_tensors;
    input_tensors.push_back(frame->images_droid_torch_);
    torch::Tensor output =
        this->droid_net_->droid_cnet_.forward(input_tensors).toTensor();
    auto tensors = output.split_with_sizes({128, 128}, 2);

    // 保存 net_map 和 context_map
    frame->net_map_ = tensors[0].tanh().squeeze(0);
    frame->context_map_ = tensors[1].relu().squeeze(0);
  }

  at::autocast::clear_cache();
  at::autocast::set_autocast_cache_enabled(false);;
}

bool Tracking::motionFilter() {
  SPDLOG_INFO("motion filter");

  // 禁用梯度计算
  torch::NoGradGuard no_grad;

  // 计算 ht 和 wd
  int64_t ht =
      curr_frame_->images_droid_torch_.size(-2) / image_downsample_scale_;
  int64_t wd =
      curr_frame_->images_droid_torch_.size(-1) / image_downsample_scale_;

  // 自动混合精度推理
  at::autocast::set_autocast_cache_enabled(true);

  // 生成坐标网格
  torch::Tensor coords0 =
      getCoordsGrid(ht, wd, droid_net_->device_).unsqueeze(0).unsqueeze(0);

  SPDLOG_INFO("getCoordsGrid");

  SPDLOG_INFO("coords0 shape: {}", coords0.sizes());

  SPDLOG_INFO("last_keyframe_->feature_map_ shape: {}",
              last_keyframe_->feature_map_.sizes());

  SPDLOG_INFO("curr_frame_->feature_map_ shape: {}",
              curr_frame_->feature_map_.sizes());

  // 计算相关性
  torch::Tensor corr = CorrBlock(
      last_keyframe_->feature_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0),
      curr_frame_->feature_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0))(coords0);

  SPDLOG_INFO("CorrBlock");

  SPDLOG_INFO("corr shape: {}", corr.sizes());

  SPDLOG_INFO("last_keyframe_->net_map_ shape: {}",
              last_keyframe_->net_map_.sizes());

  SPDLOG_INFO("last_keyframe_->context_map_ shape: {}",
              last_keyframe_->context_map_.sizes());

  // 使用 droid_update 计算
  std::vector<torch::jit::IValue> input_tensors;
  input_tensors.push_back(
      last_keyframe_->net_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0));
  input_tensors.push_back(
      last_keyframe_->context_map_.index({torch::indexing::Slice(0, 1)})
          .unsqueeze(0));
  input_tensors.push_back(corr);

  SPDLOG_INFO("push data");

  auto output = this->droid_net_->droid_update_.forward(input_tensors);

  SPDLOG_INFO("update");

  auto outputs = output.toTuple();

  at::autocast::clear_cache();
  at::autocast::set_autocast_cache_enabled(false);;

  torch::Tensor delta = outputs->elements()[1].toTensor();

  SPDLOG_INFO("output");

  if (delta.norm(2, -1).mean().item<float>() > motion_filter_thresh_) {
    return true;
  }

  return false;
}

void Tracking::extractSparseFeature(const Frame::Ptr &frame) {
  frame->extractFeature();
}

void Tracking::publishRawImage() {
  if (viz_server_) {
    cv::Mat raw_img = curr_frame_->drawRawImage();
    viz_server_->showImage("raw_img", int64_t(curr_frame_->timestamp() * 1e6),
                           raw_img);
  }
}

} // namespace slam_components