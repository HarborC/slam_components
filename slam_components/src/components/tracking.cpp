#include "components/tracking.h"

namespace slam_components {

bool Tracking::initialize(const cv::FileNode &node,
                          const DroidNet::Ptr &droid_net,
                          const Calibration::Ptr &calibration) {
  droid_net_ = droid_net;
  calibration_ = calibration;

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
  MEAN = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1}).to(droid_net_->device_);
  STDV = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1}).to(droid_net_->device_);

  return true;
}

bool Tracking::track(const TrackingInput &input) {
  

  curr_frame_.reset(new Frame(next_frame_id_++, input.images_data.size()));
  curr_frame_->setTimestamp(input.image_time);
  curr_frame_->addData(input.images_data);

  estimateInitialPose();

  estimateInitialIdepth();

  if (judgeKeyframe()) {
    last_keyframe_ = curr_frame_;
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
    
    return false;
}

void Tracking::propressImage() {
    std::vector<torch::Tensor> images;
    for (size_t i = 0; i < curr_frame_->imgs().size(); ++i) {
        cv::Mat image = curr_frame_->imgs()[i].clone();

        if (image.channels() == 1) {
            cv::Mat temp;
            cv::cvtColor(image, temp, cv::COLOR_GRAY2BGR);
            image = temp;
        }

        torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
        images.push_back(img_tensor);
    }

    torch::Tensor images_tensor = torch::stack(images).permute({0, 3, 1, 2}).to(droid_net_->device_, torch::kFloat32);
    images_tensor = images_tensor.index({torch::indexing::Slice(), torch::tensor({2, 1, 0})}).to(droid_net_->device_) / 255.0;

    curr_frame_->images_lightglue_torch = images_tensor;
    curr_frame_->images_droid_torch = images_tensor.sub(MEAN).div(STDV).unsqueeze(0);
}



} // namespace slam_components