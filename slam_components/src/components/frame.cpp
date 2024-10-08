#include "components/frame.h"
#include "components/detector.h"
#include "utils/log_utils.h"

namespace slam_components {

Frame::Frame(FrameIDType _id, int _cam_num) : id_(_id) {
  imgs_.resize(_cam_num);
  keypoints_.resize(_cam_num);
  feature_ids_.resize(_cam_num);
  bearings_.resize(_cam_num);
  descriptors_.resize(_cam_num);
  masks_.resize(_cam_num);
  matched_frames_.resize(_cam_num);
  Tcw_.resize(_cam_num);
}

FrameIDType Frame::id() { return id_; }

int Frame::camNum() { return keypoints_.size(); }

void Frame::setFeatureID(const int &cam_id, const int &pt_id,
                         const FeatureIDType &ft_id) {
  feature_ids_[cam_id][pt_id] = ft_id;
}

double *Frame::getRotaionParams() { return pose_q; }

double *Frame::getTranslationParams() { return pose_t; }

double *Frame::getVelocityParams() { return velocity; }

Eigen::Matrix4d Frame::getBodyPose() {
  Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
  Twb.block<3, 3>(0, 0) =
      Eigen::Quaterniond(pose_q[3], pose_q[0], pose_q[1], pose_q[2])
          .toRotationMatrix();
  Twb.block<3, 1>(0, 3) = Eigen::Vector3d(pose_t[0], pose_t[1], pose_t[2]);

  return Twb;
}

void Frame::setBodyPose(const Eigen::Matrix4d &Twb) {
  Eigen::Quaterniond qwb = Eigen::Quaterniond(Twb.block<3, 3>(0, 0));
  Eigen::Vector3d twb = Twb.block<3, 1>(0, 3);
  pose_q[0] = qwb.x();
  pose_q[1] = qwb.y();
  pose_q[2] = qwb.z();
  pose_q[3] = qwb.w();
  pose_t[0] = twb(0);
  pose_t[1] = twb(1);
  pose_t[2] = twb(2);
}

Eigen::Vector3d Frame::getVelocity() {
  return Eigen::Vector3d(velocity[0], velocity[1], velocity[2]);
}

void Frame::setVelocity(const Eigen::Vector3d &vel) {
  velocity[0] = vel(0);
  velocity[1] = vel(1);
  velocity[2] = vel(2);
}

void Frame::addData(const std::vector<cv::Mat> &_imgs,
                    const std::vector<torch::Tensor> &_keypoints,
                    const std::vector<std::vector<Eigen::Vector3d>> &_bearings,
                    const std::vector<torch::Tensor> &_descriptors,
                    const std::vector<cv::Mat> &_masks) {
  if (_imgs.size()) {
    assert(_imgs.size() == camNum());
    for (size_t i = 0; i < _imgs.size(); ++i) {
      imgs_[i] = _imgs[i].clone();
    }
  }

  if (_keypoints.size()) {
    assert(_keypoints.size() == camNum());
    keypoints_ = _keypoints;
    feature_ids_.resize(keypoints_.size());
    for (size_t i = 0; i < feature_ids_.size(); ++i) {
      feature_ids_[i].resize(keypoints_[i].size(0), -1);
    }
  }

  if (_bearings.size()) {
    assert(_bearings.size() == camNum());
    bearings_ = _bearings;
  }

  if (_descriptors.size()) {
    assert(_descriptors.size() == camNum());
    descriptors_ = _descriptors;
  }

  if (_masks.size()) {
    assert(_masks.size() == camNum());
    for (size_t i = 0; i < _masks.size(); ++i) {
      masks_[i] = _masks[i].clone();
    }
  }
}

// void Frame::extractFeature(std::string detector_type) {
//   Detector detector;
//   for (size_t cam_id = 0; cam_id < camNum(); ++cam_id) {
//     std::vector<cv::KeyPoint> kpts;
//     if (detector_type == "ORB") {
//       if (masks_.empty()) {
//         detector.detectORB(imgs_[cam_id], kpts, descriptors_[cam_id]);
//       } else {
//         detector.detectORB(imgs_[cam_id], kpts, descriptors_[cam_id],
//                            masks_[cam_id]);
//       }
//     } else if (detector_type == "SIFT") {
//       if (masks_.empty()) {
//         detector.detectSIFT(imgs_[cam_id], kpts, descriptors_[cam_id]);
//       } else {
//         detector.detectSIFT(imgs_[cam_id], kpts, descriptors_[cam_id],
//                             masks_[cam_id]);
//       }
//     } else {
//       std::cerr << "Unknown detector type: " << detector_type << std::endl;
//       return;
//     }
//     keypoints_[cam_id].resize(kpts.size());
//     for (size_t pt_id = 0; pt_id < kpts.size(); ++pt_id) {
//       keypoints_[cam_id][pt_id] =
//           Eigen::Vector2d(kpts[pt_id].pt.x, kpts[pt_id].pt.y);
//     }
//     feature_ids_[cam_id].resize(keypoints_[cam_id].size(), -1);
//   }
// }

cv::Mat Frame::drawKeyPoint(const int &cam_id) {
  cv::Mat all_img;
  for (size_t c_i = 0; c_i < camNum(); ++c_i) {
    if (cam_id >= 0 && c_i != cam_id) {
      continue;
    }

    int keypoint_num = keypoints_[c_i].size(0);
    auto keypoints = keypoints_[c_i].accessor<float, 2>();

    SPDLOG_INFO("keypoint_num: {}", keypoint_num);

    cv::Mat img = imgs_[c_i].clone();
    if (img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < keypoint_num; ++i) {
      cv::circle(img, cv::Point(keypoints[i][0], keypoints[i][1]), 2,
                 cv::Scalar(0, 255, 0), 2);
    }

    if (all_img.empty()) {
      all_img = img;
    } else {
      cv::hconcat(all_img, img, all_img);
    }
  }

  return all_img;
}

cv::Mat Frame::drawMatchedKeyPoint(const int &cam_id) {
  cv::Mat all_img;
  for (size_t c_i = 0; c_i < camNum(); ++c_i) {
    if (cam_id >= 0 && c_i != cam_id) {
      continue;
    }

    int keypoint_num = keypoints_[c_i].size(0);
    auto keypoints = keypoints_[c_i].accessor<float, 2>();

    cv::Mat img = imgs_[c_i].clone();
    if (img.channels() == 1) {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < keypoint_num; ++i) {
      if (feature_ids_[c_i][i] < 0) {
        continue;
      }
      cv::circle(img, cv::Point(keypoints[i][0], keypoints[i][1]), 2,
                 cv::Scalar(0, 255, 0), 2);
      cv::putText(img, std::to_string(feature_ids_[c_i][i]),
                  cv::Point(keypoints[i][0], keypoints[i][1]),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    if (all_img.empty()) {
      all_img = img;
    } else {
      cv::hconcat(all_img, img, all_img);
    }
  }

  return all_img;
}

cv::Mat Frame::drawRawImage(const int &cam_id) {
  cv::Mat all_img;
  for (size_t c_i = 0; c_i < camNum(); ++c_i) {
    if (cam_id >= 0 && c_i != cam_id) {
      continue;
    }

    cv::Mat img = imgs_[c_i].clone();
    cv::putText(img, std::to_string(id_) + "-" + std::to_string(c_i),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);

    if (all_img.empty()) {
      all_img = img;
    } else {
      cv::hconcat(all_img, img, all_img);
    }
  }

  return all_img;
}

} // namespace slam_components