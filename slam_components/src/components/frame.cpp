#include "components/frame.h"
#include "components/detector.h"

namespace slam_components {

Frame::Frame(FrameIDType _id, int _cam_num) : id_(_id) {
  imgs_.resize(_cam_num);
  keypoints_.resize(_cam_num);
  feature_ids_.resize(_cam_num);
  bearings_.resize(_cam_num);
  descriptors_.resize(_cam_num);
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
                    const std::vector<std::vector<Eigen::Vector2d>> &_keypoints,
                    const std::vector<std::vector<Eigen::Vector3d>> &_bearings,
                    const std::vector<cv::Mat> &_descriptors) {
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
      feature_ids_[i].resize(keypoints_[i].size(), -1);
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
}

void Frame::extractFeature(const std::vector<cv::Mat> &_imgs,
                           std::string detector_type,
                           const std::vector<cv::Mat> &_masks) {
  assert(_imgs.size() == camNum());
  if (_masks.size()) {
    assert(_imgs.size() == _masks.size());
  }

  int cam_num = camNum();
  for (size_t cam_id = 0; cam_id < cam_num; ++cam_id) {
    imgs_[cam_id] = _imgs[cam_id].clone();
  }

  Detector detector;
  for (size_t cam_id = 0; cam_id < cam_num; ++cam_id) {
    std::vector<cv::KeyPoint> kpts;
    if (detector_type == "ORB") {
      if (_masks.empty()) {
        detector.detectORB(_imgs[cam_id], kpts, descriptors_[cam_id]);
      } else {
        detector.detectORB(_imgs[cam_id], kpts, descriptors_[cam_id],
                           _masks[cam_id]);
      }
    } else if (detector_type == "SIFT") {
      if (_masks.empty()) {
        detector.detectSIFT(_imgs[cam_id], kpts, descriptors_[cam_id]);
      } else {
        detector.detectSIFT(_imgs[cam_id], kpts, descriptors_[cam_id],
                            _masks[cam_id]);
      }
    } else {
      std::cerr << "Unknown detector type: " << detector_type << std::endl;
      return;
    }
    keypoints_[cam_id].resize(kpts.size());
    for (size_t pt_id = 0; pt_id < kpts.size(); ++pt_id) {
      keypoints_[cam_id][pt_id] =
          Eigen::Vector2d(kpts[pt_id].pt.x, kpts[pt_id].pt.y);
    }
    feature_ids_[cam_id].resize(keypoints_[cam_id].size(), -1);
  }
}

cv::Mat Frame::drawKeyPoint(const int &cam_id) {
  cv::Mat img = imgs_[cam_id].clone();
  if (img.channels() == 1) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  }

  for (size_t i = 0; i < keypoints_[cam_id].size(); ++i) {
    cv::circle(img,
               cv::Point(keypoints_[cam_id][i](0), keypoints_[cam_id][i](1)), 2,
               cv::Scalar(0, 255, 0), 2);
  }
  return img;
}

cv::Mat Frame::drawMatchedKeyPoint(const int &cam_id) {
  cv::Mat img = imgs_[cam_id].clone();
  if (img.channels() == 1) {
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  }

  for (size_t i = 0; i < keypoints_[cam_id].size(); ++i) {
    if (feature_ids_[cam_id][i] < 0) {
      continue;
    }
    cv::circle(img,
               cv::Point(keypoints_[cam_id][i](0), keypoints_[cam_id][i](1)), 2,
               cv::Scalar(0, 255, 0), 2);
    cv::putText(img, std::to_string(feature_ids_[cam_id][i]),
                cv::Point(keypoints_[cam_id][i](0), keypoints_[cam_id][i](1)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
  return img;
}

} // namespace slam_components