#include "sparse_map/frame.h"

void Frame::addData(
    const std::vector<cv::Mat> &_imgs,
    const std::vector<std::vector<Eigen::Vector2d>> &_keypoints,
    const std::vector<std::vector<Eigen::Vector3d>> &_bearings) {
  assert(_imgs.size() == _bearings.size());
  assert(_keypoints.size() == _bearings.size());
  cam_num_ = _imgs.size();
  Tcw_.resize(cam_num_);

  keypoints_ = _keypoints;
  bearings_ = _bearings;

  feature_ids_.resize(cam_num_);
  for (size_t i = 0; i < cam_num_; ++i) {
    feature_ids_[i].resize(keypoints_[i].size(), -1);
    imgs_.push_back(_imgs[i].clone());
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
  }
  return img;
}