#include "sparse_map/frame.h"
#include "sparse_map/detector.h"

void Frame::addData(
    const std::vector<cv::Mat> &_imgs,
    const std::vector<std::vector<Eigen::Vector2d>> &_keypoints,
    const std::vector<std::vector<Eigen::Vector3d>> &_bearings,
    const std::vector<cv::Mat> &_descriptors) {
  assert(_imgs.size() == _bearings.size());
  assert(_keypoints.size() == _bearings.size());
  cam_num_ = _imgs.size();
  Tcw_.resize(cam_num_);

  keypoints_ = _keypoints;
  bearings_ = _bearings;
  descriptors_ = _descriptors;

  feature_ids_.resize(cam_num_);
  for (size_t i = 0; i < cam_num_; ++i) {
    feature_ids_[i].resize(keypoints_[i].size(), -1);
    imgs_.push_back(_imgs[i].clone());
  }
}

void Frame::extractFeature(const std::vector<cv::Mat> &_imgs,
                           std::string detector_type) {
  cam_num_ = _imgs.size();
  Tcw_.resize(cam_num_);

  keypoints_.resize(cam_num_);
  bearings_.resize(cam_num_);
  descriptors_.resize(cam_num_);
  imgs_.clear();
  for (size_t i = 0; i < cam_num_; ++i) {
    imgs_.push_back(_imgs[i].clone());
  }

  Detector detector;
  for (size_t i = 0; i < cam_num_; ++i) {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptors;
    if (detector_type == "ORB") {
      detector.detectORB(_imgs[i], kpts, descriptors);
    } else if (detector_type == "SIFT") {
      detector.detectSIFT(_imgs[i], kpts, descriptors);
    } else {
      std::cerr << "Unknown detector type: " << detector_type << std::endl;
      return;
    }
    keypoints_[i].resize(kpts.size());
    for (size_t j = 0; j < kpts.size(); ++j) {
      keypoints_[i][j] = Eigen::Vector2d(kpts[j].pt.x, kpts[j].pt.y);
    }
    descriptors_[i] = descriptors.clone();
  }

  feature_ids_.resize(cam_num_);
  for (size_t i = 0; i < cam_num_; ++i) {
    feature_ids_[i].resize(keypoints_[i].size(), -1);
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