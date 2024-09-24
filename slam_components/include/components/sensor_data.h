#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace slam_components {

struct SensorData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double timestamp_;
};

struct IMUData : public SensorData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<IMUData>;
  Eigen::Vector3d angular_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d acceleration_ = Eigen::Vector3d::Zero();

  // Orientation (optional, if using orientation data from the IMU)
  Eigen::Quaterniond orientation_ = Eigen::Quaterniond::Identity();

  IMUData() {}
  IMUData(
      double timestamp, const Eigen::Vector3d &angular_velocity,
      const Eigen::Vector3d &acceleration,
      const Eigen::Quaterniond &orientation = Eigen::Quaterniond::Identity()) {
    timestamp_ = timestamp;
    angular_velocity_ = angular_velocity;
    acceleration_ = acceleration;
    orientation_ = orientation;
  }
};

struct CameraData : public SensorData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<CameraData>;
  std::vector<cv::Mat> images_;

  CameraData() {}
  CameraData(double timestamp, const std::vector<cv::Mat> &images) {
    timestamp_ = timestamp;
    images_ = images;
  }
};

} // namespace slam_components
