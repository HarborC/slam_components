#pragma once

#include <deque>
#include <mutex>

#include "calibration/calibration.h"
#include "components/network/droid_net/droid_net.h"
#include "components/sensor_data.h"
#include "components/tracking.h"

namespace slam_components {

class System {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<System>;

public:
  System() = default;
  ~System() {}

  void feedIMU(const double &timestamp, const Eigen::Vector3d &angular_velocity,
               const Eigen::Vector3d &acceleration);

  void feedCamera(const double &timestamp, const std::vector<cv::Mat> &images);

  bool initialize(const std::string &config_path);

private:
  bool initializeNetwork(const cv::FileNode &node);
  bool initializeCalibration(const cv::FileNode &node);

  bool getTrackingInput(TrackingInput& input);

private:
  DroidNet::Ptr droid_net_;
  Calibration::Ptr calibration_;
  Tracking::Ptr tracking_;

  std::deque<IMUData::Ptr> imu_deque;
  std::deque<CameraData::Ptr> camera_deque;

  std::mutex feed_mtx;
};

} // namespace slam_components