#pragma once

#include <deque>
#include <mutex>
#include <thread>

#include "calibration/calibration.h"
#include "components/network/droid_net/droid_net.h"
#include "components/sensor_data.h"
#include "components/tracking.h"

#include "foxglove/visualizer.h"

namespace slam_components {

class System {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<System>;

public:
  System() = default;
  ~System();

  void feedIMU(const double &timestamp, const Eigen::Vector3d &angular_velocity,
               const Eigen::Vector3d &acceleration);

  void feedCamera(const double &timestamp, const std::vector<cv::Mat> &images);

  bool initialize(const std::string &config_path);

private:
  bool initializeNetwork(const cv::FileNode &node);
  bool initializeCalibration(const cv::FileNode &node);
  bool initializeViz(const cv::FileNode &node);

  bool getTrackingInput(TrackingInput &input);

  void processLoop();

private:
  DroidNet::Ptr droid_net_;
  Calibration::Ptr calibration_;
  Tracking::Ptr tracking_;
  foxglove_viz::Visualizer::Ptr viz_server;

  std::deque<IMUData::Ptr> imu_buf_;
  std::deque<CameraData::Ptr> camera_buf_;

  std::mutex feed_mtx;
  std::shared_ptr<std::thread> loop_thread;
};

} // namespace slam_components