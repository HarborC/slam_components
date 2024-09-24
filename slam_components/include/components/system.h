#pragma once

#include "calibration/calibration.h"
#include "components/network/droid_net/droid_net.h"
#include "components/tracking.h"

namespace slam_components {

class System {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<System>;
  Ptr makeShared() { return std::make_shared<System>(*this); }

public:
  System() = default;
  ~System() {}

  bool initialize(const std::string &config_path);

private:
  bool initializeNetwork(const cv::FileNode &node);
  bool initializeCalibration(const cv::FileNode &node);

private:
  DroidNet::Ptr droid_net_;
  Calibration::Ptr calibration_;
  Tracking::Ptr tracking_;
};

} // namespace slam_components