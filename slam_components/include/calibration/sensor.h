#pragma once

#include "calibration/common.h"

enum SensorType { None, ImuSensor, LidarSensor, CameraSensor };

class Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Sensor>;

public:
  Sensor();

  Eigen::Matrix4d getExtrinsic() const;

  void setExtrinsic(const Eigen::Matrix4d &_extrinsic);

protected:
  SensorType type_ = SensorType::None;
  Eigen::Matrix4d extrinsic_ = Eigen::Matrix4d::Identity();
};
typedef std::vector<Sensor> SensorVec;