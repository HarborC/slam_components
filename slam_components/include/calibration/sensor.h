#pragma once

#include "calibration/common.h"

enum SensorType { None = 0, ImuSensor = 1, LidarSensor = 2, CameraSensor = 3 };

class Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Sensor>;

public:
  Sensor();

  int type() const;

  Eigen::Matrix4d getExtrinsic() const;

  void setExtrinsic(const Eigen::Matrix4d &_extrinsic);

protected:
  SensorType type_ = SensorType::None;
  Eigen::Matrix4d extrinsic_ = Eigen::Matrix4d::Identity();

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("type", type_),
       cereal::make_nvp("extrinsic", extrinsic_));
  }
};
typedef std::vector<Sensor> SensorVec;