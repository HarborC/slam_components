#include "calibration/sensor.h"

Sensor::Sensor() {}

Eigen::Matrix4d Sensor::getExtrinsic() const { return extrinsic_; }

void Sensor::setExtrinsic(const Eigen::Matrix4d &_extrinsic) {
  extrinsic_ = _extrinsic;
}