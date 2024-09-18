#include "calibration/sensor.h"

Sensor::Sensor() {}

int Sensor::type() const { return type_; }

Eigen::Matrix4d Sensor::getExtrinsic() const { return extrinsic_; }

void Sensor::setExtrinsic(const Eigen::Matrix4d &_extrinsic) {
  extrinsic_ = _extrinsic;
}