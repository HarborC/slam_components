#pragma once

#include "calibration/sensor.h"
#include "general_camera_model/general_camera_model.hpp"

class Camera : public Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Camera>;

public:
  Camera() { type_ = SensorType::CameraSensor; }

  void setCameraModel(
      const general_camera_model::GeneralCameraModel &_camera_model) {
    camera_model_ = _camera_model;
  }

  const general_camera_model::GeneralCameraModel &getCameraModel() const {
    return camera_model_;
  }

protected:
  general_camera_model::GeneralCameraModel camera_model_;
};
typedef std::vector<Camera> CameraVec;