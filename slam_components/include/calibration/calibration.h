#pragma once

#include "calibration/camera.h"

class Calibration {
public:
  using Ptr = std::shared_ptr<Calibration>;

public:
  Calibration() {}
  ~Calibration() {}

  void addCamera(const Camera &camera) { camera_vec_.push_back(camera); }

  size_t camNum() const { return camera_vec_.size(); }

  const Camera &getCamera(const size_t &cam_id) const {
    return camera_vec_[cam_id];
  }

protected:
  CameraVec camera_vec_;
};