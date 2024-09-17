#pragma once

#include "calibration/camera.h"

class Calibration {
public:
  using Ptr = std::shared_ptr<Calibration>;

public:
  Calibration() {}
  ~Calibration() {}

  size_t camNum() const { return camera_vec_.size(); }

  void addCamera(const Camera::Ptr &camera) { camera_vec_.push_back(camera); }

  Camera::Ptr getCamera(const size_t &cam_id) { return camera_vec_[cam_id]; }

protected:
  CameraPtrVec camera_vec_;

public:
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("camera_vec", camera_vec_));
  }
};