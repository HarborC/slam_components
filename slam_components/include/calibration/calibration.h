#pragma once

#include "calibration/camera.h"
#include "calibration/imu.h"

class Calibration {
public:
  using Ptr = std::shared_ptr<Calibration>;
  Ptr makeShared() { return std::make_shared<Calibration>(*this); }

public:
  Calibration();
  ~Calibration();

  size_t camNum() const;

  size_t imuNum() const;

  void addCamera(const Camera::Ptr &camera);

  void addIMU(const IMU::Ptr &imu);

  Camera::Ptr getCamera(const size_t &cam_id);

  IMU::Ptr getIMU(const size_t &imu_id);

  Sensor::Ptr getBodySensor();

  std::vector<std::pair<int, int>> getStereoPairs();

  double getAverageDepth();

  double getGravityNorm();

  void load(const std::string &calib_file);

  void print();

  void calcOverlapViews();

private:
  CameraPtrMap camera_map_;
  IMUPtrMap imu_map_;
  Sensor::Ptr body_sensor_;
  std::vector<std::pair<int, int>> stereo_pairs_;
  double average_depth_ = 20;
  double gravity_norm_ = 9.81;

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("camera_map", camera_map_),
       cereal::make_nvp("imu_map", imu_map_),
       cereal::make_nvp("body_sensor", body_sensor_),
       cereal::make_nvp("stereo_pairs", stereo_pairs_),
       cereal::make_nvp("average_depth", average_depth_),
       cereal::make_nvp("gravity_norm", gravity_norm_));
  }
};