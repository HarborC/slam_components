#pragma once

#include "calibration/camera.h"
#include "calibration/imu.h"

class Calibration {
public:
  using Ptr = std::shared_ptr<Calibration>;

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

  void load(const std::string &calib_file);

  void print() const;

protected:
  CameraPtrMap camera_map_;
  IMUPtrMap imu_map_;
  Sensor::Ptr body_sensor_;

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("camera_map", camera_map_),
       cereal::make_nvp("imu_map", imu_map_),
       cereal::make_nvp("body_sensor", body_sensor_));
  }
};