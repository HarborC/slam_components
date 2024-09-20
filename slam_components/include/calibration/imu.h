#pragma once

#include "calibration/sensor.h"

class IMU : public Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<IMU>;
  Ptr makeShared() { return std::make_shared<IMU>(*this); }

public:
  IMU();

  double getAccNoiseDensity() const;
  double getAccRandomWalk() const;
  double getGyrNoiseDensity() const;
  double getGyrRandomWalk() const;

  void setIMUParams(double _acc_noise_density, double _acc_random_walk,
                    double _gyr_noise_density, double _gyr_random_walk);

  std::vector<double> getIMUParams() const;

  virtual void load(const std::string &calib_file);
  virtual void print() const;

protected:
  double acc_noise_density_;
  double acc_random_walk_;
  double gyr_noise_density_;
  double gyr_random_walk_;

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::base_class<Sensor>(this),
       cereal::make_nvp("acc_noise_density", acc_noise_density_),
       cereal::make_nvp("acc_random_walk", acc_random_walk_),
       cereal::make_nvp("gyr_noise_density", gyr_noise_density_),
       cereal::make_nvp("gyr_random_walk", gyr_random_walk_));
  }
};
typedef std::vector<IMU::Ptr> IMUPtrVec;
typedef std::map<std::string, IMU::Ptr> IMUPtrMap;