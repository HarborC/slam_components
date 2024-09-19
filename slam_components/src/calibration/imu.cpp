#include "calibration/imu.h"

IMU::IMU() { type_ = SensorType::IMUSensor; }

double IMU::getAccNoiseDensity() const { return acc_noise_density_; }
double IMU::getAccRandomWalk() const { return acc_random_walk_; }
double IMU::getGyrNoiseDensity() const { return gyr_noise_density_; }
double IMU::getGyrRandomWalk() const { return gyr_random_walk_; }

void IMU::setIMUParams(double _acc_noise_density, double _acc_random_walk,
                       double _gyr_noise_density, double _gyr_random_walk) {
  acc_noise_density_ = _acc_noise_density;
  acc_random_walk_ = _acc_random_walk;
  gyr_noise_density_ = _gyr_noise_density;
  gyr_random_walk_ = _gyr_random_walk;
}

std::vector<double> IMU::getIMUParams() const {
  return {acc_noise_density_, acc_random_walk_, gyr_noise_density_,
          gyr_random_walk_};
}

void IMU::load(const std::string &calib_file) {
  Sensor::load(calib_file);

  cv::FileStorage calib = cv::FileStorage(calib_file, cv::FileStorage::READ);
  if (!calib.isOpened()) {
    std::cerr << "Error: IMU calibration file " << calib_file
              << " can not be opened!" << std::endl;
    return;
  }

  if (!calib["acc_noise_density"].empty())
    calib["acc_noise_density"] >> acc_noise_density_;

  if (!calib["acc_random_walk"].empty())
    calib["acc_random_walk"] >> acc_random_walk_;

  if (!calib["gyr_noise_density"].empty())
    calib["gyr_noise_density"] >> gyr_noise_density_;

  if (!calib["gyr_random_walk"].empty())
    calib["gyr_random_walk"] >> gyr_random_walk_;

  calib.release();

  std::cout << "IMU calibration file " << calib_file << " loaded." << std::endl;
}

void IMU::print() const {
  std::cout << "IMU Parameters: " << std::endl;
  Sensor::print();
  std::cout << "acc_noise_density: " << acc_noise_density_ << std::endl;
  std::cout << "acc_random_walk: " << acc_random_walk_ << std::endl;
  std::cout << "gyr_noise_density: " << gyr_noise_density_ << std::endl;
  std::cout << "gyr_random_walk: " << gyr_random_walk_ << std::endl;
}