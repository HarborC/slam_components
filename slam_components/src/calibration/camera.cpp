#include "calibration/camera.h"
#include "utils/io_utils.h"

Camera::Camera() { type_ = SensorType::CameraSensor; }

void Camera::setCameraModel(const GeneralCameraModel::Ptr &_camera_model) {
  camera_model_ = _camera_model;
}

GeneralCameraModel::Ptr Camera::getCameraModel() { return camera_model_; }

void Camera::load(const std::string &calib_file) {
  Sensor::load(calib_file);

  cv::FileStorage calib = cv::FileStorage(calib_file, cv::FileStorage::READ);
  if (!calib.isOpened()) {
    std::cerr << "Error: Camera calibration file " << calib_file
              << " can not be opened!" << std::endl;
    return;
  }

  if (!calib["intrinsic"].empty()) {
    std::string intrinsic_file;
    calib["intrinsic"] >> intrinsic_file;

    intrinsic_file = Utils::GetParentDir(calib_file) + "/" + intrinsic_file;

    camera_model_.reset(new GeneralCameraModel());
    camera_model_->loadConfigFile(intrinsic_file);
  } else {
    std::cerr << "Error: Camera calibration file " << calib_file
              << " does not contain camera model!" << std::endl;
  }

  calib.release();

  std::cout << "Camera calibration file " << calib_file << " loaded."
            << std::endl;
}

void Camera::print() const {
  std::cout << "Camera Parameters: " << std::endl;
  Sensor::print();
  if (camera_model_) {
    std::cout << camera_model_->info() << std::endl;
  }
}