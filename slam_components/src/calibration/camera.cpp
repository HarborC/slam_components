#include "calibration/camera.h"
#include "general_camera_model/function.hpp"
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

  if (!calib["opt_intrinsics"].empty()) {
    std::vector<double> params;
    calib["opt_intrinsics"] >> params;

    if (params.size() != 6) {
      std::cerr << "Error: Camera calibration file " << calib_file
                << " does not contain correct camera model!" << std::endl;
    } else {
      setPinholeCameraModel(params[0], params[1], params[2], params[3],
                            params[4], params[5]);
      std::pair<cv::Mat, cv::Mat> maps =
          initUndistortRectifyMap(*camera_model_, *pinhole_camera_model_);
      map_x_ = maps.first;
      map_y_ = maps.second;
    }
  }

  calib.release();
}

void Camera::print() const {
  std::cout << "Camera Parameters: " << std::endl;
  Sensor::print();
  if (camera_model_) {
    std::cout << camera_model_->info() << std::endl;
  }
}

void Camera::setPinholeCameraModel(const double fx, const double fy,
                                   const double cx, const double cy,
                                   const double w, const double h) {
  pinhole_camera_model_.reset(new GeneralCameraModel());
  *pinhole_camera_model_ =
      general_camera_model::getSimplePinhole(fx, fy, cx, cy, w, h);
}

GeneralCameraModel::Ptr Camera::getPinholeCameraModel() {
  return pinhole_camera_model_;
}

std::vector<double> Camera::getPinholeParams() {
  if (pinhole_camera_model_) {
    std::vector<double> params = pinhole_camera_model_->getParams();
    int width = pinhole_camera_model_->width();
    int height = pinhole_camera_model_->height();
    return {params[0], params[1], params[2], params[3], width, height};
  } else {
    return std::vector<double>();
  }
}

cv::Mat Camera::readImage(const std::string &img_name, bool undistort,
                          bool bgr) {
  cv::Mat raw_img = cv::imread(img_name);

  if (raw_img.empty()) {
    std::cerr << "Failed to load image: " << img_name << std::endl;
    return cv::Mat();
  }

  if (undistort && pinhole_camera_model_) {
    cv::remap(raw_img, raw_img, map_x_, map_y_, cv::INTER_LINEAR);
  }

  if (bgr && raw_img.channels() == 1) {
    cv::cvtColor(raw_img, raw_img, cv::COLOR_GRAY2BGR);
  }

  return raw_img;
}
