#pragma once

#include "calibration/sensor.h"
#include "general_camera_model/general_camera_model.hpp"

using namespace general_camera_model;

class Camera : public Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Camera>;
  Ptr makeShared() { return std::make_shared<Camera>(*this); }

public:
  Camera();

  void setCameraModel(const GeneralCameraModel::Ptr &_camera_model);

  GeneralCameraModel::Ptr getCameraModel();

  void setPinholeCameraModel(const double fx, const double fy, const double cx,
                             const double cy, const double w, const double h);

  GeneralCameraModel::Ptr getPinholeCameraModel();

  std::vector<double> getPinholeParams();

  cv::Mat readImage(const std::string &img_name, bool undistort = true,
                    bool bgr = false);

  virtual void load(const std::string &calib_file);
  virtual void print() const;

protected:
  GeneralCameraModel::Ptr camera_model_;

  GeneralCameraModel::Ptr pinhole_camera_model_;
  cv::Mat map_x_, map_y_;

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::base_class<Sensor>(this),
       cereal::make_nvp("camera_model", camera_model_),
       cereal::make_nvp("pinhole_camera_model", pinhole_camera_model_),
       cereal::make_nvp("map_x", map_x_), cereal::make_nvp("map_y", map_y_));
  }
};
typedef std::vector<Camera::Ptr> CameraPtrVec;
typedef std::map<std::string, Camera::Ptr> CameraPtrMap;