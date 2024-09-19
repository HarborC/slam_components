#pragma once

#include "calibration/sensor.h"
#include "general_camera_model/general_camera_model.hpp"

using namespace general_camera_model;

class Camera : public Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Camera>;

public:
  Camera();

  void setCameraModel(const GeneralCameraModel::Ptr &_camera_model);

  GeneralCameraModel::Ptr getCameraModel();

  virtual void load(const std::string &calib_file);
  virtual void print() const;

protected:
  cv::Mat map_x_, map_y_;
  GeneralCameraModel::Ptr camera_model_;

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::base_class<Sensor>(this),
       cereal::make_nvp("camera_model", camera_model_),
       cereal::make_nvp("map_x", map_x_), cereal::make_nvp("map_y", map_y_));
  }
};
typedef std::vector<Camera::Ptr> CameraPtrVec;
typedef std::map<std::string, Camera::Ptr> CameraPtrMap;