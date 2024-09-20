#pragma once

#include "calibration/common.h"

enum SensorType { None = 0, IMUSensor = 1, LidarSensor = 2, CameraSensor = 3 };

class Sensor {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Sensor>;
  Ptr makeShared() { return std::make_shared<Sensor>(*this); }

public:
  Sensor();
  virtual ~Sensor() = default;

  int type() const;
  void setType(const SensorType &_type);

  std::string name() const;
  void setName(const std::string &_name);

  std::string topicName() const;
  void setTopicName(const std::string &_topic_name);

  int frequency() const;
  void setFrequency(const int &_frequency);

  Eigen::Matrix4d getExtrinsic() const;
  void setExtrinsic(const Eigen::Matrix4d &_extrinsic);

  virtual void load(const std::string &calib_file);
  virtual void print() const;

protected:
  SensorType type_ = SensorType::None;
  std::string name_;
  std::string topic_name_;
  int frequency_;
  Eigen::Matrix4d extrinsic_ = Eigen::Matrix4d::Identity();

private:
  friend class cereal::access;
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("type", type_),
       cereal::make_nvp("extrinsic", extrinsic_),
       cereal::make_nvp("name", name_),
       cereal::make_nvp("topic_name", topic_name_),
       cereal::make_nvp("frequency", frequency_));
  }
};
typedef std::vector<Sensor> SensorVec;
typedef std::map<std::string, Sensor::Ptr> SensorPtrMap;