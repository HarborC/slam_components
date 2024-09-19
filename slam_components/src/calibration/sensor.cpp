#include "calibration/sensor.h"

Sensor::Sensor() {}

int Sensor::type() const { return type_; }

void Sensor::setType(const SensorType &_type) { type_ = _type; }

std::string Sensor::name() const { return name_; }

void Sensor::setName(const std::string &_name) { name_ = _name; }

std::string Sensor::topicName() const { return topic_name_; }

void Sensor::setTopicName(const std::string &_topic_name) {
  topic_name_ = _topic_name;
}

int Sensor::frequency() const { return frequency_; }

void Sensor::setFrequency(const int &_frequency) { frequency_ = _frequency; }

Eigen::Matrix4d Sensor::getExtrinsic() const { return extrinsic_; }

void Sensor::setExtrinsic(const Eigen::Matrix4d &_extrinsic) {
  extrinsic_ = _extrinsic;
}

void Sensor::load(const std::string &calib_file) {
  cv::FileStorage calib = cv::FileStorage(calib_file, cv::FileStorage::READ);
  if (!calib.isOpened()) {
    std::cerr << "Error: Sensor calibration file " << calib_file
              << " can not be opened!" << std::endl;
    return;
  }

  if (!calib["name"].empty()) {
    calib["name"] >> name_;
  }

  if (!calib["topic_name"].empty()) {
    calib["topic_name"] >> topic_name_;
  }

  if (!calib["frequency"].empty()) {
    calib["frequency"] >> frequency_;
  }

  if (!calib["extrinsic"].empty()) {
    std::vector<double> extrinsic_vec;
    calib["extrinsic"] >> extrinsic_vec;

    extrinsic_ = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        extrinsic_(i, j) = extrinsic_vec[i * 4 + j];
      }
    }
  }

  calib.release();
}

void Sensor::print() const {
  std::cout << "------------------------------\n";
  std::cout << "Name: " << name_ << "\n";
  std::cout << "Type: " << int(type_) << "\n";
  std::cout << "Topic Name: " << topic_name_ << "\n";
  std::cout << "Frequency: " << frequency_ << "\n";
  std::cout << "Extrinsic: \n" << extrinsic_ << "\n";
}