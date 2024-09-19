#include "calibration/calibration.h"
#include "utils/io_utils.h"

Calibration::Calibration() {}
Calibration::~Calibration() {}

size_t Calibration::camNum() const { return camera_map_.size(); }

size_t Calibration::imuNum() const { return imu_map_.size(); }

void Calibration::addCamera(const Camera::Ptr &camera) {
  int cam_id = camera_map_.size();
  std::string cam_name = "cam" + std::to_string(cam_id);
  camera_map_[cam_name] = camera;
}

void Calibration::addIMU(const IMU::Ptr &imu) {
  int imu_id = imu_map_.size();
  std::string imu_name = "imu" + std::to_string(imu_id);
  imu_map_[imu_name] = imu;
}

Camera::Ptr Calibration::getCamera(const size_t &cam_id) {
  std::string cam_name = "cam" + std::to_string(cam_id);
  return camera_map_[cam_name];
}

IMU::Ptr Calibration::getIMU(const size_t &imu_id) {
  std::string imu_name = "imu" + std::to_string(imu_id);
  return imu_map_[imu_name];
}

Sensor::Ptr Calibration::getBodySensor() { return body_sensor_; }

void Calibration::load(const std::string &calib_file) {
  cv::FileStorage calib = cv::FileStorage(calib_file, cv::FileStorage::READ);
  if (!calib.isOpened()) {
    std::cerr << "Error: Calibration file " << calib_file
              << " can not be opened!" << std::endl;
    return;
  }

  std::string parent_dir = Utils::GetParentDir(calib_file);
  if (!calib["cam_num"].empty()) {
    int cam_num;
    calib["cam_num"] >> cam_num;
    for (int i = 0; i < cam_num; ++i) {
      Camera::Ptr camera(new Camera());
      std::string cam_file;
      calib[std::string("cam") + std::to_string(i)] >> cam_file;
      cam_file = parent_dir + "/" + cam_file;
      camera->load(cam_file);
      camera->setName(std::string("cam") + std::to_string(i));
      addCamera(camera);
    }
  }

  if (!calib["imu_num"].empty()) {
    int imu_num;
    calib["imu_num"] >> imu_num;
    for (int i = 0; i < imu_num; ++i) {
      IMU::Ptr imu(new IMU());

      std::string imu_file;
      calib[std::string("imu") + std::to_string(i)] >> imu_file;
      imu_file = parent_dir + "/" + imu_file;
      imu->load(imu_file);
      imu->setName(std::string("imu") + std::to_string(i));
      addIMU(imu);
    }
  }

  if (!calib["body_sensor"].empty()) {
    std::string body_sensor_name;
    calib["body_sensor"] >> body_sensor_name;
    if (imu_map_.find(body_sensor_name) != imu_map_.end()) {
      body_sensor_ = imu_map_[body_sensor_name];
    } else if (camera_map_.find(body_sensor_name) != camera_map_.end()) {
      body_sensor_ = camera_map_[body_sensor_name];
    } else {
      std::cerr << "Error: Body sensor " << body_sensor_name
                << " is not found in the calibration file!" << std::endl;
    }
  }

  calib.release();
}

void Calibration::print() {
  std::cout << "Calibration Parameters: " << std::endl;
  std::cout << "Body Sensor: " << getBodySensor()->name() << std::endl;
  std::cout << "Camera Number: " << camNum() << std::endl;
  for (size_t i = 0; i < camNum(); ++i) {
    getCamera(i)->print();
  }

  std::cout << "IMU Number: " << imuNum() << std::endl;
  for (size_t i = 0; i < imuNum(); ++i) {
    getIMU(i)->print();
  }
}