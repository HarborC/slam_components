#include "components/system.h"

namespace slam_components {

void System::feedIMU(const double &timestamp,
                     const Eigen::Vector3d &angular_velocity,
                     const Eigen::Vector3d &acceleration) {
  IMUData::Ptr imu_data(new IMUData(timestamp, angular_velocity, acceleration));

  feed_mtx.lock();
  imu_deque.push_back(imu_data);
  feed_mtx.unlock();
}

void System::feedCamera(const double &timestamp,
                        const std::vector<cv::Mat> &images) {
  CameraData::Ptr camera_data(new CameraData(timestamp, images));

  feed_mtx.lock();
  camera_deque.push_back(camera_data);
  feed_mtx.unlock();
}

bool System::getTrackingInput(TrackingInput& input) [

  return true;
]

bool System::initialize(const std::string &config_path) {
  cv::FileStorage node(config_path, cv::FileStorage::READ);
  if (!node.isOpened()) {
    std::cerr << "Error: Failed to open config file\n";
    return false;
  }

  if (node["Calibration"].empty() ||
      !initializeCalibration(node["Calibration"])) {
    std::cerr << "Error: Failed to calibration Network\n";
    return false;
  }

  if (node["Network"].empty() || !initializeNetwork(node["Network"])) {
    std::cerr << "Error: Failed to initialize Network\n";
    return false;
  }

  if (node["Tracking"].empty()) {
    std::cerr << "Error: Tracking is not provided\n";
    return false;
  } else {
    tracking_.reset(new Tracking());
    if (!tracking_->initialize(node["Tracking"], droid_net_, calibration_)) {
      std::cerr << "Error: Failed to initialize Tracking\n";
      return false;
    }
  }

  return true;
}

bool System::initializeNetwork(const cv::FileNode &node) {
  if (node["droidnet"].empty()) {
    std::cerr << "Error: droidnet is not provided\n";
    return false;
  }

  droid_net_.reset(new DroidNet());
  if (!droid_net_->initialize(node["droidnet"])) {
    std::cerr << "Error: Failed to initialize DroidNet\n";
    return false;
  }

  return true;
}

bool System::initializeCalibration(const cv::FileNode &node) {
  if (node["path"].empty()) {
    std::cerr << "Error: calibration.path is not provided\n";
    return false;
  }

  std::string calib_file;
  node["path"] >> calib_file;

  calibration_.reset(new Calibration());
  calibration_->load(std::string(PROJECT_DIR) + calib_file);

  return true;
}

} // namespace slam_components