#include "components/system.h"

namespace slam_components {

System::~System() { requestFinish(); }

void System::feedIMU(const double &timestamp,
                     const Eigen::Vector3d &angular_velocity,
                     const Eigen::Vector3d &acceleration) {
  IMUData::Ptr imu_data(new IMUData(timestamp, angular_velocity, acceleration));

  feed_mtx.lock();
  imu_buf_.push_back(imu_data);
  feed_mtx.unlock();
}

void System::feedCamera(const double &timestamp,
                        const std::vector<cv::Mat> &images) {
  CameraData::Ptr camera_data(new CameraData(timestamp, images));

  feed_mtx.lock();
  camera_buf_.push_back(camera_data);
  feed_mtx.unlock();
}

void System::begin() {
  is_running_ = true;
  loop_thread.reset(new std::thread(&System::processLoop, this));
}

void System::requestFinish() {
  is_running_ = false;
  if (loop_thread) {
    loop_thread->join();
  }
}

bool System::getTrackingInput(TrackingInput &input) {
  if (camera_buf_.empty() || imu_buf_.empty()) {
    return false;
  }

  feed_mtx.lock();
  double t_image = camera_buf_.front()->timestamp_;
  if (!(imu_buf_.back()->timestamp_ > t_image)) {
    feed_mtx.unlock();
    return false;
  }

  if (!(imu_buf_.front()->timestamp_ < t_image)) {
    camera_buf_.pop_front();
    feed_mtx.unlock();
    return false;
  }

  auto camera_data = camera_buf_.front();
  camera_buf_.pop_front();

  std::vector<IMUData::Ptr> IMUs;

  int idx = 0;
  for (; idx < imu_buf_.size(); idx++) {
    if (imu_buf_[idx]->timestamp_ < t_image) {
      IMUs.emplace_back(imu_buf_[idx]);
    } else {
      break;
    }
  }
  IMUs.emplace_back(imu_buf_[idx]);

  while (imu_buf_.front()->timestamp_ < t_image) {
    auto iter = imu_buf_.begin();
    iter = imu_buf_.erase(iter);
  }

  feed_mtx.unlock();

  input.camera_data = camera_data;
  input.inetial_data = IMUs;

  return true;
}

void System::processLoop() {
  while (1) {
    TrackingInput input;
    if (getTrackingInput(input)) {
      tracking_->track(input);
    } else if (!is_running_) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }
}

bool System::initialize(const std::string &config_path) {
  cv::FileStorage node(config_path, cv::FileStorage::READ);
  if (!node.isOpened()) {
    std::cerr << "Error: Failed to open config file\n";
    return false;
  }

  if (node["Visualizer"].empty() || !initializeViz(node["Visualizer"])) {
    std::cerr << "Error: Failed to initialize Visualizer\n";
    return false;
  }

  if (node["Calibration"].empty() ||
      !initializeCalibration(node["Calibration"])) {
    std::cerr << "Error: Failed to initialize calibration\n";
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
    if (!tracking_->initialize(node["Tracking"], droid_net_, calibration_,
                               viz_server_)) {
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

bool System::initializeViz(const cv::FileNode &node) {
  if (node["port"].empty()) {
    std::cerr << "Error: calibration.port is not provided\n";
    return false;
  }

  int port;
  node["port"] >> port;

  viz_server_.reset(new foxglove_viz::Visualizer(port));

  return true;
}

} // namespace slam_components