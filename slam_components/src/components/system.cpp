#include "components/system.h"
#include "components/network/utils.h"
#include "utils/log_utils.h"

namespace slam_components {

System::~System() {
  is_running_ = false;
  if (loop_thread) {
    loop_thread->join();
    loop_thread = nullptr;
  }
}

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

  std::cout << "System start!" << std::endl;
}

void System::requestFinish() {
  std::cout << "System try to end!" << std::endl;
  is_running_ = false;
  if (loop_thread) {
    loop_thread->join();
    loop_thread = nullptr;
  }

  std::cout << "System end!" << std::endl;
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
      auto keyframe = tracking_->process(input);
      if (keyframe) {
        local_mapping_->push_back(keyframe);
        local_mapping_->process();
      }
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
    std::cerr << "Failed to open config file" << std::endl;
    return false;
  }

  if (node["Log"].empty() || !initializeLog(node["Log"])) {
    std::cerr << "Failed to initialize Log" << std::endl;
    return false;
  }

  if (node["Visualizer"].empty() || !initializeViz(node["Visualizer"])) {
    SPDLOG_CRITICAL("Failed to initialize Visualizer");
    return false;
  }

  if (node["Calibration"].empty() ||
      !initializeCalibration(node["Calibration"])) {
    SPDLOG_CRITICAL("Failed to initialize calibration");
    return false;
  }

  if (node["Network"].empty() || !initializeNetwork(node["Network"])) {
    SPDLOG_CRITICAL("Failed to initialize Network");
    return false;
  }

  if (node["Tracking"].empty()) {
    SPDLOG_CRITICAL("Tracking is not provided");
    return false;
  } else {
    tracking_.reset(new Tracking());
    if (!tracking_->initialize(node["Tracking"], network_, calibration_,
                               viz_server_)) {
      SPDLOG_CRITICAL("Failed to initialize Tracking");
      return false;
    }
  }

  if (node["LocalMapping"].empty()) {
    SPDLOG_CRITICAL("LocalMapping is not provided");
    return false;
  } else {
    local_mapping_.reset(new LocalMapping());
    if (!local_mapping_->initialize(node["LocalMapping"], network_,
                                    calibration_, viz_server_)) {
      SPDLOG_CRITICAL("Failed to initialize LocalMapping");
      return false;
    }
  }

  return true;
}

bool System::initializeNetwork(const cv::FileNode &node) {
  network_.reset(new Network());
  if (!network_->initialize(node)) {
    SPDLOG_CRITICAL("Failed to initialize Network");
    return false;
  }

  return true;
}

bool System::initializeCalibration(const cv::FileNode &node) {
  if (node["path"].empty()) {
    SPDLOG_CRITICAL("calibration.path is not provided");
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
    SPDLOG_CRITICAL("calibration.port is not provided");
    return false;
  }

  int port;
  node["port"] >> port;

  viz_server_.reset(new foxglove_viz::Visualizer(port));

  return true;
}

bool System::initializeLog(const cv::FileNode &node) {
  if (node["path"].empty()) {
    std::cerr << "Log.path is not provided" << std::endl;
    return false;
  }

  std::string path;
  node["path"] >> path;

  if (node["enable_std"].empty()) {
    initSpdlog("slam_components", path);
  } else {
    bool alsologtostderr;
    node["enable_std"] >> alsologtostderr;
    initSpdlog("slam_components", path, alsologtostderr);
  }

  return true;
}

} // namespace slam_components