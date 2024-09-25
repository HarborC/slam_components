#include "components/system.h"
#include "general_dataset/eruco_dataset.h"

int main(int argc, char **argv) {
  std::string config_file =
      std::string(PROJECT_DIR) + "/configs/eruco/mh02.yaml";

  general_dataset::ErucoDataset dataset;
  dataset.read("/mnt/i/project/slam/datasets/euroc/mh02/mav0");

  slam_components::System::Ptr system(new slam_components::System());
  if (!system->initialize(config_file)) {
    std::cerr << "Error: Failed to initialize System\n";
    return -1;
  }

  auto calib = system->getCalibration();

  system->run();

  for (size_t i = 0; i < dataset.imu_data_vec_.size(); ++i) {
    system->feedIMU(dataset.imu_data_vec_[i].timestamp_,
                    dataset.imu_data_vec_[i].gyro_,
                    dataset.imu_data_vec_[i].accel_);
  }

  for (size_t i = 0; i < dataset.image_data_vec_.size(); ++i) {
    std::vector<cv::Mat> images;
    for (int cam_id = 0; cam_id < dataset.image_data_vec_[i].data_.size();
         cam_id++) {
      images.push_back(calib->getCamera(cam_id)->readImage(
          dataset.image_data_vec_[i].data_[i].img_path_));
    }
    system->feedCamera(dataset.image_data_vec_[i].timestamp_, images);
  }

  system->terminate();

  return 0;
}