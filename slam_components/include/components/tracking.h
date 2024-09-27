#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/calibration.h"
#include "components/frame.h"
#include "components/network/droid_net/droid_net.h"
#include "components/sensor_data.h"

#include "foxglove/visualizer.h"

namespace slam_components {

struct TrackingInput {
  CameraData::Ptr camera_data;
  std::vector<IMUData::Ptr> inetial_data;
  std::map<std::string, std::vector<double>> groundtruth;
};

class Tracking {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Tracking>;

public:
  Tracking() = default;
  ~Tracking() {}

  bool initialize(const cv::FileNode &node, const DroidNet::Ptr &droid_net,
                  const Calibration::Ptr &calibration,
                  const foxglove_viz::Visualizer::Ptr &viz_server);

  Frame::Ptr process(const TrackingInput &input);

  void printSetting();

private:
  void estimateInitialPose();
  void estimatePoseByConstantVelocity();
  void estimatePoseByIMU();

  bool judgeKeyframe();

  void propressImage(const Frame::Ptr &frame);
  void extractDenseFeature(const Frame::Ptr &frame,
                           bool only_feature_map = false);
  void extractSparseFeature(const Frame::Ptr &frame);
  bool motionFilter();

  void publishRawImage();

private:
  FrameIDType next_frame_id_ = 0;
  DroidNet::Ptr droid_net_;
  Calibration::Ptr calibration_;
  foxglove_viz::Visualizer::Ptr viz_server_;

  Frame::Ptr last_frame_ = nullptr;
  Frame::Ptr last_keyframe_ = nullptr;
  Frame::Ptr curr_frame_ = nullptr;
  Eigen::Vector3d linear_velocity_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_velocity_ = Eigen::Vector3d::Zero();
  bool is_imu_initial_ = false;

  // hyper parameters
  float motion_filter_thresh_ = 2.5;
  int image_downsample_scale_ = 8;
  std::string motion_model_ = "constant_velocity";
  // mean, std for image normalization
  torch::Tensor MEAN, STDV;
};

} // namespace slam_components