#pragma once

#include <deque>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calibration/calibration.h"
#include "components/frame.h"
#include "components/network/network.h"
#include "components/sensor_data.h"

#include "foxglove/visualizer.h"

namespace slam_components {

class LocalMapping {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<LocalMapping>;

public:
  LocalMapping() = default;
  ~LocalMapping() {}

  bool initialize(const cv::FileNode &node, const Network::Ptr &network,
                  const Calibration::Ptr &calibration,
                  const foxglove_viz::Visualizer::Ptr &viz_server);

  void push_back(const Frame::Ptr &frame);

  void process(bool in_loop = false);

  void printSetting();

private:
  void estimateInitialIdepth();

  void processLoop();

private:
  FrameIDType next_frame_id_ = 0;
  Network::Ptr network_;
  Calibration::Ptr calibration_;
  foxglove_viz::Visualizer::Ptr viz_server_;
  std::shared_ptr<std::thread> loop_thread;

  Frame::Ptr last_keyframe_ = nullptr;
  Frame::Ptr curr_keyframe_ = nullptr;
  bool is_imu_initial_ = false;

  // hyper parameters
  const int image_downsample_scale_ = 8;
  bool multi_thread_ = true;

  std::deque<Frame::Ptr> keyframe_buf_;
  std::mutex keyframe_buf_mtx_;
};

} // namespace slam_components