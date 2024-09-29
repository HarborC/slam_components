#include "components/local_mapping.h"
#include "components/network/droid/corr.h"
#include "components/network/utils.h"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "utils/log_utils.h"

namespace slam_components {

bool LocalMapping::initialize(const cv::FileNode &node,
                              const Network::Ptr &network,
                              const Calibration::Ptr &calibration,
                              const foxglove_viz::Visualizer::Ptr &viz_server) {
  network_ = network;
  calibration_ = calibration;
  viz_server_ = viz_server;

  if (node["multi_thread"].empty()) {
    SPDLOG_CRITICAL("LocalMapping.multi_thread is not provided");
    return false;
  } else {
    node["multi_thread"] >> multi_thread_;
  }

  if (multi_thread_) {
    loop_thread.reset(new std::thread(&LocalMapping::processLoop, this));
    loop_thread->detach();
  }

  printSetting();

  return true;
}

void LocalMapping::printSetting() {
  SPDLOG_INFO("\nLocalMapping Setting: \n - multi_thread: {}", multi_thread_);
}

void LocalMapping::estimateInitialIdepth() {}

void LocalMapping::push_back(const Frame::Ptr &frame) {
  keyframe_buf_mtx_.lock();
  keyframe_buf_.push_back(frame);
  keyframe_buf_mtx_.unlock();
}

void LocalMapping::process(bool in_loop) {
  if (multi_thread_ && !in_loop)
    return;

  if (keyframe_buf_.empty()) {
    if (multi_thread_)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return;
  }

  if (in_loop) {
    SPDLOG_INFO("In Loop");
  }

  static double process_total_time = 0.0;
  static int process_total_count = 0;

  TimeStatistics local_mapping_statistics("LocalMapping");
  local_mapping_statistics.tic();

  keyframe_buf_mtx_.lock();
  curr_keyframe_ = keyframe_buf_.front();
  keyframe_buf_.pop_front();
  keyframe_buf_mtx_.unlock();

  local_mapping_statistics.tocAndTic("initialize frame");

  estimateInitialIdepth();
  local_mapping_statistics.tocAndTic("estimate initial idepth");

  local_mapping_statistics.tocAndTic("process keyframe");

  process_total_time +=
      local_mapping_statistics.logTimeStatistics(curr_keyframe_->id());
  process_total_count += 1;

  SPDLOG_INFO("LocalMapping process average time: {} ms",
              process_total_time / process_total_count);
}

void LocalMapping::processLoop() {
  while (true) {
    process(true);
  }
}

} // namespace slam_components