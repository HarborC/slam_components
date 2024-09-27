#include "components/local_mapping.h"
#include "components/network/droid_net/corr.h"
#include "components/network/droid_net/utils.h"

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include "utils/log_utils.h"

namespace slam_components {

bool LocalMapping::initialize(const cv::FileNode &node,
                              const DroidNet::Ptr &droid_net,
                              const Calibration::Ptr &calibration,
                              const foxglove_viz::Visualizer::Ptr &viz_server) {
  droid_net_ = droid_net;
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

  keyframe_buf_mtx_.lock();
  curr_keyframe_ = keyframe_buf_.front();
  keyframe_buf_.pop_front();
  keyframe_buf_mtx_.unlock();

  estimateInitialIdepth();
}

void LocalMapping::processLoop() {
  while (true) {
    process(true);
  }
}

} // namespace slam_components