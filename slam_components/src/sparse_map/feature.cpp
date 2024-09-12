#include "sparse_map/feature.h"

Feature::Feature(const FeatureIDType &_id) : id_(_id) {}

bool Feature::isValid() {
  if (observation_size_) {
    return true;
  }

  return false;
}

void Feature::addObservation(const FrameIDType &frame_id, const int &cam_id,
                             const int &pt_id) {
  if (observations_.find(frame_id) == observations_.end()) {
    observations_[frame_id] = std::map<int, int>();
    frame_ids_.push_back(frame_id);
  }

  if (observations_[frame_id].find(cam_id) == observations_[frame_id].end()) {
    if (observation_size_ == 0) {
      ref_frame_id_ = frame_id;
      ref_cam_id_ = cam_id;
    }
    observation_size_++;
  }

  observations_[frame_id][cam_id] = pt_id;
}

bool Feature::hasObservation(const FrameIDType &frame_id, const int &cam_id) {
  if (observations_.find(frame_id) != observations_.end()) {
    if (cam_id < 0) {
      return true;
    } else {
      return observations_[frame_id].find(cam_id) !=
             observations_[frame_id].end();
    }
  }
  return false;
}

void Feature::removeObservation(const FrameIDType &frame_id,
                                const int &cam_id) {
  if (hasObservation(frame_id, cam_id)) {
    observations_[frame_id].erase(cam_id);
    observation_size_--;
    if (observations_[frame_id].empty()) {
      observations_.erase(frame_id);
      std::remove(frame_ids_.begin(), frame_ids_.end(), frame_id);
    }
  }
}

void Feature::removeObservationByFrameId(const FrameIDType &frame_id) {
  if (hasObservation(frame_id)) {
    for (auto &obs : observations_[frame_id]) {
      observation_size_--;
    }
    observations_.erase(frame_id);
    std::remove(frame_ids_.begin(), frame_ids_.end(), frame_id);

    if (observation_size_) {
      refUpdate();
    }
  }
}

int Feature::coVisFrameSize() { return frame_ids_.size(); }

int Feature::observationSize() { return observation_size_; }

void Feature::refUpdate() {
  sort(frame_ids_.begin(), frame_ids_.end());

  // find the reference frame
  ref_frame_id_ = frame_ids_[0];
  ref_cam_id_ = observations_[ref_frame_id_].begin()->first;
}

double *Feature::getInvDepthParams() { return &inv_depth_; }

double *Feature::getWorldPointParams() { return world_point_.data(); }

double Feature::getInvDepth() { return inv_depth_; }

Eigen::Vector3d Feature::getWorldPoint() { return world_point_; }

bool Feature::isTriangulated() { return inv_depth_ > 0; }

void Feature::setInvDepth(const double &inv_depth) { inv_depth_ = inv_depth; }

void Feature::setWorldPoint(const Eigen::Vector3d &world_point) {
  world_point_ = world_point;
}

int Feature::observation(const FrameIDType &frame_id, const int &cam_id) {
  if (observations_.find(frame_id) != observations_.end()) {
    if (observations_[frame_id].find(cam_id) != observations_[frame_id].end()) {
      return observations_[frame_id][cam_id];
    }
  }
  return -1;
}