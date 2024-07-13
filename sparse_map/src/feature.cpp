#include "sparse_map/feature.h"

Feature::Feature(const FeatureIDType &_id) : id_(_id) {}

void Feature::addObservation(const FrameIDType &frame_id, const int &cam_id,
                             const int &pt_id) {
  if (observations_.find(frame_id) == observations_.end()) {
    observations_[frame_id] = std::map<int, int>();
    frame_ids_.push_back(frame_id);
  }

  if (observations_[frame_id].find(cam_id) == observations_[frame_id].end()) {
    observation_size_++;
    if (observation_size_ == 1) {
      ref_frame_id_ = frame_id;
      ref_cam_id_ = cam_id;
    }
  }
  
  observations_[frame_id][cam_id] = pt_id;
}

void Feature::removeObservationByFrameId(const FrameIDType &frame_id) {
  if (observations_.find(frame_id) != observations_.end()) {
    for (auto &obs : observations_[frame_id]) {
      observation_size_--;
    }
    observations_.erase(frame_id);
    std::remove(frame_ids_.begin(), frame_ids_.end(), frame_id);
  }
}

int Feature::frameSize() {
  return frame_ids_.size();
}

int Feature::observationSize() {
  return observation_size_;
}

void Feature::update() {
  sort(frame_ids_.begin(), frame_ids_.end());

  // find the reference frame
  ref_frame_id_ = frame_ids_[0];
  ref_cam_id_ = observations_[ref_frame_id_].begin()->first;
}