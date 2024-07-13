#pragma once

#include "sparse_map/common.h"

class Feature {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Feature>;

public:
  Feature(const FeatureIDType &_id);
  ~Feature() {}

  void addObservation(const FrameIDType &frame_id, const int &cam_id,
                      const int &pt_id);
  void removeObservationByFrameId(const FrameIDType &frame_id);
  int frameSize();
  int observationSize();
  void update();

  // int triangulate(const FrameDataset::Ptr &frame_dataset);

public:
  FeatureIDType id_;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::unordered_map<FrameIDType, std::map<int, int>> observations_;
  int observation_size_ = 0;

  // all of frame ids that the feature is observed.(sorted)
  std::vector<FrameIDType> frame_ids_;

  // reference frame id
  FrameIDType ref_frame_id_;
  int ref_cam_id_;
  double inv_depth_ = -1.0;
  Eigen::Vector3d world_point_ = Eigen::Vector3d::Zero();

  //   bool is_stereo = false;
  //   int tracked_num = 0;
};
typedef std::unordered_map<FeatureIDType, Feature::Ptr> FeatureMap;
