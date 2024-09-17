#pragma once

#include "sparse_map/common.h"

class Feature {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Feature>;

public:
  Feature() {}
  Feature(const FeatureIDType &_id);
  ~Feature() {}

  FeatureIDType id() { return id_; }

  FrameIDType refFrameId() { return ref_frame_id_; }

  int refCamId() { return ref_cam_id_; }

  const std::unordered_map<FrameIDType, std::map<int, int>> &observations() {
    return observations_;
  }

  int observation(const FrameIDType &frame_id, const int &cam_id);

  void addObservation(const FrameIDType &frame_id, const int &cam_id,
                      const int &pt_id);

  bool hasObservation(const FrameIDType &frame_id, const int &cam_id = -1);

  void removeObservation(const FrameIDType &frame_id, const int &cam_id);

  void removeObservationByFrameId(const FrameIDType &frame_id);

  bool isValid();

  int coVisFrameSize();

  int observationSize();

  void refUpdate();

  bool isTriangulated();

  double getInvDepth();

  void setInvDepth(const double &inv_depth);

  Eigen::Vector3d getWorldPoint();

  void setWorldPoint(const Eigen::Vector3d &world_point);

  double *getInvDepthParams();

  double *getWorldPointParams();

private:
  // feature id
  FeatureIDType id_;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::unordered_map<FrameIDType, std::map<int, int>> observations_;
  int observation_size_ = 0;

  // all of frame ids that the feature is observed.(sorted)
  std::vector<FrameIDType> frame_ids_;

  // reference frame id
  FrameIDType ref_frame_id_ = -1;
  int ref_cam_id_ = -1;

  // inv_depth_ and world_point_
  double inv_depth_ = -1.0;
  Eigen::Vector3d world_point_ = Eigen::Vector3d::Zero();

  //   bool is_stereo = false;
  //   int tracked_num = 0;

private:
  friend class cereal::access;

  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("id", id_),
       cereal::make_nvp("observations", observations_),
       cereal::make_nvp("observation_size", observation_size_),
       cereal::make_nvp("frame_ids", frame_ids_),
       cereal::make_nvp("ref_frame_id", ref_frame_id_),
       cereal::make_nvp("ref_cam_id", ref_cam_id_),
       cereal::make_nvp("inv_depth", inv_depth_),
       cereal::make_nvp("world_point", world_point_));
  }
};
typedef std::unordered_map<FeatureIDType, Feature::Ptr> FeatureMap;
