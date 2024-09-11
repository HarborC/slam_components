#pragma once

#include "sparse_map/feature.h"
#include "sparse_map/frame.h"

class SparseMap {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<SparseMap>;
  SparseMap(bool use_ransac = true) : use_ransac_(use_ransac) {}
  ~SparseMap() {
    clear();
    std::cout << "SparseMap is destructed" << std::endl;
  }

  void clear() {
    feature_map_.clear();
    frame_map_.clear();
  }

  void setCalibration(const std::vector<Eigen::Matrix4d> &calibrations) {
    cam_num_ = calibrations.size();
    calibrations_ = calibrations;
  }

  void addKeyFrame(const FrameIDType &id, const std::vector<cv::Mat> &imgs,
                   const std::vector<std::vector<Eigen::Vector2d>> &keypoints,
                   const std::vector<std::vector<Eigen::Vector3d>> &bearings,
                   const std::vector<cv::Mat> &descriptors = {});

  void addKeyFrame(const Frame::Ptr &frame);

  void removeKeyFrame(const FrameIDType &id);

  void addIntraMatches(const FrameIDType &pre_frame_id,
                       const FrameIDType &cur_frame_id,
                       std::vector<std::vector<Eigen::Vector2i>> intra_matches);

  void addInterMatches(const FrameIDType &cur_frame_id,
                       const std::vector<Eigen::Vector2i> &stereo_ids,
                       std::vector<std::vector<Eigen::Vector2i>> inter_matches);

  void updateKeyFramePose(const FrameIDType &id, const Eigen::Matrix4d &pose);

  void triangulate();

  bool bundleAdjustment(const double &fx, const double &fy, const double &cx,
                        const double &cy, bool use_prior = false);

  std::vector<std::pair<size_t, size_t>> getMatches(const FrameIDType &f_id1,
                                                    const int &c_id1,
                                                    const FrameIDType &f_id2,
                                                    const int &c_id2);

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
  getCorrespondences2D2D(const FrameIDType &f_id1, const int &c_id1,
                         const FrameIDType &f_id2, const int &c_id2);

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
  getCorrespondences2D3D(const FrameIDType &f_id1, const int &c_id1);

  size_t getKeypointSize(const FrameIDType &f_id1, const int &c_id1);

  Frame::Ptr getFrame(const FrameIDType &id) { return frame_map_[id]; }

public:
  // debug output
  cv::Mat drawKeypoint(FrameIDType frame_id, int cam_id);

  cv::Mat drawMatchedKeypoint(FrameIDType frame_id, int cam_id);

  cv::Mat drawMatches(FrameIDType frame_id0, int cam_id0, FrameIDType frame_id1,
                      int cam_id1);

  cv::Mat drawFlow(FrameIDType frame_id, int cam_id,
                   FrameIDType last_frame_id = -1);

  cv::Mat drawStereoKeyPoint(FrameIDType frame_id);

  std::vector<Eigen::Vector3d> getWorldPoints();

  void printReprojError(const FrameIDType &f_id1, const int &c_id1, const double &fx, const double &fy, const double &cx, const double &cy);

  cv::Mat drawReprojKeyPoint(FrameIDType frame_id, int cam_id, const double &fx, const double &fy, const double &cx, const double &cy);

protected:
  void ransacWithF(const FrameIDType &left_frame_id, const int &left_cam_id,
                   const FrameIDType &right_frame_id, const int &right_cam_id,
                   std::vector<Eigen::Vector2i> &good_matches);

  void updateFeatureMap();

public:
  FeatureIDType feature_next_id = 0;
  FrameIDType frame_next_id = 0;
  Frame::Ptr last_frame_ = nullptr;
  FeatureMap feature_map_;
  FrameMap frame_map_;
  bool use_ransac_ = true;
  int cam_num_ = 1;
  std::vector<Eigen::Matrix4d> calibrations_ = {
      Eigen::Matrix4d::Identity()}; // Tbc
};