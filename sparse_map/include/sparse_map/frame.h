#pragma once

#include "sparse_map/common.h"

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frame>;

public:
  Frame(FrameIDType _id) : id_(_id) {}
  virtual ~Frame() {}

  void addData(const std::vector<cv::Mat> &_imgs,
               const std::vector<std::vector<Eigen::Vector2d>> &_keypoints,
               const std::vector<std::vector<Eigen::Vector3d>> &_bearings,
               const std::vector<cv::Mat> &_descriptors = {});
              
  void extractFeature(const std::vector<cv::Mat> &_imgs, std::string detector_type = "ORB");

  cv::Mat drawKeyPoint(const int &cam_id);

  cv::Mat drawMatchedKeyPoint(const int &cam_id);

public: // frame id
  FrameIDType id_;
  int cam_num_;
  Eigen::Matrix4d Twb_ ;
  std::vector<Eigen::Matrix4d> Tcw_;

  std::vector<cv::Mat> imgs_;
  std::vector<std::vector<Eigen::Vector2d>> keypoints_;
  std::vector<std::vector<Eigen::Vector3d>> bearings_;
  std::vector<cv::Mat> descriptors_;
  std::vector<std::vector<FeatureIDType>> feature_ids_;
};
typedef std::unordered_map<FrameIDType, Frame::Ptr> FrameMap;
