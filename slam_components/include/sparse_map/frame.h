#pragma once

#include "sparse_map/common.h"

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frame>;

public:
  Frame(FrameIDType _id, int _cam_num);

  virtual ~Frame() {}

  FrameIDType id();

  int camNum();

  const std::vector<cv::Mat> &imgs() { return imgs_; }

  const std::vector<std::vector<Eigen::Vector2d>> &keypoints() const {
    return keypoints_;
  }

  const std::vector<std::vector<Eigen::Vector3d>> &bearings() const {
    return bearings_;
  }

  const std::vector<cv::Mat> &descriptors() const { return descriptors_; }

  const std::vector<std::vector<FeatureIDType>> &feature_ids() {
    return feature_ids_;
  }

  void setFeatureID(const int &cam_id, const int &pt_id,
                    const FeatureIDType &ft_id);

  double *getRotaionParams();

  double *getTranslationParams();

  Eigen::Matrix4d getBodyPose();

  void setBodyPose(const Eigen::Matrix4d &Twb);

  void addData(const std::vector<cv::Mat> &_imgs = {},
               const std::vector<std::vector<Eigen::Vector2d>> &_keypoints = {},
               const std::vector<std::vector<Eigen::Vector3d>> &_bearings = {},
               const std::vector<cv::Mat> &_descriptors = {});

  void extractFeature(const std::vector<cv::Mat> &_imgs,
                      std::string detector_type = "ORB",
                      const std::vector<cv::Mat> &_masks = {});

  cv::Mat drawKeyPoint(const int &cam_id);

  cv::Mat drawMatchedKeyPoint(const int &cam_id);

  cv::Mat drawReprojKeyPoint(const int &cam_id);

public:
  Eigen::Matrix4d Twb_prior_;
  std::vector<Eigen::Matrix4d> Tcw_;
  std::vector<std::set<std::pair<FrameIDType, int>>> matched_frames_;

private:
  FrameIDType id_;
  std::vector<cv::Mat> imgs_;
  std::vector<std::vector<Eigen::Vector2d>> keypoints_;
  std::vector<std::vector<Eigen::Vector3d>> bearings_;
  std::vector<cv::Mat> descriptors_;
  std::vector<std::vector<FeatureIDType>> feature_ids_;

  // Pose
  double pose_q[4]; // x, y, z, w (Rotation)
  double pose_t[3]; // x, y, z (Translation)
};
typedef std::map<FrameIDType, Frame::Ptr> FrameMap;