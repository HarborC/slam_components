#pragma once

#include "sparse_map/common.h"

class Frame {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Frame>;

public:
  Frame(FrameIDType _id) : id_(_id) {}
  virtual ~Frame() {}

  Eigen::Matrix4d getBodyPose() {
    Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d Rwb = qwb_.toRotationMatrix();
    Twb.block<3, 3>(0, 0) = Rwb;
    Twb.block<3, 1>(0, 3) = twb_;

    return Twb;
  }

  void setBodyPose(const Eigen::Matrix4d &Twb) {
    qwb_ = Eigen::Quaterniond(Twb.block<3, 3>(0, 0));
    twb_ = Twb.block<3, 1>(0, 3);
    pose_q[0] = qwb_.x();
    pose_q[1] = qwb_.y();
    pose_q[2] = qwb_.z();
    pose_q[3] = qwb_.w();
  }

  Eigen::Matrix4d getCameraPose(const int &cam_id) {
    Eigen::Matrix4d Tcw = Tcw_[cam_id];
    return Tcw;
  }

  void addData(const std::vector<cv::Mat> &_imgs,
               const std::vector<std::vector<Eigen::Vector2d>> &_keypoints,
               const std::vector<std::vector<Eigen::Vector3d>> &_bearings,
               const std::vector<cv::Mat> &_descriptors = {});
              
  void extractFeature(const std::vector<cv::Mat> &_imgs, std::string detector_type = "ORB",
                      const std::vector<cv::Mat> &_masks = {});

  cv::Mat drawKeyPoint(const int &cam_id);

  cv::Mat drawMatchedKeyPoint(const int &cam_id);

  cv::Mat drawReprojKeyPoint(const int &cam_id);

public: // frame id
  FrameIDType id_;
  int cam_num_;
  Eigen::Matrix4d Twb_prior_;
  Eigen::Quaterniond qwb_;
  Eigen::Vector3d twb_;
  double pose_q[4];
  std::vector<Eigen::Matrix4d> Tcw_;

  std::vector<cv::Mat> imgs_;
  std::vector<std::vector<Eigen::Vector2d>> keypoints_;
  std::vector<std::vector<Eigen::Vector3d>> bearings_;
  std::vector<cv::Mat> descriptors_;
  std::vector<std::vector<FeatureIDType>> feature_ids_;
};
typedef std::map<FrameIDType, Frame::Ptr> FrameMap;
